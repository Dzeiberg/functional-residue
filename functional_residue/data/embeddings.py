from pathlib import Path
import h5py
import numpy as np
from Bio import SeqIO
import gzip
import datetime
import time
from typing import Dict
import torch
from transformers import T5EncoderModel, T5Tokenizer


class EmbeddingSet(object):
    def __init__(self, **kwargs):
        """
        Create an EmbeddingSet object

        Required Args:
        None

        Optional Args:
        - ids (list): a list of sequence IDs
        - embeddings (np.ndarray): a 2D array of embeddings
        - sequences (list): a list of sequences
        - embedding_filepath (str): the filepath to save the embeddings
        None
        """
        self.embedding_filepath = kwargs.pop("embedding_filepath", None)
        self._init_data(**kwargs)

    def _init_data(self, ids=[], embeddings=np.array([]), sequences=[]):
        self.id_to_position = {id: i for i, id in enumerate(ids)}
        self.ids = set(ids)
        self.embeddings = embeddings
        self.embeddings = self.embeddings.reshape((-1, 1024))
        self.sequence_to_position = {seq: i for i, seq in enumerate(sequences)}
        self.sequences = set(sequences)

    @classmethod
    def from_file(cls, embedding_filepath):
        raise NotImplementedError(
            "Loading from file is not implemented. Use the constructor instead. Seems to load buggy data."
        )
        with h5py.File(embedding_filepath, "r") as f:
            data = {
                "ids": list(f["ids"][...]),
                "embeddings": np.array(f["embeddings"][...]),
                "sequences": list(f["sequences"][...]),
                "embedding_filepath": embedding_filepath,
            }
        # decode byte strings
        data["ids"] = [id.decode() for id in data["ids"]]
        data["sequences"] = [seq.decode() for seq in data["sequences"]]
        return cls(**data)

    def to_file(self, embedding_filepath):
        raise NotImplementedError(
            "Saving to file is not implemented. Use the constructor instead. Seems to save buggy data."
        )
        with h5py.File(embedding_filepath, "w") as f:
            f.create_dataset("ids", data=[_id.encode() for _id in self.ids])
            f.create_dataset("embeddings", data=self.embeddings)
            f.create_dataset("sequences", data=[seq.encode() for seq in self.sequences])

    def __len__(self):
        return len(self.ids)

    def get_embedding(self, sequence=None, _id=None) -> np.ndarray:
        """
        Get the embedding for a sequence or ID. Sequence or ID must be provided.
        First queries by sequence, then by ID.
        If neither are found, returns an array of NaNs.

        Args:
        sequence (str): the sequence
        _id (str): the ID of the sequence

        Returns:
        np.ndarray: the embedding
        """
        if sequence is None and _id is None:
            raise ValueError("Either sequence or _id must be provided")
        if sequence is not None and sequence in self.sequences:
            return self.embeddings[self.sequence_to_position[sequence]]
        if _id is not None and _id in self.ids:
            return self.embeddings[self.id_to_position[_id]]
        return np.ones(1024) * np.nan

    def get_many_embeddings(self, sequences=None, ids=None) -> np.ndarray:
        """
        Get the embeddings for many sequences or IDs. Sequences or IDs must be provided.
        First queries by sequence, then by ID.
        If neither are found, generate embedding and add to EmbeddingSet.

        Args:
        sequences (list): a list of sequences
        ids (list): a list of IDs

        Returns:
        np.ndarray: the embeddings
        """
        embeddings = np.zeros((len(sequences), 1024))
        # 1) get embeddings for sequences already in EmbeddingSet
        for i, (seq, _id) in enumerate(zip(sequences, ids)):
            embeddings[i] = self.get_embedding(sequence=seq, _id=_id)
        # 2) generate embeddings for sequences not in EmbeddingSet
        missing_indices = np.where(np.isnan(embeddings).any(axis=1))[0]
        if len(missing_indices) == 0:
            return embeddings
        missing_sequences = [sequences[i] for i in missing_indices]
        missing_ids = [ids[i] for i in missing_indices]
        seq_path = Path("/tmp/test.fasta")
        emb_path = Path("/tmp/test.h5")
        with open(seq_path, "w") as f:
            for seq, _id in zip(missing_sequences, missing_ids):
                f.write(f">{_id}\n{seq}\n")
        get_embeddings(seq_path, emb_path, per_protein=True, model_dir=None)
        with h5py.File(emb_path, "r") as f:
            generated_embeddings = np.array([f[_id][...] for _id in missing_ids])
        # 3) add generated embeddings to EmbeddingSet
        self.id_to_position.update(
            {id: i for i, id in enumerate(missing_ids, start=len(self.ids))}
        )
        self.sequence_to_position.update(
            {
                seq: i
                for i, seq in enumerate(missing_sequences, start=len(self.sequences))
            }
        )
        self.ids.update(missing_ids)
        self.embeddings = np.vstack([self.embeddings, generated_embeddings])
        self.sequences.update(missing_sequences)
        # 4) remove temporary files
        seq_path.unlink()
        emb_path.unlink()
        # 5) write to file
        if self.embedding_filepath is not None:
            self.to_file(self.embedding_filepath)
        # 6) update embeddings array
        embeddings[missing_indices] = generated_embeddings
        return embeddings


def timestamp():
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return current_timestamp


def parse_fasta(fasta_filepath):
    fasta_filepath = Path(fasta_filepath)
    gzipped = fasta_filepath.suffix == ".gz"
    if gzipped:
        with gzip.open(fasta_filepath, "rt") as fasta_f:
            for record in SeqIO.parse(fasta_f, "fasta"):
                yield record
    else:
        for record in SeqIO.parse(fasta_filepath, "fasta"):
            yield record


"""
REMAINING CODE:

Source : https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger
"""


def get_T5_model(
    model_dir=None, device=None, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"
):
    print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path) -> Dict[str, str]:
    """
    Read in fasta file containing multiple sequences

    Return Dict mapping sequence id to sequence

    Args:
    fasta_path (str): path to fasta file
    compressed (bool): whether the file is compressed with gzip (default: False)

    Returns:
    Dict[str,str]: mapping of sequence id to sequence
    """
    fasta_path = Path(fasta_path)
    compressed = False
    if fasta_path.suffix == ".gz":
        compressed = True
    records = dict()
    if compressed:
        with gzip.open(fasta_path, "rt") as fasta_f:
            for record in SeqIO.parse(fasta_f, "fasta"):
                records[record.id] = str(record.seq)
    else:
        for record in SeqIO.parse(fasta_path, "fasta"):
            records[record.id] = str(record.seq)
    return records


def get_embeddings(
    seq_path,
    emb_path,
    per_protein,  # whether to derive per-protein (mean-pooled) embeddings
    model_dir=None,
    max_residues=4000,  # number of cumulative residues per batch
    max_seq_len=1000,  # max length after which we switch to single-sequence processing to avoid OOM
    max_batch=100,  # max number of sequences per single batch
):
    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict = read_fasta(seq_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    model, vocab = get_T5_model(model_dir, device)

    print("########################################")
    print(
        "Example sequence: {}\n{}".format(
            next(iter(seq_dict.keys())), next(iter(seq_dict.values()))
        )
    )
    print("########################################")
    print("Total number of sequences: {}".format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict = sorted(
        seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True
    )

    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        seq_len = len(seq)
        seq = " ".join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == len(seq_dict)
            or seq_len > max_seq_len
        ):
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(
                seqs, add_special_tokens=True, padding="longest"
            )
            input_ids = torch.tensor(token_encoding["input_ids"]).to(device)
            attention_mask = torch.tensor(token_encoding["attention_mask"]).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print(
                    "RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(
                        pdb_id, seq_len
                    )
                    + "If single sequence processing does not work, you need more vRAM to process your protein."
                )
                continue

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                if per_protein:
                    emb = emb.mean(dim=0)

                if len(emb_dict) == 0:
                    print(
                        "Embedded protein {} with length {} to emb. of shape: {}".format(
                            identifier, s_len, emb.shape
                        )
                    )

                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    end = time.time()

    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)

    print("\n############# STATS #############")
    print("Total number of embeddings: {}".format(len(emb_dict)))
    print(
        "Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}".format(
            end - start, (end - start) / len(emb_dict), avg_length
        )
    )
    return True
