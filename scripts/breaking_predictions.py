import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from functional_residue.inference import InferencePipeline


def main(structures_dir: str, predictions_dir: str, model_checkpoints_dir: str) -> None:
    structures_dir = Path(structures_dir)  # type: ignore
    if not structures_dir.exists():  # type: ignore
        raise FileNotFoundError(
            f"Structures directory {structures_dir} does not exist."
        )
    predictions_dir = Path(predictions_dir)  # type: ignore
    predictions_dir.mkdir(parents=True, exist_ok=True)  # type: ignore
    # Get the list of PDB files in the structures directory
    pdb_files = list(structures_dir.glob("*.pdb"))  # type: ignore
    model_checkpoints_dir = Path(model_checkpoints_dir)  # type: ignore
    if not model_checkpoints_dir.exists():  # type: ignore
        raise FileNotFoundError(
            f"Model checkpoints directory {model_checkpoints_dir} does not exist."
        )
    # Get the list of model checkpoints
    model_checkpoint_filepaths = list(model_checkpoints_dir.glob("*.pt"))  # type: ignore
    if len(model_checkpoint_filepaths) == 0:  # type: ignore
        raise FileNotFoundError(
            f"No model checkpoints found in {model_checkpoints_dir}."
        )
    # Create the inference pipeline
    inference_pipeline = InferencePipeline(model_checkpoint_filepaths=model_checkpoint_filepaths)  # type: ignore
    # Iterate over the PDB files and run inference
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        prediction = inference_pipeline.predict(pdb_file, "A")  # type: ignore
        # Save the prediction to the predictions directory
        prediction_file = predictions_dir / (pdb_file.stem + "_prediction.npy")
        # Save the prediction as a numpy array
        np.save(prediction_file, prediction)


if __name__ == "__main__":
    main("data/breaking/structures", "data/breaking/predictions", "data/cat_models")
    # import argparse
    # parser = argparse.ArgumentParser(description="Run inference on PDB files using a trained GAT model.")
    # parser.add_argument("--structures_dir", type=str, required=True, help="Directory containing the PDB files.")
    # parser.add_argument("--predictions_dir", type=str, required=True, help="Directory to save the predictions.")
    # parser.add_argument("--model_checkpoints_dir", type=str, required=True, help="Directory containing the model checkpoints.")
    # args = parser.parse_args()
    # main(args.structures_dir, args.predictions_dir, args.model_checkpoints_dir)
    # Run the script with the following command:
    # python scripts/breaking_predictions.py --structures_dir /path/to/structures --predictions_dir /path/to/predictions --model_checkpoints_dir /path/to/model_checkpoints
    # Example:
    # python scripts/breaking_predictions.py --structures_dir data/breaking/structures --predictions_dir data/breaking/predictions --model_checkpoints_dirdata/cat_models
