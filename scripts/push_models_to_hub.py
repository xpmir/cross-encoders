import torch
import shutil
import os
import argparse
import logging
import re
from pathlib import Path
from huggingface_hub import HfApi, login
from transformers import AutoConfig, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def extract_metadata(card_path):
    """Extract base_model and model_name from the markdown card frontmatter."""
    content = card_path.read_text()
    base_model_match = re.search(r"base_model:\s*([^\s\n]+)", content)
    model_name_match = re.search(r"model_name:\s*([^\s\n]+)", content)
    
    base_model = base_model_match.group(1) if base_model_match else None
    # Use the filename-friendly version if model_name is not found
    model_id = model_name_match.group(1) if model_name_match else None
    
    return base_model, model_id

def upload_full_package(repo_id, base_model, weights_path, card_path, logs_dir, config_path, token, collection_slug=None):
    api = HfApi()
    
    # 1. Create the repo (if it doesn't exist)
    logger.info(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    
    # 2. Prepare the model weights/config locally for a clean upload
    temp_save_path = Path("./temp_model_upload")
    if temp_save_path.exists():
        shutil.rmtree(temp_save_path)
    temp_save_path.mkdir(parents=True)
    
    logger.info(f"Preparing model files for {repo_id} using base {base_model}...")
    try:
        config = AutoConfig.from_pretrained(base_model)

        # ensure that num_labels is one for a Cross-encoder
        if hasattr(config, "num_labels"):
            config.num_labels = 1
        else:
            logger.warning(
                "no 'num_labels param found in config, check that classifier outputs one label"
            )

        model = AutoModelForSequenceClassification.from_config(config)
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {key[6:]: value for key, value in state_dict.items() if not key.startswith("_param")}
        
        model.load_state_dict(state_dict)
        
        logger.info("successfully loaded checkpoint")
        # Save locally first to get the correct structure
        model.save_pretrained(temp_save_path)
        
        # 3. Upload the model files (bin/safetensors + config)
        logger.info(f"Uploading model files to {repo_id}...")
        api.upload_folder(
            folder_path=str(temp_save_path),
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        
        # 4. Upload the TensorBoard logs to a 'runs' folder in the repo
        if logs_dir and logs_dir.exists():
            logger.info(f"Uploading logs from {logs_dir}...")
            api.upload_folder(
                folder_path=str(logs_dir),
                path_in_repo="runs", 
                repo_id=repo_id,
                token=token
            )
        else:
            logger.warning(f"Logs directory {logs_dir} not found, skipping...")
        
        # 5. Upload the config.yaml if it exists
        if config_path and config_path.exists():
            logger.info(f"Uploading config.yaml to {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.yaml",
                repo_id=repo_id,
                token=token
            )
        else:
            logger.warning(f"Config file {config_path} not found, skipping...")

        # 6. Upload the Model Card as README.md
        logger.info(f"Uploading model card to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token
        )

        # 7. Add to collection if specified
        if collection_slug:
            try:
                # Ensure the collection slug includes the namespace if not already present
                full_collection_slug = collection_slug
                if "/" not in collection_slug:
                    namespace = repo_id.split("/")[0]
                    full_collection_slug = f"{namespace}/{collection_slug}"
                
                logger.info(f"Adding model to collection: {full_collection_slug}")
                api.add_collection_item(
                    collection_slug=full_collection_slug,
                    item_id=repo_id,
                    item_type="model",
                    token=token
                )
            except Exception as e:
                logger.warning(f"Failed to add to collection {full_collection_slug}: {e}")

        logger.info(f"Success! View your metrics at https://huggingface.co/{repo_id}/tensorboard")
    
    except Exception as e:
        logger.error(f"Failed to upload {repo_id}: {e}")
    finally:
        if temp_save_path.exists():
            shutil.rmtree(temp_save_path)

def main():
    parser = argparse.ArgumentParser(description="Push models from results directory to Hugging Face Hub")
    parser.add_argument("results_dir", type=str, help="Path to the experiment results directory (containing 'models' folder)")
    parser.add_argument("--namespace", type=str, default="xpmir", help="HF username or organization namespace")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="HF API token")
    parser.add_argument("--collection", type=str, default="reproducing-cross-encoders", help="HF collection slug or name")
    parser.add_argument("--include", type=str, default="*", help="Glob pattern to filter model folders (e.g., '*electra*')")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    models_dir = results_path / "models"
    
    if not models_dir.exists():
        logger.error(f"Models directory not found at {models_dir}")
        return
    token = args.token

    if not token:
        logger.error("Hugging Face token not provided. Use --token or set HUGGING_FACE_HUB_TOKEN env var.")
        return

    # Iterate over best model folders matching the pattern
    matched_folders = list(models_dir.glob(args.include))
    logger.info(f"Found {len(matched_folders)} folders matching pattern '{args.include}'")

    for model_folder in matched_folders:
        if not model_folder.is_dir():
            continue
            
        logger.info(f"Processing model in {model_folder.name}...")
        
        weights_path = model_folder / "model_weights.pt"
        card_path = model_folder / "README.md"
        logs_dir = model_folder / "tensorboard_logs"
        config_path = model_folder / "config.yaml"
        
        if not weights_path.exists() or not card_path.exists():
            logger.warning(f"Missing model.pth or README.md in {model_folder}, skipping...")
            continue
            
        base_model, model_id_from_card = extract_metadata(card_path)
        
        if not base_model:
            logger.error(f"Could not find base_model in {card_path}, skipping...")
            continue
            
        # Construct the full HF repo ID
        # We use the model_id from card if available, otherwise sanitize the folder name
        model_name = model_folder.name
        repo_id = f"{args.namespace}/{model_name}"
        
        upload_full_package(
            repo_id=repo_id,
            base_model=base_model,
            weights_path=weights_path,
            card_path=card_path,
            logs_dir=logs_dir,
            config_path=config_path,
            token=token,
            collection_slug=args.collection
        )

if __name__ == "__main__":
    main()
