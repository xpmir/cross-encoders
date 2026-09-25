import torch
from sentence_transformers import CrossEncoder

model_name = "cross-encoder/ettin-reranker-68m-v1"

# Flash Attention requires fp16 or bf16
model_kwargs = {
    "attn_implementation": "flash_attention_2",
    "torch_dtype": torch.float16
}

print(f"Loading {model_name} with Flash Attention 2...")

try:
    # 1. Load the model
    model = CrossEncoder(
        model_name,
        model_kwargs=model_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Verify the config
    # sentence-transformers wraps the Hugging Face model in `model.model`
    hf_model = model.model
    attn_impl = getattr(hf_model.config, "_attn_implementation", "unknown")

    if attn_impl == "flash_attention_2":
        print("✅ Success: Flash Attention 2 is active in the model config.")
    else:
        print(f"⚠️ Warning: Model loaded, but attention implementation is '{attn_impl}'.")

    # 3. Run a test pass to verify it executes without errors
    print("Running a forward pass...")
    scores = model.predict([
        ("How do I install PyTorch?", "You can install PyTorch using pip or conda."),
        ("How do I install PyTorch?", "The mitochondria is the powerhouse of the cell.")
    ])

    print("✅ Inference successful!")
    print(f"Scores: {scores}")

except ImportError as e:
    print(f"❌ Import Error: {e}. Make sure you have 'flash-attn' installed.")
except Exception as e:
    print(f"❌ Error: {e}")
