from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

api = HfApi()

template = Path("src/MICE/modeling/mice_v2/Micev2Card.md").read_text(encoding="utf-8")

configs = [
    {
        "repo_id": "VictorMorand/Mice-P-61m",
        "short_name": "Mice-P-61m",
        "base": "jhu-clsp/ettin-encoder-68m",
        "total_parameters": "61,328,385 (61.33M)",
        "n_ctx": "10",
        "n_inter": "4",
    },
    {
        "repo_id": "VictorMorand/Mice-P-130m",
        "short_name": "Mice-P-130m",
        "base": "cross-encoder/ettin-reranker-150m-v1",
        "total_parameters": "128,564,737 (128.56M)",
        "n_ctx": "12",
        "n_inter": "4",
    },
]

for cfg in configs:
    repo = cfg["repo_id"]
    old_readme = Path(hf_hub_download(repo, "README.md")).read_text(encoding="utf-8")

    # Extract the results table
    results_idx = old_readme.find("| dataset")
    if results_idx == -1:
        raise ValueError(f"Could not find results table in {repo} README.md")
    table_str = old_readme[results_idx:].strip()

    # Render template
    rendered = template
    rendered = rendered.replace("{{base}}", cfg["base"])
    rendered = rendered.replace("{{model_id}}", cfg["repo_id"])
    rendered = rendered.replace("{{total_parameters}}", cfg["total_parameters"])
    rendered = rendered.replace("{{n_contextualization_layers}}", cfg["n_ctx"])
    rendered = rendered.replace("{{n_interaction_layers}}", cfg["n_inter"])

    # Adjust title to short name
    rendered = rendered.replace(f"# {cfg['repo_id']}", f"# {cfg['short_name']}")

    eval_section = (
        "## Evaluations\n\n"
        "We provide evaluations of this cross-encoder re-ranking top 100 of `naver/splade-v3-distilbert`:\n\n"
        f"{table_str}\n"
    )

    full_readme = rendered.rstrip() + "\n\n" + eval_section

    print(f"Uploading new README.md to {repo}...")
    api.upload_file(
        path_or_fileobj=full_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        commit_message="Update README.md with new model card and top 100 splade-v3-distilbert evaluations",
    )
    print(f"Successfully updated README.md for {repo}!")
