from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('eleutherai_models', 'gpt-neo-1.3B')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="EleutherAI/gpt-neo-1.3B",
    local_dir=mistral_models_path
)

