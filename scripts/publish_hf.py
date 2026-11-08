"""Publish the trained project to Hugging Face: a Docker Space (dashboard + API) and a model repo (artifacts + card).

    pip install huggingface_hub
    # create a WRITE token at https://huggingface.co/settings/tokens, then in the SAME terminal:
    #   PowerShell:  $env:HF_TOKEN="hf_xxx"        (or run:  hf auth login)
    python scripts/publish_hf.py --dry-run            # builds the folders and shows what would be uploaded
    python scripts/publish_hf.py                      # uploads (username is read from your token)

Never paste your token into chat, code or git. Both repos are created PUBLIC by default; use --private to change that.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="HF username or org (default: owner of the token)")
    ap.add_argument("--space-name", default="sentinel-fraud-detection")
    ap.add_argument("--model-name", default="sentinel-fraud-xgboost")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-space", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for script in ("make_hf_space.py", "make_hf_model_repo.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

    if a.dry_run:
        user = a.user or "<your-username>"
    else:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit("Run: pip install huggingface_hub")
        api = HfApi()
        try:
            user = a.user or api.whoami()["name"]
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Not logged in to Hugging Face ({exc.__class__.__name__}). Set HF_TOKEN or run: hf auth login")

    plan = []
    if not a.skip_space:
        plan.append(("space", f"{user}/{a.space_name}", ROOT / "hf_space"))
    if not a.skip_model:
        plan.append(("model", f"{user}/{a.model_name}", ROOT / "hf_model"))

    for kind, repo_id, folder in plan:
        print(f"{'[dry-run] would upload' if a.dry_run else 'Uploading'} {folder.name}/ -> {kind} {repo_id}")
        if a.dry_run:
            continue
        kwargs = {"space_sdk": "docker"} if kind == "space" else {}
        api.create_repo(repo_id, repo_type=kind, private=a.private, exist_ok=True, **kwargs)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=kind,
                          commit_message="Publish SENTINEL")
        url = f"https://huggingface.co/{'spaces/' if kind == 'space' else ''}{repo_id}"
        print(f"  done: {url}")
    if not a.dry_run and not a.skip_space:
        print("The Space now builds its Docker image (a few minutes). Watch the 'Logs' tab on the Space page.")


if __name__ == "__main__":
    main()
