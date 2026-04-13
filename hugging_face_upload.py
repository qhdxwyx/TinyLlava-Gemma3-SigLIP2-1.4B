import argparse

from huggingface_hub import create_repo, upload_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a local model folder to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target Hub repo, for example user/model-name.")
    parser.add_argument("--folder-path", type=str, required=True, help="Local model folder to upload.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private before upload.")
    return parser.parse_args()


def main():
    args = parse_args()
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    upload_folder(folder_path=args.folder_path, repo_id=args.repo_id, repo_type="model")


if __name__ == "__main__":
    main()
