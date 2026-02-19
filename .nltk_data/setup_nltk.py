from pathlib import Path
import nltk

def main():
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / ".nltk_data"
    target.mkdir(parents=True, exist_ok=True)

    nltk.download("wordnet", download_dir=str(target))
    nltk.download("omw-1.4", download_dir=str(target))
    print("NLTK data downloaded to:", target)

if __name__ == "__main__":
    main()
