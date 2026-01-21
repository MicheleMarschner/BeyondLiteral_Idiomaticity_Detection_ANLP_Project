import argparse
# from src.preprocessing import clean_data

def main():
    parser = argparse.ArgumentParser(description="NLP Project: Sarcasm Detection")
    parser.add_argument("--data", default="data/raw/test.csv", help="Path to test data")
    parser.add_argument("--model_path", default="models/best_model.pt", help="Path to best weights")
    args = parser.parse_args()

    print(f"--- Starting Reproducibility Script ---")
    
    # 1. Preprocessing
    print("Preprocessing test data...")
    test_df = clean_data(args.data)
    
    # 2. Load the "Submission" Model
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)
    
    # 3. Evaluation
    print("Running evaluation metrics...")
    results = run_evaluation(model, test_df)
    
    # 4. Final Output
    print("\nFINAL RESULTS:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()