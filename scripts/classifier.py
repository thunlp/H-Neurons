import os
import json
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train H-Neuron detector with flexible directory inputs.")
    # Model config
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model for config")
    
    # Training inputs
    parser.add_argument("--train_ids", type=str, help="Path to train_qids.json")
    parser.add_argument("--train_ans_acts", type=str, help="Directory of answer tokens activations for training")
    parser.add_argument("--train_other_acts", type=str, help="Directory of other tokens activations (required for 3-vs-1)")
    
    # Testing inputs
    parser.add_argument("--test_ids", type=str, help="Path to test_qids.json")
    parser.add_argument("--test_acts", type=str, help="Directory of activations to evaluate (usually answer_tokens)")
    
    # Model Persistence
    parser.add_argument("--save_model", type=str, default="models/detector.pkl")
    parser.add_argument("--load_model", type=str, help="Load a pre-trained model for evaluation")
    
    # Training Parameters
    parser.add_argument("--train_mode", type=str, choices=["1-vs-1", "3-vs-1"], default="3-vs-1")
    parser.add_argument("--penalty", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="liblinear") # change to saga if needed
    
    return parser.parse_args()

def load_data(ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1"):
    """
    Flexible data loader.
    1-vs-1: False Answer Tokens (Label 1) vs True Answer Tokens (Label 0).
    3-vs-1: False Answer Tokens (Label 1) vs (True Ans + True Other + False Other) (Label 0).
    """
    with open(ids_path, "r") as f:
        id_map = json.load(f)
    
    X, y = [], []

    # 1. Load False Answer Tokens -> Always Label 1 (Positive)
    for qid in tqdm(id_map["f"], desc="Loading False Ans (Label 1)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(1)

    # 2. Load True Answer Tokens -> Always Label 0 (Negative)
    for qid in tqdm(id_map["t"], desc="Loading True Ans (Label 0)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(0)

    # 3. Load Other Tokens if 3-vs-1 mode is enabled
    if mode == "3-vs-1":
        if not other_acts_dir:
            raise ValueError("train_other_acts directory is required for 3-vs-1 mode.")
        
        for label_key in ["t", "f"]:
            for qid in tqdm(id_map[label_key], desc=f"Loading Other Tokens - {label_key} (Label 0)"):
                path = os.path.join(other_acts_dir, f"act_{qid}.npy")
                if os.path.exists(path):
                    X.append(np.load(path).flatten())
                    y.append(0)

    return np.array(X), np.array(y)

def run_evaluation(model, X, y, dataset_name="Test"):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
    auroc = roc_auc_score(y, probs)
    
    print(f"\n--- Results: {dataset_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUROC:     {auroc:.4f}")

def main():
    args = parse_args()
    model = None

    # --- Training Phase ---
    if args.load_model:
        print(f"Loading pre-trained model: {args.load_model}")
        model = joblib.load(args.load_model)
    elif args.train_ids and args.train_ans_acts:
        print(f"Training in {args.train_mode} mode...")
        X_train, y_train = load_data(
            args.train_ids, 
            args.train_ans_acts, 
            other_acts_dir=args.train_other_acts, 
            mode=args.train_mode
        )
        
        model = LogisticRegression(
            penalty=args.penalty, C=args.C, solver=args.solver, 
            max_iter=1000, random_state=42, verbose=1
        )
        model.fit(X_train, y_train)
        
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            joblib.dump(model, args.save_model)
            print(f"Model saved. Identified {np.sum(model.coef_[0] > 0)} potential H-Neurons.")
            
        run_evaluation(model, X_train, y_train, "Training Set")
    else:
        print("Please provide --load_model OR (--train_ids AND --train_ans_acts)")
        return

    # --- Evaluation Phase ---
    if args.test_ids and args.test_acts:
        print(f"Evaluating on {args.test_ids}...")
        # Evaluation is typically 1-vs-1 (True Ans vs False Ans)
        X_test, y_test = load_data(args.test_ids, args.test_acts, mode="1-vs-1")
        run_evaluation(model, X_test, y_test, "Test Set")

if __name__ == "__main__":
    main()