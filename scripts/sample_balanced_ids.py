import json
import argparse
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sample balanced True and False IDs for training.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to extraction results (jsonl)")
    parser.add_argument("--output_path", type=str, default="data/train_qids.json", help="Path to save balanced IDs (json)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples per class (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    true_ids = []
    false_ids = []

    # Categorize IDs based on labels
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading IDs"):
            try:
                data = json.loads(line)
                qid = list(data.keys())[0]
                label = data[qid].get("judge")

                if label == "true":
                    true_ids.append(qid)
                elif label == "false":
                    false_ids.append(qid)
            except Exception as e:
                print(f"Skipping line due to error: {e}")
 
    print(f"Total available - True: {len(true_ids)}, False: {len(false_ids)}")

    # Determine final sample count based on availability
    actual_samples = min(args.num_samples, len(true_ids), len(false_ids))
    if actual_samples < args.num_samples:
        print(f"Warning: Only {actual_samples} samples per class available. Sampling maximum possible.")

    # Randomly sample equal amounts
    sampled_t = random.sample(true_ids, actual_samples)
    sampled_f = random.sample(false_ids, actual_samples)

    output_data = {
        "t": sampled_t,
        "f": sampled_f
    }

    # Save to JSON
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully saved {actual_samples * 2} balanced IDs to {args.output_path}")

if __name__ == "__main__":
    main()