import os
import json
import re
import string
import argparse
import time
from typing import List, Set, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI  # 用于 LLM Judge

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Filtering with Rule or LLM Judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TriviaQA parquet file")
    parser.add_argument("--output_path", type=str, default="data/consistency_samples.jsonl", help="Output path")
    
    parser.add_argument("--sample_num", type=int, default=10, help="Samples per question")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--gpu_util", type=float, default=0.7, help="vLLM GPU memory utilization")
    parser.add_argument("--tp_size", type=int, default=None, help="Tensor parallel size")

    parser.add_argument("--judge_type", type=str, choices=["rule", "llm"], default="rule", help="How to judge correctness")
    parser.add_argument("--api_key", type=str, default=None, help="API key for LLM Judge")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge")
    
    return parser.parse_args()

# ==========================================
# Utilities
# ==========================================

def normalize_answer(s: str) -> str:
    """Standardize answer strings for Rule Judge."""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "‘’´`")
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    if not s: return ""
    return white_space_fix(remove_articles(handle_punc(str(s).lower().replace('_', ' ')))).strip()

def load_existing_qids(path: str) -> Set[str]:
    if not os.path.exists(path): return set()
    qids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except: continue
    return qids

# ==========================================
# Consistency Sampler with Multi-Judge Support
# ==========================================

class ConsistencySampler:
    def __init__(self, args):
        self.args = args
        
        # 1. Init Sampling LLM (vLLM)
        self.tp_size = args.tp_size or torch.cuda.device_count()

        self.sampling_llm = LLM(
            model=args.model_path,
            tensor_parallel_size=self.tp_size,
            gpu_memory_utilization=args.gpu_util,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=1.0, top_p=0.9, top_k=50, max_tokens=50
        )

        # 2. Init Judge Client (if needed)
        self.judge_client = None
        if args.judge_type == "llm":
            if not args.api_key:
                raise ValueError("API Key is required for LLM Judge.")
            self.judge_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def rule_judge(self, response: str, norm_gts: List[str]) -> str:
        """Simple string matching judge."""
        norm_res = normalize_answer(response)
        for gt in norm_gts:
            if gt and gt in norm_res:
                return "true"
        return "false"

    def llm_judge(self, question: str, response: str, answer_list: List[str]) -> str:
        """LLM based judge with retry logic."""
        prompt = (
            f"Question: {question}\n"
            f"Response: {response}\n"
            f"Correct Answers: {answer_list}\n"
            f"Please judge whether the response is correct or not. "
            f"Return 't' if the response is correct, and 'f' if the response is incorrect. "
            f"Don't add any additional information."
        )
        
        for attempt in range(5):
            try:
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                res = completion.choices[0].message.content.strip().lower()
                if 't' in res: return "true"
                if 'f' in res: return "false"
            except Exception as e:
                print(f"Judge API failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return "error"

    def process_data(self):
        dataset = load_dataset("parquet", data_files=self.args.data_path, split="train")
        if self.args.max_samples:
            dataset = dataset.select(range(self.args.max_samples))
        processed_qids = load_existing_qids(self.args.output_path)
        
        all_correct_count = 0
        all_incorrect_count = 0

        with open(self.args.output_path, 'a', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Sampling ({self.args.judge_type} judge)"):
                qid = str(item.get('question_id', ''))
                if qid in processed_qids: continue

                question = item.get('question', '')
                if not question or 'answer' not in item: continue

                # Get ground truth
                raw_aliases = []
                for col in ['aliases', 'normalized_aliases']:
                    val = item['answer'].get(col)
                    if val: raw_aliases.extend(val) if isinstance(val, list) else raw_aliases.append(str(val))
                
                norm_gts = [normalize_answer(a) for a in set(raw_aliases) if a]
                if not norm_gts: continue

                suffix = "Respond with the answer only, without any explanation."
                # Sampling
                messages = [{"role": "user", "content": f"{question.strip()} {suffix}"}]
                responses = []
                judges = []
                
                # Cache for LLM judge to avoid redundant API calls for the same response in 10 samples
                judge_cache = {}

                for _ in range(self.args.sample_num):
                    try:
                        outputs = self.sampling_llm.chat(messages, self.sampling_params, use_tqdm=False)
                        ans = outputs[0].outputs[0].text.strip()
                        responses.append(ans)

                        # 1. Uncertainty check (Rule-based pre-filter)
                        uncertain_terms = ["don't know", "cannot", "not provided", "no information"]
                        if any(term in ans.lower() for term in uncertain_terms):
                            judges.append("uncertain")
                            continue

                        # 2. Correctness check
                        if self.args.judge_type == "rule":
                            judges.append(self.rule_judge(ans, norm_gts))
                        else:
                            # Use cache to save tokens if model repeats the same answer
                            if ans not in judge_cache:
                                judge_cache[ans] = self.llm_judge(question, ans, raw_aliases)
                            judges.append(judge_cache[ans])

                    except Exception as e:
                        print(f"Sampling error at {qid}: {e}")
                        break

                if len(responses) < self.args.sample_num: continue

                # Stats update
                true_count = judges.count("true")
                if true_count == self.args.sample_num: all_correct_count += 1
                elif true_count == 0: all_incorrect_count += 1

                # Save record
                result = {
                    qid: {
                        "question": f"{question.strip()} {suffix}",
                        "responses": responses,
                        "judges": judges,
                        "ground_truth": list(set(raw_aliases))
                    }
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                if len(processed_qids) % 10 == 0:
                    tqdm.write(f"Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}")

if __name__ == "__main__":
    args = parse_args()
    sampler = ConsistencySampler(args)
    sampler.process_data()