import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CETT activations for multiple token positions.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True, help="Path to answer_tokens.jsonl")
    parser.add_argument("--train_ids_path", type=str, required=True, help="Path to train_qids.json")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for saving .npy files")
    
    # Extraction Parameters
    parser.add_argument("--locations", nargs="+", default=["answer_tokens"], 
                        choices=["input", "output", "answer_tokens", "all_except_answer_tokens"],
                        help="List of positions to extract activations from")
    parser.add_argument("--method", type=str, choices=["mean", "max"], default="mean")
    parser.add_argument("--use_mag", action="store_true", default=True)
    parser.add_argument("--use_abs", action="store_true", default=True)
    return parser.parse_args()

class CETTManager:
    def __init__(self, model):
        self.model = model
        self.activations = [] # Input to down_proj (neuron activations)
        self.output_norms = [] # Layer output norms for normalization
        self.hooks = []
        self._register_hooks()
        self.weight_norms = self._get_weight_norms()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(input[0].detach())
            self.output_norms.append(torch.norm(output.detach(), dim=-1, keepdim=True))

        for name, module in self.model.named_modules():
            if 'down_proj' in name:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _get_weight_norms(self):
        norms = []
        for name, module in self.model.named_modules():
            if 'down_proj' in name:
                norms.append(torch.norm(module.weight.data, dim=0))
        return torch.stack(norms).to(self.model.device)

    def clear(self):
        self.activations.clear()
        self.output_norms.clear()

    def get_cett_tensor(self, use_abs=True, use_mag=True):
        """Returns tensor of shape [layers, tokens, neurons]"""
        self.activations = [act.squeeze(0) if act.dim() == 3 and act.size(0) == 1 else act for act in self.activations]
        self.output_norms = [norm.squeeze(0) if norm.dim() == 3 and norm.size(0) == 1 else norm for norm in self.output_norms]

        acts = torch.stack(self.activations).transpose(0, 1).to(self.model.device) # [tokens, layers, neurons]
        norms = torch.stack(self.output_norms).transpose(0, 1).to(self.model.device) # [tokens, layers, 1]

        if use_abs: acts = torch.abs(acts)
        if use_mag: acts = acts * self.weight_norms.unsqueeze(0)
        
        return (acts / (norms + 1e-8)).transpose(0, 1) # [layers, tokens, neurons]

def get_region_indices(full_ids: torch.Tensor, tokenizer, question: str, response: str, answer_tokens: List[str]) -> Dict[str, Optional[Tuple[int, int]]]:
    """Identify token indices for different sequence regions."""
    full_tokens = [tokenizer.decode([tid]) for tid in full_ids[0]]
    answer_tokens = [token.replace("▁", " ").replace("Ġ", " ") for token in answer_tokens]  # Normalize to tokenizer format
    token_str_list = "".join(full_tokens)
    
    # 1. Identify Input Region (User Prompt)
    user_ids = tokenizer.apply_chat_template([{"role": "user", "content": question}], return_tensors="pt")
    input_len = user_ids.shape[1] - 1 # Exclude potential separator/header
    
    # 2. Identify Output Region (Assistant Response)
    # Usually starts after the assistant header
    output_start = input_len + 1 
    output_end = len(full_tokens) - 1 # Exclude EOS
    
    # 3. Identify Answer Tokens Region
    ans_start, ans_end = None, None
    m = len(answer_tokens)

    if m != 0:
        for i in range(output_start, len(full_tokens) - m + 1):
            if full_tokens[i : i + m] == answer_tokens:
                ans_start, ans_end = i, i + m
                break
        
    return {
        "input": (0, input_len),
        "output": (output_start, output_end),
        "answer_tokens": (ans_start, ans_end) if ans_start is not None else None
    }

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    cett_manager = CETTManager(model)

    with open(args.train_ids_path, "r") as f:
        id_map = json.load(f)
        target_ids = set(id_map["t"] + id_map["f"])
    print(f"Loaded {len(target_ids)} target IDs for extraction.")

    # Prepare directories
    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    for sample_dict in tqdm(samples, desc="Processing"):
        qid = list(sample_dict.keys())[0]
        if qid not in target_ids:
            continue
        data = sample_dict[qid]
        cett_manager.clear()

        # Forward Pass
        msgs = [{"role": "user", "content": data["question"]}, {"role": "assistant", "content": data["response"]}]
        input_ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False).to(model.device)
        with torch.no_grad(): model(input_ids)
        
        cett_full = cett_manager.get_cett_tensor(use_abs=args.use_abs, use_mag=args.use_mag)
        regions = get_region_indices(input_ids, tokenizer, data["question"], data["response"], data["answer_tokens"])
        for loc in args.locations:
            indices = None
            if loc in ["input", "output", "answer_tokens"]:
                indices = regions[loc]
            elif loc == "all_except_answer_tokens" and regions["answer_tokens"]:
                # Special logic for inverse slicing
                ans_s, ans_e = regions["answer_tokens"]
                # Concatenate segments before and after answer tokens within the output
                seg1 = cett_full[:, :ans_s, :]
                seg2 = cett_full[:, ans_e:, :]
                selected_cett = torch.cat([seg1, seg2], dim=1)
            
            if loc != "all_except_answer_tokens" and indices:
                selected_cett = cett_full[:, indices[0]:indices[1], :]
            elif loc != "all_except_answer_tokens":
                continue

            # Aggregate
            if args.method == "mean": final_act = selected_cett.mean(dim=1)
            else: final_act, _ = selected_cett.max(dim=1)

            save_path = os.path.join(args.output_root, loc, f"act_{qid}.npy")
            np.save(save_path, final_act.cpu().float().numpy())

if __name__ == "__main__":
    main()