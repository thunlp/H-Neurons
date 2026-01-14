import os
import json
import argparse
import time
from typing import List, Optional, Set

import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Extract answer tokens from consistent responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to samples files")
    parser.add_argument("--output_path", type=str, default="data/answer_tokens.jsonl", help="Path to save processed results")
    parser.add_argument("--tokenizer_path", type=str, default="data/activations", help="Path to the target model tokenizer")
    
    # LLM Extractor Config
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM for extraction")
    
    return parser.parse_args()

# ==========================================
# LLM Prompt Templates
# ==========================================

USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response: {response_tokens}
Please help extract the "answer tokens" from all tokens, removing all redundant information, and the tokens you return must be part of the input Tokenized Response list."""

EXAMPLE_MESSAGES = [
    {
        "role": "user", 
        "content": "Question: What is the correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator.\nResponse: The correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator is the Spirit of Ecstasy.\nTokenized Response: ['▁The', '▁correct', '▁name', '▁for', '▁the', '▁\"', 'F', 'lying', '▁Lady', '\"', '▁or', 'nament', '▁on', '▁a', '▁Roll', 's', '▁Roy', 'ce', '▁radi', 'ator', '▁is', '▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy', '.']\nPlease help extract the \"answer tokens\" from all tokens, removing all redundant information, and the tokens you return must form a continuous segment of the input Tokenized Response list."
    },
    {
        "role": "assistant", 
        "content": "['▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy']"
    },
]

# ==========================================
# Extraction Logic
# ==========================================

class AnswerTokenExtractor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def get_tokenized_list(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return tokens
    
    def extract_via_llm(self, question: str, response: str, tokens: List[str]) -> Optional[List[str]]:
        """Request LLM to select tokens from the tokenized list."""
        prompt = USER_INPUT_TEMPLATE.format(
            question=question,
            response=response,
            response_tokens=str(tokens)
        )
        
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                reply = completion.choices[0].message.content.strip().replace("'", "\"")
                extracted = json.loads(reply)
                
                # Validation: selected tokens must exist in original sequence
                if all(t in tokens for t in extracted):
                    return extracted
            except Exception as e:
                print(f"Extraction failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return None

    def load_processed_ids(self) -> Set[str]:
        """Resume from existing output file."""
        if not os.path.exists(self.args.output_path): return set()
        ids = set()
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try: ids.update(json.loads(line).keys())
                except: continue
        return ids

    def run(self):
        processed_ids = self.load_processed_ids()
        
        with open(self.args.input_path, "r", encoding="utf-8") as f_in, \
             open(self.args.output_path, "a", encoding="utf-8") as f_out:
            
            for line in tqdm(f_in, desc="Processing tokens"):
                data = json.loads(line)
                qid = list(data.keys())[0]
                content = data[qid]

                if qid in processed_ids: continue

                # Ensure all 10 responses have the same judge outcome (true/false)
                judges = content["judges"]
                if len(set(judges)) != 1 or "uncertain" in judges or "error" in judges:
                    continue

                # Take the most frequent response as representative
                responses = content["responses"]
                rep_response = max(set(responses), key=responses.count)
                
                # Tokenization
                tokenized_list = self.get_tokenized_list(rep_response)

                # LLM Extraction
                answer_tokens = self.extract_via_llm(content["question"], rep_response, tokenized_list)

                if answer_tokens:
                    result = {
                        qid: {
                            "question": content["question"],
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": judges[0] # Consistently correct or consistently hallucinated
                        }
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

if __name__ == "__main__":
    args = parse_args()
    extractor = AnswerTokenExtractor(args)
    extractor.run()