# H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs

<p align="center">
  <a href="#features">Features</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#citation">Citation</a>
</p>
This repository contains the official implementation of the paper:

Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun. [H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs](https://arxiv.org/abs/2512.01797).

Our research demonstrates that a remarkably sparse subset of neurons can reliably predict hallucination occurrences and are causally linked to over-compliance behaviors.

## Features
1. **Microscopic Identification**: Locate the precise neurons (H-Neurons) that can predict hallucination occurrences.
2. **CETT Metric**: Implementation of the Neuron Contribution Quantification (CETT).
3. **Linear Classifier Training**: Support for various training strategies to train a linear classifier for hallucination prediction.
4. **Generalizable Detection**: Identified neurons show robust hallucination prediction performance across in-domain and out-of-distribution scenarios.

## Pipeline
The H-Neuron identification process consists of five main steps:

1. **Response Collection**: Perform multiple samplings (10x) for training and a single sampling for evaluation.
2. **Answer Token Extraction**: Use LLM-based extraction to pinpoint factual tokens in the response.
3. **Balanced Sampling**: Create a balanced dataset of True/False samples for training/testing.
4. **Activation Extraction**: Quantify neuron-level contributions.
5. **Classifier Training/Evaluating**: Train a sparse Logistic Regression to identify H-Neurons and detect hallucination.

## Installation
This project utilizes `vLLM` for efficient model response sampling. If you encounter any issues during the installation of `vLLM`, please refer to the [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation).

```bash
git clone https://github.com/thunlp/H-Neurons.git
cd H-Neurons
pip install -r requirements.txt
```

## Usage

Ensure your environment is set up and you have access to a target model (e.g., Llama-3, Mistral).

### 1. Collect Responses
Sample responses from TriviaQA to construct train/test set. You can choose between `rule` or `llm` judging.

*   **For Training**: We perform multiple samplings (default 10) per question. We only retain questions that are **consistently correct** or **consistently hallucinated** across all samples.
*   **For Evaluation**: We perform only **1 sampling** per question to evaluate the classifier's performance in a standard, real-world inference scenario.

We provide sampled TriviaQA training and evaluation data for Gemma-3-27B-it, Llama-3.3-70B-Instruct, and Mistral-Small-3.1-24B-Instruct-2503 in `data/examples`.

```bash
python scripts/collect_responses.py \
    --model_path /path/to/your/model \
    --data_path data/TriviaQA/rc.nocontext/triviaqa_train.parquet \
    --output_path data/train_samples.jsonl \
    --sample_num 10 \
    --max_samples 10000 \
    --judge_type llm \
    --api_key "YOUR_OPENAI_API_KEY" \
    --base_url "YOUR_BASE_URL" \
    --judge_model gpt-4o
```


### 2. Extract Answer Tokens (Optional)

Use an LLM (e.g., GPT-4o) to pinpoint the exact "answer tokens" within the tokenized response. This ensures the classifier focuses on factual content.

**Note:** We observed that training specifically on exact **answer tokens** yields higher prediction accuracy during evaluation. However, if you wish to avoid the additional overhead (cost and time) of tagging answer tokens, you can choose to train and evaluate on all **output tokens**. To do this, you can manually construct an `answer_tokens.jsonl` file where the `answer_tokens` field is an empty list `[]` for all entries, and then set `--locations output` in Step 4 for both train and test sets.

```bash
python scripts/extract_answer_tokens.py \
    --input_path data/consistency_samples.jsonl \
    --output_path data/answer_tokens.jsonl \
    --tokenizer_path /path/to/your/model \
    --api_key "YOUR_OPENAI_API_KEY" \
    --base_url "YOUR_BASE_URL" \
    --judge_model gpt-4o
```

### 3. Sample Balanced Training IDs

Generate a balanced set of Question IDs (equal numbers of Faithful and Hallucinated samples) to serve as the training or validation set.

```bash
python scripts/sample_balanced_ids.py \
    --input_path data/answer_tokens.jsonl \
    --output_path data/train_qids.json \
    --num_samples 1000
```

### 4. Quantify Neuron Contribution

Compute and save the neuron contributions (CETT). This script handles the forward pass and supports extracting activations from multiple locations (Input, Output, Answer Tokens, All Tokens except Answer Tokens).

```bash
python scripts/extract_activations.py \
    --model_path /path/to/your/model \
    --input_path data/answer_tokens.jsonl \
    --train_ids_path data/train_qids.json \
    --output_root data/activations \
    --locations answer_tokens all_except_answer_tokens input output \
    --method mean \
    --use_mag --use_abs
```

### 5. Train Hallucination Classifier

Train the linear classifier to identify H-Neurons.
```bash
python scripts/classifier.py \
    --model_path /path/to/your/model \
    --train_ids data/train_qids.json \
    --train_ans_acts data/activations/answer_tokens \
    --train_other_acts data/activations/all_except_answer_tokens \
    --train_mode 3-vs-1 \
    --penalty l1 \
    --C 1.0 \
    --save_model models/classifier.pkl
```

*   **`--train_mode`**:
    *   **`1-vs-1`**: Distinguishes between hallucinated answer tokens and faithful answer tokens. **If you just want to achieve the highest predictive accuracy, we recommend using this setting.**
    *   **`3-vs-1`**: The strategy used in our paper. It contrasts hallucinated answer tokens against a combined negative set of faithful answer tokens and non-answer tokens. This mode is designed to isolate neurons specifically linked to factual errors.

*   **`--penalty`**:
    *   **`l1`**: Encourages sparsity by driving most neuron weights to zero. This is essential for **identifying a sparse subset of H-Neurons** for downstream analysis.
    *   **`l2`**: Does not produce sparse weights but typically provides a **1-3% boost in accuracy**. Use this if you only need a hallucination detector.

*   **`--C` (Regularization Strength)**:
    *   In `l1` regularization, the value of $C$ (inverse of regularization strength) controls the number of identified non-zero weights neurons. 
    *   Selecting an appropriate \(C\) is a trade-off. On one hand, setting \(C\) too low enforces aggressive sparsity, which risks excluding too many neurons. On the other hand, setting \(C\) too high introduces noise by including neurons essential for general language modeling, thereby causing damage to the model's fundamental capabilities during intervention.
    We perform a grid search to select \(C\) to maximize the sum of (1) classification accuracy on a held-out set and (2) model performance on TriviaQA when suppressing the identified H-Neurons.
    But if you only need a hallucination detector, just use L2 regularization and set \(C\) as default.


*   **Model Management**:
    *   **Training**: Provide `--train_ids` and activation directories to train and save a new model via `--save_model`.
    *   **Evaluation**: Use `--load_model` to skip training and evaluate a pre-trained `.pkl` model on your test set `--test_acts`.

### Model Intervention

To perform neuron-level intervention, we recommend modulating the activations directly during the forward pass within your inference code. Specifically, you should scale the **input to the `down_proj` layer** at the indices identified as **H-Neurons** (those with positive weights in the trained $L_1$ classifier).

We also provide example functions in `scripts/intervene_model.py`.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{gao2025hneuronsexistenceimpactorigin,
      title={H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs}, 
      author={Cheng Gao and Huimin Chen and Chaojun Xiao and Zhiyi Chen and Zhiyuan Liu and Maosong Sun},
      year={2025},
      eprint={2512.01797},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.01797}, 
}
```