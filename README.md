# Retrieval-Augmented Generation for Multi-Hop Question Answering

Master's thesis project at TU Braunschweig (IFN). Implements and evaluates a multi-hop RAG pipeline with GRPO-based LLM fine-tuning for complex question answering.

---

## Overview

Standard RAG pipelines retrieve documents once and generate an answer in a single pass. This fails for **multi-hop questions** that require chaining information across multiple documents. This project addresses that by:

1. Implementing an **iterative retrieval-and-reasoning loop** where the model dynamically issues sub-queries, retrieves documents, and checks whether it has enough information before producing a final answer.
2. Adapting **Group Relative Policy Optimization (GRPO)** to fine-tune the LLM across entire multi-hop reasoning trajectories, not just single-step responses.

---

## Results

Evaluated on subsets of [HotpotQA](https://hotpotqa.github.io/) (medium + hard questions):

| Pipeline | Exact Match | F1 |
|---|---|---|
| One-hop baseline | 23.5% | 31.3% |
| Multi-hop baseline (no fine-tuning) | 25.9% | 34.3% |
| **Multi-hop + GRPO fine-tuning** | **30.6%** | **39.8%** |

- **+7.1% Exact Match** over one-hop baseline
- **+4.7% Exact Match** over non-fine-tuned multi-hop baseline

---

## Pipeline Architecture

### Multi-hop inference loop

The pipeline has three components: an initial retrieval step, then an iterative loop alternating between generation sub-tasks (Short Response, Verify Response, Create Sub-queries) and retrieval. The loop terminates either when the response is verified or a loop limit is reached.

![Multi-hop pipeline](diagrams/multihop_pipeline.png)

### GRPO adaptation

Standard GRPO was modified to operate over full multi-hop trajectories. Each trajectory consists of multiple prompt-response pairs across loop iterations. Rewards and advantages are calculated per output step and propagated back across all tokens in that step.

![GRPO adaptation](diagrams/grpo_adaptation.png)

---

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Qwen3-4B (non-thinking mode) |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | FAISS (IVF index, ~14M sentence vectors from Wikipedia) |
| Fine-tuning | TRL (GRPOTrainer, extended for multi-hop) + LoRA adapters |
| Structured generation | Outlines (regex-constrained decoding) |
| Dataset | HotpotQA (medium + hard splits) |

---

## Project Structure

```
multihop-retrieval/
├── multihop_retrieval/          # Main package
│   ├── trainer.py               # MultihopGRPOTrainer (extends TRL GRPOTrainer)
│   ├── utils/
│   │   ├── inference_utils.py   # Inferrer class (iterative RAG loop)
│   │   ├── retrieval_utils.py   # Retriever class (FAISS wrapper)
│   │   ├── generic_utils.py     # Task enums, reward functions, utilities
│   │   └── outlines_transformers.py  # Grammar-constrained generation
│   ├── preprocessing/
│   │   └── embedding.py         # Build FAISS index from Wikipedia
│   └── script_helpers/
│       ├── inference.py         # Batch inference helpers
│       └── evaluation.py        # EM / F1 metric computation
├── example-scripts/             # Runnable end-to-end examples
│   ├── training_script.py       # Fine-tuning with MultihopGRPOTrainer
│   ├── inference_script.py      # Run inference and save results
│   ├── evaluation_script.py     # Evaluate saved results
│   └── adapter_inference_script.py  # Inference with loaded LoRA adapters
├── thesis-scripts/              # Experimental configurations used in thesis
├── diagrams/                    # Architecture diagrams (PNG)
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.12, CUDA 12.6

```bash
git clone https://github.com/your-username/multihop-retrieval.git
cd multihop-retrieval
pip install -r requirements.txt
```

> The torch line in `requirements.txt` targets CUDA 12.8 wheels. Adjust the `--index-url` if your CUDA version differs.

### Data setup

You will need:

1. **HotpotQA dataset**: download from [hotpotqa.github.io](https://hotpotqa.github.io/) and place splits under `data/HotpotQA_split/`.
2. **Wikipedia FAISS index**: ~14M sentence vectors embedded with `all-MiniLM-L6-v2`. Build it with:
   ```bash
   python -m multihop_retrieval.preprocessing.embedding
   ```
   Place the resulting `ivf_index.faiss` and `merged_lookup.json` under `data/minilm-embedded/`.
3. **Qwen3-4B model weights**: download via HuggingFace (`Qwen/Qwen3-4B`) and point `MODEL` paths in the scripts to your local cache.

Environment variables (used by training scripts via `.env`):

```
BASE_PATH=<path to data root>
EMBEDDER=all-MiniLM-L6-v2
EMBEDDING_DIR=minilm-embedded
WIKI_PATH=<path to Wikipedia text>
DATA_PATH=<path to HotpotQA splits>
WANDB_KEY=<optional W&B API key>
```

---

## Usage

### Fine-tuning

```bash
python example-scripts/training_script.py
```

Key hyperparameters (configured inside the script):

| Parameter | Value |
|---|---|
| Base model | Qwen3-4B |
| LoRA rank | 64 |
| Learning rate | 3e-6 |
| LR schedule | Cosine with 10% warmup |
| Batch size | 64 (per device) |
| Iterations per trajectory | 2 |
| Generations per prompt | 8 |
| Loss type | DR-GRPO |

Checkpoints are saved to `OUTPUT_PATH` every 25 steps as LoRA adapters.

### Inference

```bash
python example-scripts/inference_script.py
```

This runs the full `infer_vod_hist` pipeline on the dev split and saves results to JSON. It also demonstrates loading a saved LoRA checkpoint for inference.

### Evaluation

```bash
python example-scripts/evaluation_script.py
```

Computes Exact Match and F1 scores over the saved inference outputs.

---

## Inference Modes

The `Inferrer` class exposes several pipeline variants:

| Mode | Description |
|---|---|
| `infer_onehop` | Single-step baseline: one retrieval + one answer |
| `infer_basic` | INFO_CHECK + SUBQUERY_CONSTRUCT (no history) |
| `infer_vod` | PROVIDE_ANSWER + VERIFY_OR_DENY + SUBQUERY_CONSTRUCT |
| `infer_vod_hist` | **Main pipeline**: PROVIDE_ANSWER + VERIFY_OR_DENY + SUBQUERY_CONSTRUCT_WITH_HISTORY |

---

## Reward Design

| Reward | Description | Applied at |
|---|---|---|
| Exact Match (EM) | Binary match after normalization | Trajectory level |
| F1-score | Token overlap with gold answer | Trajectory level |
| Formatting consistency | Penalty for invalid output structure | Step level |
| Document Retrieval Score | Recall of gold supporting documents | Trajectory level |
| Loop Exit Decision Score | Correctness of verify-response decisions | Trajectory level |
| Length Regularization | Exponential penalty for verbose responses | Trajectory level |

---

## Reference

Full thesis available on request. Supervised by Patrick Blumenberg, M.Sc. under Prof. Dr.-Ing. Tim Fingscheidt, Institute for Communications Technology, TU Braunschweig.