"""Module includes example script for inference.
"""
import os, json, time

program_start = time.time()

import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

from multihop_retrieval.utils.inference_utils import Inferrer, InferrerConfig
from transformers.generation.configuration_utils import GenerationConfig
from sentence_transformers import SentenceTransformer
from multihop_retrieval.utils.retrieval_utils import Retriever
from multihop_retrieval.script_helpers.inference import run_inference_and_save

nprobe = 32
EMBEDDER = "all-MiniLM-L6-v2"
EMBEDDING_DIR = "../data/minilm-embedded"
WIKI_PATH = "../"
DATA_PATH = "../data"
MODEL = "unsloth/Qwen3-4B"
TOOLS_PATH = "./tools/var_6.json"
OUTPUT_PATH = "../data/inf_/test_0.0"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "../data/cache_infer/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        )

    # 2. Load the base model (fp16/bfloat recommended)
    model = AutoModelForCausalLM.from_pretrained(
        "../data/cache_infer/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        device_map="cuda"
    )
    print("model loaded.")
    
    with open(f"{EMBEDDING_DIR}/merged_lookup.json", "r") as f:
        metadata = json.load(f)
    print("lookup table loaded.")
    
    start = time.time()
    cpu_index = faiss.read_index(f"{EMBEDDING_DIR}/ivf_index.faiss")
    cpu_index.nprobe = nprobe
    end = time.time()
    print(f"ivf index loaded in {(end - start)/60:.4f} minutes.")

    with open(os.path.join(DATA_PATH, "./HotpotQA_split/hotpot_dev_shuffled.json"), "r") as f:
        test_data = json.load(f)[:1000]

    print(f"dataset loaded {len(test_data)}.")
    

    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    print("embedder loaded.")

    with open(f"multihop_retrieval/{TOOLS_PATH}", "r") as f:
        prompts_and_tools = json.load(f)
    
    config = GenerationConfig(
                max_new_tokens=128,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=0.0,)
    inf_config = InferrerConfig(generation_config=config, use_tqdm=False, logs=False, iterations=2, remove_tensors=True, add_onehop=False)
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    inferrer = Inferrer(retriever, model, tokenizer, prompts_and_tools, inf_config)
    run_inference_and_save(test_data, "../data/_complex/test_exp_2/checkpoint-0", inferrer.infer_vod_hist, 20, start=0)
    
    with open(os.path.join(DATA_PATH, "./HotpotQA_split/hotpot_dev_shuffled.json"), "r") as f:
        test_data = json.load(f)[:1000]
        
    model.load_adapter("./results/_complex/checkpoint-500", adapter_name=f"adapter_checkpoint")
    model.set_adapter(f"adapter_checkpoint")
    inferrer = Inferrer(retriever, model, tokenizer, prompts_and_tools, inf_config)
    run_inference_and_save(test_data, "../data/_complex/test_exp_2/checkpoint-500", inferrer.infer_vod_hist, 20, start=0)
    model.delete_adapter(f"adapter_checkpoint")
    
    program_end = time.time()
    print(f"program concluded in {(program_end - program_start)/60:.4f} minutes.")    