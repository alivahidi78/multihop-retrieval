import os, json, time

program_start = time.time()

import unsloth
import faiss

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
TOOLS_PATH = "./tools/var_3.json"
OUTPUT_PATH = "../data/inf_/4_vodh"

if __name__ == "__main__":
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = 8000,
        dtype = None,
        load_in_4bit = False,
        fast_inference=False, #needs vllm
        gpu_memory_utilization=0.6,
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

    with open(os.path.join(DATA_PATH, "./HotpotQA_split/hotpot_train_train_subset_mh.json"), "r") as f:
        all_data = json.load(f)

    print(f"dataset loaded {len(all_data)}.")
    
    all_data = all_data[-1000:]

    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    print("embedder loaded.")

    with open(f"multihop_retrieval/{TOOLS_PATH}", "r") as f:
        prompts_and_tools = json.load(f)
    
    config = GenerationConfig(
                max_new_tokens=128,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,)
    inf_config = InferrerConfig(use_tqdm=False, logs=False, iterations=2, remove_tensors=True, add_onehop=False)
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    inferrer = Inferrer(retriever, model, tokenizer, prompts_and_tools, inf_config)
    
    run_inference_and_save(all_data, OUTPUT_PATH, inferrer.infer_vod_hist, 20)
    program_end = time.time()
    print(f"program concluded in {(program_end - program_start)/60:.4f} minutes.")    