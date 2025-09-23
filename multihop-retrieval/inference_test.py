import json, os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from utils.inferenece import Inferrer 

BASE_PATH = "./"
EMBEDDER = "all-MiniLM-L6-v2"
EMBEDDING_DIR = "./minilm-embedded"
WIKI_PATH = "./"
DATA_PATH = "./data"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_CACHE = "./qwen3_0.6b"
PROMPTS_PATH = "./tools/prompts_var_0.json"
TOOLS_PATH = "./tools/tools_var_0.json"

if __name__ == "__main__":
    
    limit = 8
    nprobe = 32
    iterations = 3
    
    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    chunk_dir =  os.path.join(BASE_PATH, EMBEDDING_DIR)
    cache_dir = os.path.join(BASE_PATH, MODEL_CACHE)
    
    with open(f"{chunk_dir}/merged_lookup.json", "r") as f:
        metadata = json.load(f)
        
    cpu_index = faiss.read_index(f"{chunk_dir}/ivf_index.faiss")
    cpu_index.nprobe = nprobe
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=cache_dir)
    model = model.to("cuda")
    
    with open(os.path.join(BASE_PATH, DATA_PATH, "./hotpot_train_dev_subset.json"), "r") as f:
        all_data = json.load(f)
    inferrer = Inferrer.create_with_retriever(BASE_PATH, WIKI_PATH, embedder, cpu_index, metadata, model, tokenizer, PROMPTS_PATH, TOOLS_PATH)
    
    data = all_data[:limit]
    data = inferrer.infer(data, iterations=3)
    with open(os.path.join(BASE_PATH, DATA_PATH, f"./dev_iter_{iterations}_processed_5.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)