import json, os
from dotenv import load_dotenv
from multihop_retrieval.utils.inference import Inferrer 
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

if __name__ == "__main__":
    
    load_dotenv()
    BASE_PATH = os.getenv("BASE_PATH")
    EMBEDDER = os.getenv("EMBEDDER")
    EMBEDDING_DIR = os.getenv("EMBEDDING_DIR")
    WIKI_PATH = os.getenv("WIKI_PATH")
    DATA_PATH = os.getenv("DATA_PATH")
    MODEL = os.getenv("MODEL")
    MODEL_CACHE = os.getenv("MODEL_CACHE")
    PROMPTS_PATH = os.getenv("PROMPTS_PATH")
    TOOLS_PATH = os.getenv("TOOLS_PATH")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH")
    
    print("Current working directory: ", os.getcwd())
    print("base path: ", BASE_PATH)
    print("embedder: ", EMBEDDER)
    print("embedding dir: ", EMBEDDING_DIR)
    print("wiki path: ", WIKI_PATH)
    print("data path: ", DATA_PATH)
    print("model: ", MODEL)
    print("model cache: ", MODEL_CACHE)
    print("prompts path: ", PROMPTS_PATH)
    print("tools path: ", TOOLS_PATH)
    print("output path: ", OUTPUT_PATH)
    
    limit = 16
    nprobe = 32
    iterations = 3
    
    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    chunk_dir =  os.path.join(BASE_PATH, EMBEDDING_DIR)
    cache_dir = os.path.join(BASE_PATH, MODEL_CACHE)
    
    print("embedder loaded.")
    
    with open(f"{chunk_dir}/merged_lookup.json", "r") as f:
        metadata = json.load(f)
        
    print("lookup table loaded.")
        
    cpu_index = faiss.read_index(f"{chunk_dir}/ivf_index.faiss")
    cpu_index.nprobe = nprobe
    
    print("ivf index loaded.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=cache_dir)
    model = model.to("cuda")
    
    print("llm loaded.")
    
    with open(os.path.join(BASE_PATH, DATA_PATH, "./hotpot_train_dev_subset.json"), "r") as f:
        all_data = json.load(f)
        
    print("dataset loaded.")
    
    inferrer = Inferrer.create_with_retriever(BASE_PATH, WIKI_PATH, embedder, cpu_index, metadata, model, tokenizer, PROMPTS_PATH, TOOLS_PATH)
    
    data = all_data[:limit]
    data = inferrer.infer(data, iterations=3, use_tqdm=True, logs=True)
    with open(os.path.join(BASE_PATH, OUTPUT_PATH, f"./dev_iter_{iterations}_processed_5.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)