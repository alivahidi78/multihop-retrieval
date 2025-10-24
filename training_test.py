import json, os, copy
from dotenv import load_dotenv
from multihop_retrieval.utils.inference import Inferrer 
from multihop_retrieval.utils.retrieval import Retriever
from multihop_retrieval.trainer import MultihopGRPOTrainer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from tqdm import tqdm

USE_SPLIT = False

if __name__ == "__main__":
    
    load_dotenv()
    BASE_PATH = os.getenv("BASE_PATH")
    EMBEDDER = os.path.join(BASE_PATH, os.getenv("EMBEDDER"))
    EMBEDDING_DIR = os.path.join(BASE_PATH, os.getenv("EMBEDDING_DIR"))
    WIKI_PATH = os.path.join(BASE_PATH, os.getenv("WIKI_PATH"))
    DATA_PATH = os.path.join(BASE_PATH, os.getenv("DATA_PATH"))
    MODEL = os.path.join(BASE_PATH, os.getenv("MODEL"))
    MODEL_CACHE = os.path.join(BASE_PATH, os.getenv("MODEL_CACHE"))
    TOOLS_PATH = os.path.join(BASE_PATH, os.getenv("TOOLS_PATH"))
    OUTPUT_PATH = os.path.join(BASE_PATH, os.getenv("OUTPUT_PATH"))
    
    print("Current working directory: ", os.getcwd())
    print("base path: ", BASE_PATH)
    print("embedder: ", EMBEDDER)
    print("embedding dir: ", EMBEDDING_DIR)
    print("wiki path: ", WIKI_PATH)
    print("data path: ", DATA_PATH)
    print("model: ", MODEL)
    print("model cache: ", MODEL_CACHE)
    print("tools path: ", TOOLS_PATH)
    print("output path: ", OUTPUT_PATH)
    
    with open(TOOLS_PATH, 'r') as f:
        prompts_and_tools = json.load(f)
    
    limit = 16
    nprobe = 32
    iterations = 3
    
    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    
    print("embedder loaded.")
    
    with open(f"{EMBEDDING_DIR}/merged_lookup.json", "r") as f:
        metadata = json.load(f)
        
    print("lookup table loaded.")
    
    if not USE_SPLIT:
        cpu_index = faiss.read_index(f"{EMBEDDING_DIR}/ivf_index.faiss")
        cpu_index.nprobe = nprobe
    else:
        chunk_dir = "../data/minilm-embedded/split"
        index_files = sorted(f for f in os.listdir(chunk_dir) if f.startswith("ivf_shard") and f.endswith(".faiss"))
        print(len(index_files))
        cpu_index = faiss.IndexShards()
        for fpath in tqdm(index_files):
            index = faiss.read_index(os.path.join(chunk_dir, fpath))
            cpu_index.add_shard(index)
        cpu_index.nprobe = nprobe
        
    print("ivf index loaded.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=MODEL_CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=MODEL_CACHE)
    model = model.to("cuda")
    
    print("llm loaded.")
    
    with open(os.path.join(DATA_PATH, "./hotpot_train_dev_subset.json"), "r") as f:
        all_data = json.load(f)
        
    print("dataset loaded.")
    
    inferrer = Inferrer.create_with_retriever(WIKI_PATH, embedder, cpu_index, metadata, model, tokenizer, prompts_and_tools)
    
    data = all_data[:limit]
    data = inferrer.infer(data, iterations=3, use_tqdm=True, logs=True)
    with open(os.path.join(OUTPUT_PATH, f"./dev_iter_{iterations}_processed_5.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("inference successful.")
    
    from peft import get_peft_model, LoraConfig, TaskType
    from trl import GRPOConfig
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        # lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules = ['q_proj', 'v_proj'],
    )

    training_args = GRPOConfig(
        output_dir="./results",
        logging_dir="./logs",
        num_generations=2, # 8 is too much and fills memory
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=50,
        num_train_epochs=2,
        label_names=["labels"],
        # beta = 0.5,
        # eval_steps=200,
        # eval_strategy="steps",
        # report_to="none",
    )
    
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    data = copy.deepcopy(all_data[:limit])
    trainer = MultihopGRPOTrainer(
        model=model,
        args=training_args,
        # iterations = 3,
        retriever = retriever,
        prompts_and_tools = prompts_and_tools,
        # reward_funcs=reward_funcs(prompts_and_tools),
        train_dataset = data,
        peft_config = lora_config,
    )
    
    trainer.train()