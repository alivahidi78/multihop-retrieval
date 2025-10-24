import unsloth
import json, os, copy
from dotenv import load_dotenv
from multihop_retrieval.utils.inference import Inferrer 
from multihop_retrieval.utils.retrieval import Retriever
from multihop_retrieval.trainer import MultihopGRPOTrainer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from tqdm import tqdm
import wandb

USE_SPLIT = False

if __name__ == "__main__":
    
    train_limit = 500
    eval_limit = 100
    nprobe = 32
    iterations = 3
    epochs = 1
    
    # run_name = "exp-1 (baseline)"
    # no eval
    # epochs = 2
    run_name = "exp-2 (discrete rewards)"
    
    load_dotenv()
    WANDB_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="huggingface",
        name=run_name,
    )
    
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
    
    with open(os.path.join(DATA_PATH, "./hotpot_train_dev_subset.json"), "r") as f:
        all_data = json.load(f)
        
    print("dataset loaded.")
    
    # inferrer = Inferrer.create_with_retriever(WIKI_PATH, embedder, cpu_index, metadata, model, tokenizer, TOOLS_PATH)
    
    # data = all_data[:limit]
    # data = inferrer.infer(data, iterations=3, use_tqdm=True, logs=True)
    # with open(os.path.join(OUTPUT_PATH, f"./dev_iter_{iterations}_processed_5.json"), "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)
        
    # print("inference successful.")
    
    from trl import GRPOConfig
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-0.6B",
        max_seq_length = 8000,
        dtype = None,
        load_in_4bit = False,
        fast_inference=False, #needs vllm
        gpu_memory_utilization=0.9,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ['q_proj', 'v_proj'],
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    training_args = GRPOConfig(
        output_dir="./results",
        logging_dir="./logs",
        num_generations=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=5,
        save_steps=100,
        num_train_epochs= epochs,
        label_names=["labels"],
        gradient_accumulation_steps = 1,
        beta = 0.0,
        eval_on_start=True,
        eval_steps=50,
        eval_strategy="steps",
        report_to="wandb",
        run_name=run_name
    )
    
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    train_set = copy.deepcopy(all_data[:train_limit])
    eval_set = copy.deepcopy(all_data[train_limit:train_limit + eval_limit])
    trainer = MultihopGRPOTrainer(
        model=model,
        args=training_args,
        # iterations = 3,
        retriever = retriever,
        tools_path = TOOLS_PATH,
        # reward_funcs=reward_funcs(model),
        train_dataset = train_set,
        eval_dataset = eval_set,
        unbundled_batching = 8
    )
    
    trainer.train()