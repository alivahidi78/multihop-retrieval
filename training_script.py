import json, os, copy, time
import unsloth
from multihop_retrieval.utils.inference_utils import Inferrer 
from multihop_retrieval.utils.retrieval_utils import Retriever
from multihop_retrieval.utils.generic_utils import Task
from multihop_retrieval.utils import generic_utils as utils
from multihop_retrieval.trainer import MultihopGRPOTrainer
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import wandb
import torch

from dotenv import load_dotenv
load_dotenv()
MODEL = "unsloth/Qwen3-4B"
OUTPUT_PATH = "./results/test"
BASE_PATH = os.getenv("BASE_PATH")
EMBEDDER = os.getenv("EMBEDDER")
EMBEDDING_DIR = os.path.join(BASE_PATH, os.getenv("EMBEDDING_DIR"))
WIKI_PATH = os.path.join(BASE_PATH, os.getenv("WIKI_PATH"))
DATA_PATH = os.path.join(BASE_PATH, os.getenv("DATA_PATH"))
TOOLS_PATH = os.path.join(BASE_PATH, os.getenv("TOOLS_PATH"))

def get_reward_functions(prompts_and_tools):
    def info_decision_judge(data, final_answers, bundle_lengths, **kwargs):
        rewards = [0]*sum(bundle_lengths)
        index = 0
        for d in data:
            supporting_facts = d["supporting_facts"]
            original_context = d["context"]
            prompts = d["prompt"]
            completions = d["completion_decoded"]
            for j in range(3):
                pr_label = Inferrer.task_label(Task.PROVIDE_ANSWER, j)
                vod_label = Inferrer.task_label(Task.VERIFY_OR_DENY, j)
                sc_label = Inferrer.task_label(Task.SUBQUERY_CONSTRUCT_WITH_HISTORY, j)
                context = original_context.copy()
                for k in range(0, j):
                    try:
                        context.extend(d[Inferrer.task_label(Task.RETRIEVE, k)])
                    except:
                        print("missing retrieval")
                supported_ret = [False]*len(supporting_facts)
                for x, f in enumerate(supporting_facts):
                    for c in context:
                        if(f[0] == c[0]):
                            supported_ret[x] = True
                if pr_label in prompts.keys():
                    enough, malformed = utils.information_judgement(prompts_and_tools, completions[pr_label], Task.PROVIDE_ANSWER)
                    if malformed:
                        rewards[index] = -5 
                    elif enough: 
                        if all(supported_ret):
                            rewards[index] = 1
                        elif True in supported_ret:
                            rewards[index] = 0
                        else:
                            rewards[index] = -1
                    else:
                        if all(supported_ret):
                            rewards[index] = -1
                        elif True in supported_ret:
                            rewards[index] = 0
                        else:
                            rewards[index] = 1.  
                    index += 1
                if vod_label in prompts.keys():
                    enough, malformed = utils.information_judgement(prompts_and_tools, completions[vod_label], Task.VERIFY_OR_DENY)
                    if malformed:
                        rewards[index] = -5
                    elif enough:
                        if all(supported_ret):
                            rewards[index] = 1
                        elif True in supported_ret:
                            rewards[index] = 0
                        else:
                            rewards[index] = -1
                    else:
                        if all(supported_ret):
                            rewards[index] = -1
                        elif True in supported_ret:
                            rewards[index] = 0
                        else:
                            rewards[index] = 1.5     
                    index += 1
                if sc_label in prompts.keys():
                    rewards[index] = 0  
                    index += 1  
        assert sum(bundle_lengths) == index   
        return rewards
    
    def subq_decision_judge(data, final_answers, bundle_lengths, **kwargs):
        rewards = [0]*sum(bundle_lengths)
        index = 0
        for d in data:
            supporting_facts = d["supporting_facts"]
            context = d["context"].copy()
            prompts = d["prompt"]
            completions = d["completion_decoded"]
            retrievals = {}
            for k in range(0, 3):
                try:
                    retrievals.extend(d[Inferrer.task_label(Task.RETRIEVE, k)])
                except:
                    pass
            context_supported_ret = [False]*len(supporting_facts)
            new_supported_ret = [False]*len(supporting_facts)
            for x, f in enumerate(supporting_facts):
                for c in context:
                    if(f[0] == c[0]):
                        context_supported_ret[x] = True
                for r in retrievals:
                    if(f[0] == r[0]):
                        new_supported_ret[x] = True
            for j in range(3):
                pr_label = Inferrer.task_label(Task.PROVIDE_ANSWER, j)
                vod_label = Inferrer.task_label(Task.VERIFY_OR_DENY, j)
                sc_label = Inferrer.task_label(Task.SUBQUERY_CONSTRUCT_WITH_HISTORY, j)
                if pr_label in prompts.keys():
                    rewards[index] = 0     
                    index += 1
                if vod_label in prompts.keys():
                    rewards[index] = 0     
                    index += 1
                if sc_label in prompts.keys():
                    proper = utils.format_judgement(prompts_and_tools, completions[sc_label], Task.SUBQUERY_CONSTRUCT_WITH_HISTORY)
                    if not proper:
                        rewards[index] = -5
                    else:
                        if all([not(a) and b for a, b in zip(context_supported_ret, new_supported_ret)]):
                            rewards[index] = 1
                        elif (True in [(not a) and b for a, b in zip(context_supported_ret, new_supported_ret)]):
                            rewards[index] = 0.5
                        else:
                            pass
                            # rewards[index] = -1
                    index += 1  
        assert sum(bundle_lengths) == index      
        return rewards 
    
    r_functions = MultihopGRPOTrainer.get_default_reward_functions(prompts_and_tools)
    return r_functions + [info_decision_judge, subq_decision_judge]

if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 10**8
    train_limit = 500
    # eval_limit = 100
    nprobe = 32
    iterations = 2
    epochs = 1
    
    run_name = "exp-5 (vod-hist test - 4B)"
    WANDB_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="huggingface",
        name=run_name,
    )
    
    print("Current working directory: ", os.getcwd())
    print("base path: ", BASE_PATH)
    print("embedder: ", EMBEDDER)
    print("model: ", MODEL)
    print("embedding dir: ", EMBEDDING_DIR)
    print("wiki path: ", WIKI_PATH)
    print("data path: ", DATA_PATH)
    print("tools path: ", TOOLS_PATH)
    print("output path: ", OUTPUT_PATH)
    
    with open(TOOLS_PATH, 'r') as f:
        prompts_and_tools = json.load(f)
    
    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    
    print("embedder loaded.")
    
    with open(os.path.join(DATA_PATH, "./hotpot_train_train_subset_mh.json"), "r") as f:
        all_data = json.load(f)
        
    print("dataset loaded.")
    
    from trl import GRPOConfig
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = 8000,
        dtype = None,
        load_in_4bit = False,
        fast_inference=False, #needs vllm
        gpu_memory_utilization=0.9,
        cache_dir = "../data/cache_test"
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
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    with open(f"{EMBEDDING_DIR}/merged_lookup.json", "r") as f:
        metadata = json.load(f)
        
    print("lookup table loaded.")
    
    start = time.time()
    cpu_index = faiss.read_index(f"{EMBEDDING_DIR}/ivf_index.faiss")
    cpu_index.nprobe = nprobe
    end = time.time()
    print(f"ivf index loaded in {(end - start)/60:.4f} minutes.")
        
    print("ivf index loaded.")
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_PATH,
        logging_dir="./logs",
        num_generations=8,
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        logging_steps=5,
        save_steps=50,
        num_train_epochs= epochs,
        label_names=["labels"],
        gradient_accumulation_steps = 1,
        beta = 0.0,
        eval_on_start=False,
        # eval_steps=200,
        # eval_strategy="steps",
        report_to="wandb",
        run_name=run_name,
        torch_empty_cache_steps = 1,
        reward_weights=[5, 3, 3, 3],
        temperature=1.0,
        generation_kwargs={"temperature": 1.0}
    )
    
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    train_set = copy.deepcopy(all_data[:train_limit])
    trainer = MultihopGRPOTrainer(
        model = model,
        retriever = retriever,
        prompts_and_tools = prompts_and_tools,
        args=training_args,
        iterations = 2,
        reward_funcs=get_reward_functions(prompts_and_tools),
        train_dataset = train_set,
        unbundled_batching = 8,
        no_cache = True,
        inference_mode="vod_hist"
    )
    
    trainer.train(
        #  resume_from_checkpoint=True
        )