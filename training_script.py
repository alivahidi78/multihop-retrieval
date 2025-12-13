import json, os, copy, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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
# MODEL = "unsloth/Qwen3-4B"
OUTPUT_PATH = "./results/test-f5-7"
TOOLS_PATH = "./multihop_retrieval/tools/var_6.json"
BASE_PATH = os.getenv("BASE_PATH")
EMBEDDER = os.getenv("EMBEDDER")
EMBEDDING_DIR = os.path.join(BASE_PATH, os.getenv("EMBEDDING_DIR"))
WIKI_PATH = os.path.join(BASE_PATH, os.getenv("WIKI_PATH"))
DATA_PATH = os.path.join(BASE_PATH, os.getenv("DATA_PATH"))
RUN_NAME = "f5-7 (simple 3e-6)"

def get_reward_functions(prompts_and_tools):
    def info_decision_judge(data, final_answers, bundle_lengths, **kwargs):
        count = 0
        r_sum = 0
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
                if pr_label in prompts.keys():
                    context = original_context.copy()
                    for k in range(0, j):
                        try:
                            context.extend(d[Inferrer.task_label(Task.RETRIEVE, k)])
                        except KeyError:
                            print("missing retrieval")
                    supported_ret = [False]*len(supporting_facts)
                    for x, f in enumerate(supporting_facts):
                        for c in context:
                            if(f[0] == c[0]):
                                supported_ret[x] = True
                    index += 1
                if vod_label in prompts.keys():
                    enough, malformed = utils.information_judgement(prompts_and_tools, completions[vod_label], Task.VERIFY_OR_DENY)
                    if malformed:
                        pass
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
                            rewards[index] = 1.2
                    count += 1     
                    r_sum += rewards[index]
                    index += 1
                if sc_label in prompts.keys():
                    rewards[index] = 0  
                    index += 1
        # for i in range(len(rewards)):
        #     rewards[i] = r_sum/count  
        assert sum(bundle_lengths) == index   
        return rewards
    
    def subq_decision_judge(data, final_answers, bundle_lengths, **kwargs):
        count = 0
        r_sum = 0
        rewards = [0]*sum(bundle_lengths)
        index = 0
        for d in data:
            supporting_facts = d["supporting_facts"]
            context = d["context"].copy()
            prompts = d["prompt"]
            completions = d["completion_decoded"]
            retrievals = []
            for k in range(0, 3):
                try:
                    retrievals.extend(d[Inferrer.task_label(Task.RETRIEVE, k)])
                except KeyError:
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
                        pass
                    else:
                        if all([not(a) and b for a, b in zip(context_supported_ret, new_supported_ret)]):
                            rewards[index] = 1
                        elif (True in [(not a) and b for a, b in zip(context_supported_ret, new_supported_ret)]):
                            rewards[index] = 0.5
                        else:
                            pass
                            # rewards[index] = -1
                    count += 1     
                    r_sum += rewards[index]
                    index += 1  
        # for i in range(len(rewards)):
        #     rewards[i] = r_sum/count 
        assert sum(bundle_lengths) == index      
        return rewards
    
    def formatting_judge(data, final_answers, bundle_lengths, **kwargs):
        rewards = [0]*sum(bundle_lengths)
        index = 0
        for d in data:
            prompts = d["prompt"]
            completions = d["completion_decoded"]
            for j in range(3):
                pr_label = Inferrer.task_label(Task.PROVIDE_ANSWER, j)
                vod_label = Inferrer.task_label(Task.VERIFY_OR_DENY, j)
                sc_label = Inferrer.task_label(Task.SUBQUERY_CONSTRUCT_WITH_HISTORY, j)
                if pr_label in prompts.keys():
                    enough, malformed = utils.information_judgement(prompts_and_tools, completions[pr_label], Task.PROVIDE_ANSWER)
                    if malformed:
                        rewards[index] = -1.0   
                    index += 1
                if vod_label in prompts.keys():
                    enough, malformed = utils.information_judgement(prompts_and_tools, completions[vod_label], Task.VERIFY_OR_DENY)
                    if malformed:
                        rewards[index] = -1.0     
                    index += 1
                if sc_label in prompts.keys():
                    proper = utils.format_judgement(prompts_and_tools, completions[sc_label], Task.SUBQUERY_CONSTRUCT_WITH_HISTORY)
                    if not proper:
                        rewards[index] = -1.0
                    index += 1  
        assert sum(bundle_lengths) == index      
        return rewards
         
    
    r_functions = MultihopGRPOTrainer.get_default_reward_functions(prompts_and_tools)
    return r_functions + [info_decision_judge, subq_decision_judge, formatting_judge]

if __name__ == "__main__":    
    torch._dynamo.config.cache_size_limit = 10**8
    train_limit = 5000
    # eval_limit = 100
    acc_steps = 1
    nprobe = 32
    iterations = 2
    epochs = 1
    
    WANDB_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="huggingface",
        name=RUN_NAME,
    )
    
    print("Current working directory: ", os.getcwd())
    print("base path: ", BASE_PATH)
    print("embedder: ", EMBEDDER)
    # print("model: ", MODEL)
    print("embedding dir: ", EMBEDDING_DIR)
    print("wiki path: ", WIKI_PATH)
    print("data path: ", DATA_PATH)
    print("tools path: ", TOOLS_PATH)
    print("output path: ", OUTPUT_PATH)
    
    with open(TOOLS_PATH, 'r') as f:
        prompts_and_tools = json.load(f)
    
    # embedder = SentenceTransformer(EMBEDDER, device="cuda", cache_folder="../data/embedder")
    embedder = SentenceTransformer("../data/embedder/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf", device="cuda")
    
    print("embedder loaded.")
    
    with open(os.path.join(DATA_PATH, "./hotpot_train_train_subset_mh.json"), "r") as f:
        all_data = json.load(f)
        
    print("dataset loaded.")
    
    from trl import GRPOConfig
    
    model_name = "Qwen/Qwen3-4B"

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "../data/cache_test/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        # model_name
        )

    # 2. Load the base model (fp16/bfloat recommended)
    model = AutoModelForCausalLM.from_pretrained(
        "../data/cache_test/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        # model_name,
        # torch_dtype=torch.bfloat16,
        # device_map="auto",
        # cache_dir = "../data/cache_test"
    )

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=64,                # rank
        lora_alpha=64,       # scaling
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],                   # Qwen uses LLaMA-like module names
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
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
        loss_type="dr_grpo",
        gradient_accumulation_steps=acc_steps,
        epsilon=0.1,
        output_dir=OUTPUT_PATH,
        logging_dir="./logs",
        num_generations=8,
        per_device_train_batch_size=64,
        # per_device_eval_batch_size=8,
        logging_steps=5,
        save_steps=25,
        num_train_epochs= epochs,
        label_names=["labels"],
        beta = 0.0,
        eval_on_start=False,
        # eval_steps=200,
        # eval_strategy="steps",
        report_to="wandb",
        run_name=RUN_NAME,
        torch_empty_cache_steps = 1,
        reward_weights=[0.5, 0.3, 0.0, 0.0, 1.0],
        temperature=1.0,
        generation_kwargs={"temperature": 1.0},
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0
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
        unbundled_batching = 32,
        no_cache = True,
        inference_mode="vod_hist"
    )
    trainer.train(
            resume_from_checkpoint=False, 
    )
