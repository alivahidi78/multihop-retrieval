import os, json, time, re, traceback

program_start = time.time()

import unsloth
import faiss
from tqdm import tqdm
import torch
import pandas as pd

from multihop_retrieval.utils.inference_utils import Inferrer, InferrerConfig
from transformers.generation.configuration_utils import GenerationConfig
from sentence_transformers import SentenceTransformer
from multihop_retrieval.utils.retrieval_utils import Retriever
from multihop_retrieval.script_helpers.inference import run_inference_and_save
from dotenv import load_dotenv
load_dotenv()
import wandb

nprobe = 32
EMBEDDER = "all-MiniLM-L6-v2"
EMBEDDING_DIR = "../data/minilm-embedded"
WIKI_PATH = "../"
DATA_PATH = "../data"
MODEL = "unsloth/Qwen3-4B"
TOOLS_PATH = "./tools/var_5.json"
OUTPUT_PATH = "../data/_test_4"
CHECKPOINT_PATH = "./results/test-4"
RUN_NAME = "eval_4(pro is shortanswer)"

from multihop_retrieval.utils.inference_utils import Inferrer
from multihop_retrieval.utils import generic_utils as utils
from multihop_retrieval.utils.generic_utils import Task
from multihop_retrieval.trainer import MultihopGRPOTrainer

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
                    index += 1  
        assert sum(bundle_lengths) == index      
        return rewards
    
    def formatting_judge(data, final_answers, bundle_lengths, **kwargs):
        rewards = [0]*sum(bundle_lengths)
        index = 0
        for d in data:
            supporting_facts = d["supporting_facts"]
            context = d["context"].copy()
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

from multihop_retrieval.script_helpers.evaluation import load_data, assess_data

def upload_results(index, file_path):
    with open(f"multihop_retrieval/{TOOLS_PATH}", "r") as f:
        prompts_and_tools = json.load(f)
        
    all_data = []
    try:
        0.5, 0.3, 0.2, 0.2, 0.2
        all_data = load_data(file_path)[:]
        res = assess_data(all_data, index, get_reward_functions(prompts_and_tools), 2, min_llm=2)
        res.update({
            "reward": (res['compute_exact/mean']*0.5 + res['compute_f1/mean']*0.3 + res['info_decision_judge/mean']*0.2 + res['subq_decision_judge/mean']*0.2 + res['formatting_judge/mean']*0.2)/1.4,
            "reward_emf1": (res['compute_exact/mean']*0.5 + res['compute_f1/mean']*0.3)/0.8
        })
        wandb.log({
            "count": res["count"],
            "reward": res["reward"],
            "reward_emf1":res["reward_emf1"],
            "EM": res['compute_exact/mean'],
            "F1": res['compute_f1/mean'],
            "info_dec": res['info_decision_judge/mean'],
            "subq_dec": res['subq_decision_judge/mean'],
            "format": res['formatting_judge/mean'],
            "info_err": res["errors/info/f"],
            "subq_err": res["errors/subq/f"],
            "missing_ans": res["missing_ans"],
            "init_r":  round(res["init+r"]["part_match"] +res["init+r"]["full_match"], 4),
            "ret0":  round(res["ret0"]["part_match"] +res["ret0"]["full_match"], 4),
            "ret1":  round(res["ret1"]["part_match"] +res["ret1"]["full_match"], 4),
        }, step=index)
    except:
        traceback.print_exc()
        

def infer_from_adapter_and_llm(all_data, prompts_and_tools, model, tokenizer, retriever, checkpoints_dir, output_dir, method, reverse=False, generation_config=None, add_onehop=False, start=0, step=50, end=20):
    numbers = [100,200,300,400,500]
    checkpoints = [
        './checkpoint-100',
        './checkpoint-200',
        './checkpoint-300',
        './checkpoint-400',
        './checkpoint-500',
        ]
    checkpoints = sorted([os.path.join(checkpoints_dir, c) for c in checkpoints],
                         key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)),
    reverse=reverse)

    print("discovered checkpoints:", checkpoints)
    it = 0
    for num, ckpt_path in tqdm(zip(numbers,checkpoints), desc="checkpoints processed"):
        print(f"waiting for the data in {ckpt_path}...")
        while not os.path.isdir(ckpt_path):
            time.sleep(120)
        print("data online...")
        time.sleep(120)
        print("Waited two minutes.\nBeginning...")
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n🧩 Loading adapter from {ckpt_name}...")

        # Load LoRA adapter weights
        model.load_adapter(ckpt_path, adapter_name=f"adapter_checkpoint")
        model.set_adapter(f"adapter_checkpoint")
        # Run your operation
        if not generation_config:
            generation_config = GenerationConfig(
                        max_new_tokens=128,
                        do_sample=True,
                        top_k=None,
                        top_p=None,
                        temperature=0.6,)
        inf_config = InferrerConfig(use_tqdm=False, logs=False, iterations=2, remove_tensors=True, add_onehop=add_onehop, generation_config=generation_config)
        inferrer = Inferrer(retriever, model, tokenizer, prompts_and_tools, inf_config)
        save_path = os.path.join(output_dir, ckpt_name)
        os.makedirs(save_path, exist_ok=True)
        if method == "basic":
            infer_func = inferrer.infer_basic
        elif method == "hist":
            infer_func = inferrer.infer_vod_hist
        elif method == "vod":
            infer_func = inferrer.infer_vod
        elif method == "vod_hist":
            infer_func = inferrer.infer_vod_hist
        else:
            raise ValueError(f"inference mode {method} is unknown.")
        run_inference_and_save(all_data, save_path, infer_func, end, start=start, step=step)
        upload_results(num, save_path)
        # Unload adapter to save VRAM
        model.delete_adapter(f"adapter_checkpoint")
        torch.cuda.empty_cache()
        it+=1

    print("\nDone processing all checkpoints!")
    
if __name__ == "__main__":
    WANDB_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="evaluation",
        name=RUN_NAME,
        entity="alivahidi"
    )
    time.sleep(20)
    wandb.log({
        "count": 1000,
        "reward": 0.228925,
        "reward_emf1": 0.385677,
        "EM": 0.362,
        "F1": 0.425137,
        "info_dec": 0.026193,
        "subq_dec": 0.013654,
        "format": 0.0,
        "info_err": 0.004,
        "subq_err": 0.0,
        "missing_ans": 4,
        "init_r": 0.6790,
        "ret0":  0.1100,
        "ret1":  0.0070,
    }, step=0)
    wandb.log({"Test": 5}, step=0)
    wandb.log({"Test": 5}, step=50)
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
    
    retriever = Retriever(WIKI_PATH, embedder, cpu_index, metadata)
    
    infer_from_adapter_and_llm(all_data, prompts_and_tools, model, tokenizer, retriever, CHECKPOINT_PATH, OUTPUT_PATH, "vod_hist", reverse=False)
    
    program_end = time.time()
    print(f"program concluded in {(program_end - program_start)/60:.4f} minutes.")
    wandb.finish()