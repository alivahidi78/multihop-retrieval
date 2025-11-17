from tqdm import tqdm
import os, copy, json, traceback, re, time, torch
from multihop_retrieval.utils import utils
from multihop_retrieval.utils.inference_utils import InferrerConfig, Inferrer
from multihop_retrieval.utils.retrieval_utils import Retriever
from transformers import GenerationConfig

def run_inference_and_save(all_data, output_dir, method, end, start=0, step=50, use_tdqm=True):
    start_idx = step * start
    end_idx = start_idx + step
    for i in tqdm(range(start, end), desc="chunk processed", disable=use_tdqm):
        try:
            eval_set = copy.deepcopy(all_data[start_idx:end_idx])
            inferred_data = method(eval_set)
            with open(os.path.join(f"{output_dir}", f"test_data_{i + 1}.json"), "w") as f:
                json.dump(inferred_data, f)
        except Exception as e:
            traceback.print_exc()
        start_idx += 50
        end_idx = start_idx + 50

def infer_from_adapter_and_llm(all_data, prompts_and_tools, model, tokenizer, retriever, checkpoints_dir, output_dir, method, include_numbers=None, skip_numbers=None, reverse=False, generation_config=None, add_onehop=False, start=0, step=50, end=20):
    checkpoints = [d for d in os.listdir(checkpoints_dir)]
    if include_numbers:
        checkpoints = [c for c in checkpoints if c.startswith("checkpoint-") and int(c.split("checkpoint-")[1]) 
        in include_numbers]
    if skip_numbers:
        checkpoints = [c for c in checkpoints if c.startswith("checkpoint-") and int(c.split("checkpoint-")[1]) 
        not in skip_numbers]
    checkpoints = sorted([os.path.join(checkpoints_dir, c) for c in checkpoints],
                         key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)),
    reverse=reverse)

    print("discovered checkpoints:", checkpoints)
    it = 0
    for ckpt_path in tqdm(checkpoints, desc="checkpoints processed"):
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

        # Unload adapter to save VRAM
        model.delete_adapter(f"adapter_checkpoint")
        torch.cuda.empty_cache()
        it+=1

    print("\nDone processing all checkpoints!")
