"""Module includes methods useful for evaluation.
"""
import json, os, copy
import numpy as np

def load_data(folder_path):
    """Loads and returns inferred data from a folder. Includes every json file in this folder.
    """
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
    return all_data

def check_retrieval_matches(all_data, context_labels, discard_labels = None):
    """checks the recall rates of the inferences.

    Args:
        all_data (list): the inferred data.
        context_labels (str): label for key representing retrieved docs.
        discard_labels (str, optional): labels for keys whose values are to be subtracted.

    Returns:
        dict: what fraction of supporting facts was partially or fully retrieved.
    """
    total_items = 0
    fully_matched_items = 0
    partially_matched_items = 0
    unmatched_items = 0

    for item in all_data:
        total_items += 1
        context = []
        for l in context_labels:
            try:
                context.extend(copy.deepcopy(item[l]))
            except KeyError:
                pass
        context_titles = set(title for title, _ in context)
        if discard_labels:
            for dl in discard_labels:
                discard_titles = set(title for title, _ in (item.get(dl ,[])))
                context_titles -= discard_titles
        supporting_titles = set(title for title, _ in item['supporting_facts'])

        matches = supporting_titles.intersection(context_titles)
        match_count = len(matches)
        total_supporting = len(supporting_titles)

        if match_count == total_supporting:
            fully_matched_items += 1
        elif match_count > 0:
            partially_matched_items += 1
        else:
            unmatched_items += 1

    return {
        "full_match": round(fully_matched_items/total_items, 6),
        "part_match": round(partially_matched_items/total_items, 6),
        "no_match": round(unmatched_items/total_items, 6)
    }
    
def evaluate(all_data, reward_functions, min_llm=1, include_onehop=False, limit=None, iterations=2, final_label = None):
    """Assesses inferred data for evaluation.

    Args:
        all_data (list): inferred data.
        reward_functions (list): list of reward functions.
        min_llm (int, optional): minimum number of llm calls in inference. Defaults to 1.
        include_onehop (bool, optional): whether to include one hop. Defaults to False.
        limit (int, optional): filters the data until this number if specified. Defaults to None.
        iterations (int, optional): maximum inference iterations. Defaults to 2.
        final_label (int, optional): label for final output if different than default.

    Returns:
        dict: final assessment of data.
    """
    if not final_label:
        final_label = f"multihop{iterations}"
    if limit:
        all_data = all_data[:limit]
    def g_index(data):
        name = data["level"]
        return ["easy", "medium", "hard"].index(name)
    final_answers = [d[final_label] for d in all_data]
    llm_calls =  [d[f"llm_calls"] for d in all_data]
    sc_calls = [d[f"sc_calls"] for d in all_data]
    if include_onehop:
        onehop = [d[f"onehop"] for d in all_data]
    iter_c = [0, 0, 0]
    total_g = [0, 0, 0]
    multistep_count_g = [0, 0, 0]
    multistep_count = 0
    llm_calls_only_multi = 0
    sc_calls_only_multi = 0
    total_errors = 0
    retrieval_errors = 0
    info_errors = 0
    mult_data = []
    missing_ans = 0
    for d in all_data:
        if d[final_label]=="":
            missing_ans+=1
        total_g[g_index(d)]+=1
        if d["llm_calls"] > min_llm:
            mult_data.append(d)
            multistep_count += 1
            multistep_count_g[g_index(d)]+=1
            llm_calls_only_multi += d["llm_calls"]
            sc_calls_only_multi += d["sc_calls"]
        if d["error"]:
            total_errors += 1
            if d["error"] == "subquery":
                retrieval_errors += 1
            if d["error"] == "info":
                info_errors += 1
        iter_c[d["last_iter"]] += 1
    results = {
        "generation_count": float(np.mean(llm_calls)),
        "generation_count/multi": round(llm_calls_only_multi/(multistep_count+0.00000001),8),
        "iterations": float(np.mean(sc_calls)),
        "iterations/multi": round(sc_calls_only_multi/(multistep_count+0.00000001),8),
        "last_iter": iter_c,
        "multistep_calls": multistep_count,
        "multistep_calls/grouped": multistep_count_g,
        "multistep_calls/grouped/f": [round(a/(b+0.0000001),8) for a,b in zip(multistep_count_g, total_g)],
        "multistep_calls/f": multistep_count/len(all_data),
        "group_sizes": total_g,
        "errors": total_errors,
        "errors/f": total_errors/len(all_data),
        "errors/info": info_errors,
        "errors/info/f": info_errors/len(all_data), 
        "errors/subq": retrieval_errors,
        "errors/subq/f": retrieval_errors/len(all_data),
        "missing_ans": missing_ans
    }
    for r in reward_functions:
        values = r(all_data, final_answers, bundle_lengths=llm_calls)
        # multi_values = r(mult_data,  [d[f"multihop{iterations}"] for d in mult_data], bundle_lengths=llm_calls)
        group_avgs = []
        index = 0
        for size in llm_calls:
            group = values[index:index+size]
            index += size
            group_avgs.append(sum(group) / len(group))
        final_average = sum(group_avgs) / len(group_avgs)
        results.update({
            f"{r.__name__}/mean": float(final_average),
            # f"{r.__name__}/mult/mean": float(mavg),
        })
        if include_onehop:
            ovalues = r(all_data, onehop)
            oavg = float(np.mean(ovalues))
            results.update({
                f"{r.__name__}/onehop/mean": float(oavg),
            })
    return results   

def assess_data(all_data, index, reward_functions, iterations=2, min_llm=1, final_label = None):
    """A wrapper for `evaluate` that adds more data.

    Args:
        all_data (list): the inferred data.
        index (int): the index for the output.
        reward_functions (list): list of reward functions.
        iterations (int, optional): maximum number of inference iterations. Defaults to 2.
        min_llm (int, optional): minimum number of llm calls for inference. Defaults to 1.
        final_label (str, optional): output label if different than default. Defaults to None.

    Returns:
        dict: assessment of the data.
    """
    res = evaluate(all_data, reward_functions, min_llm= min_llm, final_label=final_label)
    res.update({
        "index": index,
        "count": len(all_data),
        "iter0_count": sum(d.get("sc_calls") == 0 for d in all_data),
        "iter0_f": sum(d.get("sc_calls") == 0 for d in all_data)/len(all_data),
        "iter1_count": sum(d.get("sc_calls") == 1 for d in all_data),
        "iter1_f": sum(d.get("sc_calls") == 1 for d in all_data)/len(all_data),
        "iter2_count": sum(d.get("sc_calls") == 2 for d in all_data),
        "iter2_f": sum(d.get("sc_calls") == 2 for d in all_data)/len(all_data),
        "init": check_retrieval_matches(all_data, ["context"]),
        "init+1": check_retrieval_matches(all_data, ["context", "step_ret_0"]),
        "init+r": check_retrieval_matches(all_data, ["context", "step_ret_0", "step_ret_1", "step_ret_2"]),
        "r": check_retrieval_matches(all_data, ["step_ret_0", "step_ret_1", "step_ret_2"]),
        "r-init": check_retrieval_matches(all_data, ["step_ret_0", "step_ret_1", "step_ret_2"], discard_labels = ["context"]),
        "ret0": check_retrieval_matches(all_data, ["step_ret_0"], discard_labels=["context"]),
        "ret1": check_retrieval_matches(all_data, ["step_ret_1"], discard_labels=["context", "step_ret_0"]),
    })
    return res
     

def assess_checkpoint_data(checkpoints_dir, numbers, reward_functions, iterations=2):
    """Assesses all data produced by various steps in the training.

    Args:
        checkpoints_dir (str): checkpoints output data directory.
        numbers (list): the numbers for checkpoints to be checked.
        reward_functions (list): list of reward functions used for evaluation.
        iterations (int, optional): number of maximum inference iterations. Defaults to 2.

    Returns:
        list: list of dict containing stepwise assessment of model training.
    """
    results = []
    for number in numbers:
        file_path = os.path.join(checkpoints_dir, f"./checkpoint-{number}")
        all_data = load_data(file_path)
        res = assess_data(all_data, number, reward_functions, iterations)
        results.append(res)
    return results