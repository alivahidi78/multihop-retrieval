import json, os, copy
import numpy as np

def load_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
    return all_data

def check_retrieval_matches(all_data, context_labels, discard_label = None):
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
        if discard_label:
            discard_titles = set(title for title, _ in item[discard_label])
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
        "full_match": fully_matched_items/total_items,
        "part_match": partially_matched_items/total_items,
        "no_match": unmatched_items/total_items
    }
    
def evaluate(all_data, prompts_and_tools, reward_functions, min_llm=1, include_onehop=False, limit=1000, iterations=2):
    all_data = all_data[:limit]
    def g_index(data):
        name = data["level"]
        return ["easy", "medium", "hard"].index(name)
    final_answers = [d[f"multihop{2}"] for d in all_data]
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
        if d[f"multihop{iterations}"]=="":
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

def assess_checkpoint_data(checkpoints_dir, numbers, prompts_and_tools):
    results = []
    for number in numbers:
        all_data = load_data(os.path.join(checkpoints_dir, f"./checkpoint-{number}"))
        res = evaluate(all_data, prompts_and_tools)
        res.update({
            "index": number,
            "count": len(all_data),
            "init": check_retrieval_matches(all_data, ["context"]),
            "init+1": check_retrieval_matches(all_data, ["context", "step_ret_0"]),
            "init+r": check_retrieval_matches(all_data, ["context", "step_ret_0", "step_ret_1", "step_ret_2"]),
            "r": check_retrieval_matches(all_data, ["step_ret_0", "step_ret_1", "step_ret_2"]),
            "r-init": check_retrieval_matches(all_data, ["step_ret_0", "step_ret_1", "step_ret_2"], discard_label = "context")
        })
        results.append(res)
    return results