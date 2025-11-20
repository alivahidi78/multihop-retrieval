import json
from multihop_retrieval.script_helpers.evaluation import assess_data
import pandas as pd
from multihop_retrieval.utils.inference_utils import Inferrer
from multihop_retrieval.utils import generic_utils as utils
from multihop_retrieval.utils.generic_utils import Task
from multihop_retrieval.trainer import MultihopGRPOTrainer

TOOLS_PATH = "./tools/var_4.json"

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
                supported_ret = [False]*len(supporting_facts)
                for x, f in enumerate(supporting_facts):
                    for c in context:
                        if(f[0] == c[0]):
                            supported_ret[x] = True
                if pr_label in prompts.keys():
                    for k in range(1, j):
                        try:
                            context.extend(d[Inferrer.task_label(Task.RETRIEVE, k)])
                        except:
                            print("missing retrieval", k, j, d.keys())
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
    with open(f"multihop_retrieval/{TOOLS_PATH}", "r") as f:
        prompts_and_tools = json.load(f)
    all_data = []
    for i in range(1, 21):
        try:
            with open(f"../data/inf_/4_vodh/test_data_{i}.json", "r") as f:
                all_data.extend(json.load(f))
        except:
            print("missing", i)
    result = assess_data(all_data, 0, reward_functions=get_reward_functions(prompts_and_tools))
    print(result)
    # df = pd.DataFrame(result)
    # df.to_csv("../data/inf_/4_vodh/summary.csv")