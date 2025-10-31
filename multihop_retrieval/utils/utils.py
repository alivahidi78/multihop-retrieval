import re
import os
import json
import torch
import os
from enum import Enum
from string import Template
import string
from collections import Counter

class Task(Enum):
    RETRIEVE_INIT = "RETRIEVE_INIT"
    RETRIEVE = "RETRIEVE"
    INFO_CHECK = "INFO_CHECK"
    SHORT_ANSWER = "SHORT_ANSWER"
    PROVIDE_ANSWER = "PROVIDE_ANSWER"
    VERIFY_OR_DENY = "VERIFY_OR_DENY"
    SUBQUERY_CONSTRUCT = "SUBQUERY_CONSTRUCT"
    SUBQUERY_CONSTRUCT_WITH_HISTORY = "SUBQUERY_CONSTRUCT_WITH_HISTORY"

def method_task_id(task_id:Task):
    def decorator(func):
        # func.task_id = task_id
        def wrapper(self, *args, **kwargs):
            return func(self, *args, task_id=task_id, **kwargs)
        return wrapper
    return decorator

def unbundle(data):
    flat = [item for sublist in data for item in sublist]
    counts = [len(sublist) for sublist in data]
    return flat, counts

def rebundle(flat, counts):
    reconstructed = []
    idx = 0
    for count in counts:
        reconstructed.append(flat[idx: idx + count])
        idx += count
    return reconstructed
    
def get_tools(prompts_and_tools, task: Task):
    return prompts_and_tools[task.value]["tools"]

def get_prompts(prompts_and_tools, task:Task, query, context=None, past_queries=None):
    def substitute(obj, values):
        if isinstance(obj, dict):
            return {k: substitute(v, values) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(i, values) for i in obj]
        elif isinstance(obj, str):
            return Template(obj).safe_substitute(values)
        else:
            return obj
        
    context_modified = []
    for i, (title, contents) in enumerate(context, start=1):
        # join all content parts into one string
        content_str = "".join(contents)
        # format as required
        context_modified.append(f"{i}. {title}:\n{content_str}")

    context_str = "\n".join(context_modified)
    
    task_title = task.value
    values = {
        "query": query,
        "context":  context_str,
    }
    return substitute(prompts_and_tools[task_title]["prompt"], values)

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If both are empty
        return int(gold_tokens == pred_tokens)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def remove_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return None  # or skip entirely
    elif isinstance(obj, dict):
        return {k: remove_tensors(v) for k, v in obj.items() if not isinstance(v, torch.Tensor)}
    elif isinstance(obj, list):
        return [remove_tensors(v) for v in obj if not isinstance(v, torch.Tensor)]
    elif isinstance(obj, tuple):
        return tuple(remove_tensors(v) for v in obj if not isinstance(v, torch.Tensor))
    else:
        return obj

def remove_intermediate_steps(data):
    keys_to_remove = ["prompt", "prompt_ids", "prompt_mask", "thought_and_completion_ids", "completion_decoded"]
    data = remove_tensors(data)
    for i, d in enumerate(data):
        data[i] = {k: v for k, v in d.items() if k not in keys_to_remove}
    return data
    
