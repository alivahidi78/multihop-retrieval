"""Module includes utility tools used throughout the project.
"""
import re
import os
import json
import torch
import os
from enum import Enum
from string import Template
import string
from collections import Counter
import random

class Task(Enum):
    """An enum representing various generation sub-tasks.
    """
    RETRIEVE_INIT = "RETRIEVE_INIT"
    RETRIEVE = "RETRIEVE"
    INFO_CHECK = "INFO_CHECK"
    SHORT_ANSWER = "SHORT_ANSWER"
    PROVIDE_ANSWER = "PROVIDE_ANSWER"
    VERIFY_OR_DENY = "VERIFY_OR_DENY"
    SUBQUERY_CONSTRUCT = "SUBQUERY_CONSTRUCT"
    SUBQUERY_CONSTRUCT_WITH_HISTORY = "SUBQUERY_CONSTRUCT_WITH_HISTORY"

def method_task_id(task_id:Task):
    """Inserts the task_id as a parameter into the method.
    """
    def decorator(func):
        # func.task_id = task_id
        def wrapper(self, *args, **kwargs):
            if 'task_id' not in kwargs:
                kwargs['task_id'] = task_id
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def unbundle(data):
    """Unbundles a bundled list.
    """
    flat = [item for sublist in data for item in sublist]
    counts = [len(sublist) for sublist in data]
    return flat, counts

def rebundle(flat, counts):
    """Bundles an unbundled list.
    """
    reconstructed = []
    idx = 0
    for count in counts:
        reconstructed.append(flat[idx: idx + count])
        idx += count
    return reconstructed
    
def get_tools(prompts_and_tools, task: Task):
    """Returns tools from the prompts_and_tools dict, depending on the task.
    """
    return prompts_and_tools[task.value]["tools"]


def get_prompts(prompts_and_tools, task:Task, query, **kwargs):
    """Extracts relevant prompt from prompts_and_tools dict depending on task, query, 
    and other parameters.
    """
    def substitute(obj, values):
        if isinstance(obj, dict):
            return {k: substitute(v, values) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(i, values) for i in obj]
        elif isinstance(obj, str):
            return Template(obj).safe_substitute(values)
        else:
            return obj
        
    values = {
        "query": query,
    }
    
    if kwargs:
        values.update(kwargs)
        
    task_title = task.value
    return substitute(prompts_and_tools[task_title]["prompt"], values)

def context_to_string(context):
    """Converts a list of retrieved documents into a numbered list in one string.
    """
    context_modified = []
    for i, (title, contents) in enumerate(context, start=1):
        content_str = "".join(contents)
        context_modified.append(f"{i}. {title}:\n{content_str}")
    context_str = "\n".join(context_modified)
    return context_str

def list_to_numbered(l):
    """Converts list into numbered list in one string.
    """
    subquery_list = []
    for i, subquery in enumerate(l, start=1):
        subquery_list.append(f"{i}. {subquery}\n")
    past_queries_str = "\n".join(subquery_list)
    return past_queries_str

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
    """Computes exact match between two strings.
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """Computes F1-score between two strings.
    """
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
    """Removes all tensors from given nested dict and list structure.
    """
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
    """Removes all intermediate steps from the final inference result.
    """
    keys_to_remove = ["prompt", "prompt_ids", "prompt_mask", "thought_and_completion_ids", "completion_decoded"]
    data = remove_tensors(data)
    for i, d in enumerate(data):
        data[i] = {k: v for k, v in d.items() if k not in keys_to_remove}
    return data
    
def information_judgement(prompts_and_tools, response, task_id):
    """Returns judgement result and format check."""
    positive_tag = prompts_and_tools[task_id.value]["positive_tag"]
    negative_tag = prompts_and_tools[task_id.value]["negative_tag"]
    tag_group = prompts_and_tools[task_id.value]["tag_group"]
    pattern = prompts_and_tools[task_id.value]["pattern"]
    if not pattern:
        return True, False
    match = re.search(pattern, response) 
    if match:
        group_text = match.group(tag_group)
        if group_text is None:
            pass
        elif positive_tag in match.group(tag_group):
            return True, False
        elif negative_tag in match.group(tag_group):
            return False, False
    # print(f"Response malformed: {response}")
    # enough, malformed
    return False, True

def format_judgement(prompts_and_tools, response, task_id):
    """Judges the format of the response given task_id
    """
    pattern = prompts_and_tools[task_id.value]["pattern"]
    if not pattern:
        return True
    match = re.search(pattern, response) 
    if match:
        return True
    return False

def create_train_dev_test(original_train, original_dev, target_folder, seed=42, train_limit=5000, dev_limit=1000, test_limit=1000):
    """Creates train/dev/test subsets in the default configuration.
    """
    rng = random.Random(seed)
    
    with open(original_train, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [d for d in data if d.get("level") in ["hard", "medium"]]
    rng.shuffle(filtered)
    
    with open(os.path.join(target_folder, "train_set.json"), "w", encoding="utf-8") as f:
        json.dump(filtered[:train_limit], f, indent=2)
    
    with open(os.path.join(target_folder, "dev_set.json"), "w", encoding="utf-8") as f:
        json.dump(filtered[-dev_limit:], f, indent=2)
    
    with open(original_dev, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [d for d in data if d.get("level") in ["hard", "medium"]]
    rng.shuffle(filtered)
    with open(os.path.join(target_folder, "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(filtered[:test_limit], f, indent=2)