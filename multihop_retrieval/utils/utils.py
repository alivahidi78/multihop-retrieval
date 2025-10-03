import re
import os
import json
from tqdm import tqdm
import os
from enum import Enum
from string import Template
import string
from collections import Counter

class Task(Enum):
    INFO_CHECK = "info_check"
    SUBQUERY_CONSTRUCT = "subquery_construct"
    SHORT_ANSWER = "short_answer"

def cond_tqdm(iterable, use_tqdm=True, **kwargs):
    if use_tqdm:
        return tqdm(iterable, **kwargs)
    else:
        return iterable
    
def extract_subqueries(text: str):
    # Step 1: Find the <tool_call>...</tool_call> block
    match = re.search(r">\s*(\{.*?\})\s*<", text, re.DOTALL)
    if not match:
        return None, "No valid <...> block found"

    json_str = match.group(1)

    # Step 2: Try to parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    # Step 3: Validate format
    if "arguments" not in data or not isinstance(data["arguments"], dict):
        return None, "Missing or invalid 'arguments' key in JSON"

    # Step 4: Extract subqueries
    subqueries = [
        v for k, v in sorted(data["arguments"].items(), key=lambda x: int(x[0].replace("subquery", "")))
        if k.startswith("subquery")
    ]

    if not subqueries:
        return None, "No subqueries found"

    return subqueries, None

def get_tools(base_path, tools_path, task):
    with open(os.path.join(base_path, tools_path), 'r') as f:
        tools = json.load(f)
    if(task == Task.INFO_CHECK):
        task_title = "INFO_CHECK"
    elif(task == Task.SUBQUERY_CONSTRUCT):
        task_title = "SUBQUERY_CONSTRUCT"
    return tools[task_title]

def get_prompts(base_path, prompts_path, task, query, context=None):
    def substitute(obj, values):
        if isinstance(obj, dict):
            return {k: substitute(v, values) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(i, values) for i in obj]
        elif isinstance(obj, str):
            return Template(obj).safe_substitute(values)
        else:
            return obj
        
    # context = list(set(context))
    context_modified = []
    for i, (title, contents) in enumerate(context, start=1):
        # join all content parts into one string
        content_str = "".join(contents)
        # format as required
        context_modified.append(f"{i}. {title}:\n{content_str}")

    context_str = "\n".join(context_modified)
    
    with open(os.path.join(base_path, prompts_path), 'r') as f:
        template = json.load(f)
    if(task == Task.INFO_CHECK):
        task_title = "INFO_CHECK"
    elif(task == Task.SUBQUERY_CONSTRUCT):
        task_title = "SUBQUERY_CONSTRUCT"
    elif(task == Task.SHORT_ANSWER):
        task_title = "SHORT_ANSWER"
    values = {
        "query": query,
        "context":  context_str,
    }
    return substitute(template[task_title], values)

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
