import re
import os
import json
from tqdm import tqdm
import os
from enum import Enum
from string import Template

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

def flatten_and_count(data):
    flat = [item for sublist in data for item in sublist]
    counts = [len(sublist) for sublist in data]
    return flat, counts

def reconstruct_2d_list(flat, counts):
    reconstructed = []
    idx = 0
    for count in counts:
        reconstructed.append(flat[idx: idx + count])
        idx += count
    return reconstructed