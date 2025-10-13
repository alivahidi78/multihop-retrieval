import torch, re, json, time
from transformers.generation.configuration_utils import GenerationConfig
from multihop_retrieval.utils import utils
from multihop_retrieval.utils.outlines_transformers import CustomizedTransformers
from .utils import Task
from .retrieval import Retriever
from outlines.types import Regex

# TODO Priority
# 1. Does the tokenizer properly encode <tool_call> tags? 
# Possibly include the tags as part of the prompt instead.
# 2. The grammar needs to be much more restrictive and more specialized for separate
# calls by different methods.
# 3. Specifically about query construction, a simple list might function better than
# the same tool call.
BASIC_CALL_GRAMMAR = Regex(r"<tool_call>\n\{.*\}\n</tool_call>")

class Inferrer:
    def __init__(self, retriever, model, tokenizer, prompts_path, tools_path):
        self.input_preparation_func = None
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.prompts_path = prompts_path
        self.tools_path = tools_path
        self.base_path = retriever.base_path
    
    @classmethod    
    def create_with_retriever(cls,  base_path, wiki_path, embedder, index, metadata, model, tokenizer, prompts_path, tools_path):
        return cls(Retriever(base_path, wiki_path, embedder, index, metadata), model, tokenizer, prompts_path, tools_path) 
      
    #################################### retrieval ####################################  
      
    def retrieve_init(self, data, use_tqdm=True, logs=True):
        query_count= 0
        error_count= 0
        for i in utils.cond_tqdm(range(len(data)), use_tqdm=use_tqdm, desc=f"init_ret"):
            try:
                value = data[i][f"question"]
                query_count += 1
            except KeyError:
                continue
            else:
                context_add = self.retriever.retrieve_info_rag([value])
                flat_context_add = [item for sublist in context_add for item in sublist]
                result = [[d["title"], d["full_text"]] for d in flat_context_add]
                data[i][f"context"] = result
        if logs:
            print(f"init total queries: {query_count}\nerrors: {error_count}")
        return data
    
    def retrieve_info_iter(self, data, iteration, use_tqdm=True, logs=True):
        query_count= 0
        error_count= 0
        for i in utils.cond_tqdm(range(len(data)), use_tqdm=use_tqdm, desc=f"iter_{iteration}_2_ret"):
            try:
                value = data[i][f"nothink_query_iteration_{iteration}"]
                query_count += 1
            except KeyError:
                continue
            subqueries, error_desc = utils.extract_subqueries(value)
            if(subqueries == None):
                error_count += 1
            else:
                context_add = self.retriever.retrieve_info_rag(subqueries)
                flat_context_add = [item for sublist in context_add for item in sublist]
                result = [[d["title"], d["full_text"]] for d in flat_context_add]
                data[i][f"nothink_retrieve_iteration_{iteration}"] = result
        if logs:
            print(f"{iteration} total queries: {query_count}\nerrors: {error_count}")
        return data
    
    #################################### generation #################################### 
    
    def one_hop(self, query, context, generation_config, thinking=False):
        prompt = self._get_prompts(Task.SHORT_ANSWER, query, context)
        llm_res = self._call_llm(generation_config, prompt, enable_thinking=thinking, skip_special_tokens=True)
        #TODO deal with thought
        return llm_res["completion_decoded"]
    
    def info_check_iter(self, data, iteration, generation_config, enforce_grammar=True, add_onehop = False, ignore_ids=False, use_tqdm = True):
        prompt_list = []
        response_list = []
        col_name = f"nothink_gen_iteration_{iteration}"
        for i in utils.cond_tqdm(range(0, len(data), 1), use_tqdm = use_tqdm, desc=f"iter_{iteration}_0_gen"):
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            if not (iteration == 0):
                try:
                    for k in range(0, iteration):
                        context = context.copy()
                        context.extend(selected[f"nothink_retrieve_iteration_{k}"])
                except KeyError:
                    # TODO do caching before this
                    continue

            tools = self._get_tools(Task.INFO_CHECK)
            prompt = self._get_prompts(Task.INFO_CHECK, query, context)
            prompt_list.append(prompt)

            grammar = None
            if enforce_grammar:
                grammar = BASIC_CALL_GRAMMAR
            
            llm_res = self._call_llm(generation_config, prompt, tools, grammar=grammar, enable_thinking=False, skip_special_tokens=False)
            predicted_m = llm_res["completion_decoded"]
            # TODO deal with thought
            response_list.append(predicted_m)

            if add_onehop:
                # TODO deal with onehop training
                predicted_o = self.one_hop(query, context, generation_config)
            try:
                data[i][col_name] = predicted_m
                if not ignore_ids:
                    data = self._append_llm_res(data, llm_res, i, iteration, "info")
                if add_onehop:
                    data[i]["onehop"] = predicted_o
            except Exception as e:
                print(f"exception at {i}")
        return data, prompt_list, response_list
    
    def _append_llm_res(self, data, llm_res, data_num, iteration, step_name):
        i = data_num
        labels = ["prompt", "prompt_ids", "prompt_mask", "thought_and_completion_ids",
                  "completion_decoded",
                  ]
        for label in labels:
            if label not in data[i].keys():
                data[i][label] = dict()
            data[i][label].update({
                f"{iteration}_{step_name}": llm_res[label],
            })   
            # if iteration not in data[i][label].keys():
            #     data[i][label][iteration] = dict()
            # data[i][label][iteration].update({
            #     step_name: llm_res[label]
            # })
            # data[i][label][iteration].update({
            #     step_name: llm_res[label]
            # })            
        return data    
    
    def _get_tools(self, task):
        return utils.get_tools(self.base_path, self.tools_path, task)
        
    def _get_prompts(self, task, query, context = None):
        return utils.get_prompts(self.base_path, self.prompts_path, task, query, context)
    
    def _check_done(self, response):
        return "information_is_sufficient" in response
    
    def _call_llm(self, generation_config, prompt, grammar = None, tools = None, skip_special_tokens=False, enable_thinking=False):
        #TODO is it never batched?
        text = self.tokenizer.apply_chat_template(
            prompt,
            tools = tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = enable_thinking)
        
        prompt_mask = None
        
        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            # add_special_tokens=False,
        ).to("cuda")  
        
        if self.input_preparation_func:
            inputs = self.input_preparation_func(inputs)
            prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
            inputs = {}
            inputs["input_ids"], inputs["attention_mask"] = prompt_ids, prompt_mask
        
        if grammar:
            outlines_transformers = CustomizedTransformers(self.model, self.tokenizer)
            output_type = grammar
            outputs = outlines_transformers.generate(text, inputs, output_type=output_type, generation_config=generation_config)
            
        else:
            outputs = self.model.generate(
                **inputs,
                generation_config = generation_config
            )
        
        # Note: thinking is disabled for now
        if(enable_thinking):
            raise NotImplementedError()
        # modified_config = GenerationConfig.from_dict(generation_config.to_dict())
        # modified_config.max_new_tokens = 25
        
        # If still thinking
        # if enable_thinking and (151668 not in outputs[0]):
        #     extended_ids = torch.cat([outputs[0], torch.tensor([151668], device="cuda")])
        #     outputs = self.model.generate(
        #         input_ids=extended_ids.unsqueeze(0),
        #         generation_config = modified_config
        #     )
        
        # # If not done    
        # if self.tokenizer.eos_token_id not in outputs[0]:
        #     outputs = self.model.generate(
        #         input_ids=extended_ids.unsqueeze(0),
        #         generation_config = modified_config
        #     )

        output_ids = outputs[0]
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        prompt_ids = outputs[0][:input_len]

        try:
            end_thinking_index = generated_ids.tolist().index(151668)
            thought_ids = generated_ids[:end_thinking_index+1]
            generated_ids = generated_ids[end_thinking_index+1:]
            thought = self.tokenizer.decode(thought_ids, skip_special_tokens=skip_special_tokens)
        except ValueError:
            thought = None

        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)
        
        if prompt_mask.dim() == 2: #FIXME
            prompt_mask = prompt_mask.squeeze(0)
            
        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "prompt_ids": prompt_ids,
            "thought_and_completion_ids": generated_ids,
            "thought_decoded": thought,
            "completion_decoded": completion,
        }
    
    #################################### subquery ####################################
    
    def infer_subquery_iter(self, data, iteration, generation_config, enforce_grammar=True, ignore_ids=False, use_tqdm=True):
        prompt_list = []
        response_list = []
        for i in utils.cond_tqdm(range(0, len(data), 1), use_tqdm=use_tqdm, desc=f"iter_{iteration}_1_2q"):
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            try:
                prev_response = selected[f"nothink_gen_iteration_{iteration}"]
            except KeyError:
                continue

            if(self._check_done(prev_response)):
                continue
            tools = self._get_tools(Task.SUBQUERY_CONSTRUCT)
            prompt = self._get_prompts(Task.SUBQUERY_CONSTRUCT, query, context)
            prompt_list.append(prompt)
            
            grammar = None
            if enforce_grammar:
                grammar = BASIC_CALL_GRAMMAR
            
            llm_res = self._call_llm(generation_config, prompt, grammar=grammar, tools=tools)
            subquery = llm_res["completion_decoded"]
            response_list.append(subquery)
            
            try:
                data[i][f"nothink_query_iteration_{iteration}"] = subquery
                if not ignore_ids:
                    data = self._append_llm_res(data, llm_res, i, iteration, "subq")
            except Exception as e:
                print(f"exception at {i}")
        return data, prompt_list, response_list
    
    #################################### main ####################################
        
    def finalize_data(self, data, iterations = 3):
        tool_pattern = re.compile(
            r">\s*(\{.*?\})\s*<",
            re.DOTALL
        )

        for d in data:
            d[f"multihop{iterations}"] = ""
            d["error"] = ""

            # check from iteration_3 down to iteration_0
            for i in range(iterations, -1, -1):
                key = f"nothink_gen_iteration_{i}"
                query_key= f"nothink_query_iteration_{i}"
                if key in d and isinstance(d[key], str):
                    content = d[key]

                    if query_key in d:
                        d["error"] = "query"
                        break

                    if "information_not_sufficient" in content:
                        d["error"] = "info"
                        break

                    match = tool_pattern.search(content)
                    if not match:
                        d["error"] = "format"
                    else:
                        try:
                            tool_call = json.loads(match.group(1))
                            name = tool_call.get("name", "")
                            args = tool_call.get("arguments", {})

                            if name == "information_is_sufficient":
                                d[f"multihop{iterations}"] = args.get("answer", "")
                            else:
                                d["error"] = "format"
                        except Exception:
                            d["error"] = "format"

                    break  # stop after first valid key
        return data
    
    #TODO put some of the parameters inside config
    def infer(self, data, generation_config, enforce_grammar=True, iterations=3, start_iter=0, use_tqdm=False, logs=False, add_onehop=False, calculate_time=False, ignore_ids=False, input_preparation_func=None):
        self.input_preparation_func = input_preparation_func
        timer_start = time.time()
        if start_iter == 0:
            data = self.retrieve_init(data, use_tqdm=use_tqdm, logs=logs)
        iterations_processed = 0
        for i in range(start_iter, iterations):
            data, _, __ = self.info_check_iter(data, i, generation_config, enforce_grammar=enforce_grammar, add_onehop=(add_onehop and i == 0), ignore_ids=ignore_ids, use_tqdm=use_tqdm)
            data, _, __ = self.infer_subquery_iter(data, i, generation_config, enforce_grammar=enforce_grammar, ignore_ids=ignore_ids, use_tqdm=use_tqdm)
            data = self.retrieve_info_iter(data, i, use_tqdm=use_tqdm, logs=logs)
            iterations_processed += 1
        if iterations_processed == iterations - start_iter:
            data, _, __ = self.info_check_iter(data, iterations, generation_config, enforce_grammar=enforce_grammar, use_tqdm=use_tqdm)
        data = self.finalize_data(data, iterations)
        timer_end = time.time()
        
        elapsed_time = timer_end - timer_start
        hours = int(elapsed_time // 3600)
        minutes = int(elapsed_time % 3600 // 60)
        seconds = int(elapsed_time % 60)
        if calculate_time:
            print(f"inference execution time: {hours}h {minutes}m {seconds}s")
        return data