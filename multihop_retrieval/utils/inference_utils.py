import torch, re, json, time, os
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig
from multihop_retrieval.utils import generic_utils as utils
from multihop_retrieval.utils.outlines_transformers import OutlinesWrapper
from .generic_utils import Task, method_task_id
from .retrieval_utils import Retriever
from outlines.types import Regex
from collections.abc import Mapping
import traceback

class InferrerConfig:
    def __init__(self, **kwargs): 
        defaults = {
            "generation_config": GenerationConfig(
                    max_new_tokens=128,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=None,),
            "enforce_grammar": True,
            "iterations": 2,
            "use_tqdm": False,
            "logs": False,
            "add_onehop": False,
            "calculate_time": False,
            "remove_intermediate_steps": False,
            "remove_tensors":False,
            "device": "cuda",
            "frozen_tasks": None, # Has to be excluded also in trainer
            # list of tasks
            "frozen_models": None
            # {Task.A: model_1}
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)
            
class Inferrer:
    dict_labels = {
        Task.RETRIEVE: "step_ret",
        Task.INFO_CHECK: "step_inf",
        Task.PROVIDE_ANSWER: "step_pro", # variation of info_check
        Task.VERIFY_OR_DENY: "step_vod",
        Task.SUBQUERY_CONSTRUCT: "step_que",
        Task.SUBQUERY_CONSTRUCT_WITH_HISTORY: "step_quh" # variation of subq_construct
    }
    
    @staticmethod
    def task_label(task, iteration):
        return f"{Inferrer.dict_labels[task]}_{iteration}"
    
    def task_func(self, task):
        if task == Task.RETRIEVE_INIT: return self.retrieval_init
        if task == Task.RETRIEVE: return self.retrieval_iter
        if task == Task.INFO_CHECK: return self.info_check_iter
        if task == Task.PROVIDE_ANSWER: return self.provide_answer_iter
        if task == Task.SUBQUERY_CONSTRUCT: return self.subq_construct_iter
        if task == Task.SUBQUERY_CONSTRUCT_WITH_HISTORY: return self.subq_construct_history_iter
        if task == Task.VERIFY_OR_DENY: return self.verify_or_deny_iter
    
    def __init__(self, retriever, model, tokenizer, prompts_and_tools, inferrer_config=None, no_cache=False):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.prompts_and_tools = prompts_and_tools
        self.no_cache = no_cache
        self.config = inferrer_config if inferrer_config else InferrerConfig()
    
    @classmethod    
    def create_with_retriever(cls, wiki_path, embedder, index, metadata, model, tokenizer, prompts_and_tools, inferrer_config=None):
        return cls(Retriever(wiki_path, embedder, index, metadata), model, tokenizer, prompts_and_tools, inferrer_config) 
      
    #################################### retrieval ####################################  
    
    @method_task_id(Task.RETRIEVE_INIT)
    def retrieval_init(self, data, iteration=-1, prev_task_id=None, task_id=None):
        query_count= 0
        error_count= 0
        for i in tqdm(range(len(data)), disable=not self.config.use_tqdm, desc=f"init_ret"):
            try:
                value = data[i][f"question"]
                query_count += 1
            except KeyError:
                error_count += 1
                continue
            else:
                context_add = self.retriever.retrieve_info_rag([value])
                flat_context_add = [item for sublist in context_add for item in sublist]
                result = [[d["title"], d["full_text"]] for d in flat_context_add]
                data[i]["context"] = result
        if self.config.logs:
            print(f"init total queries: {query_count}\nerrors: {error_count}")
        return data
    
    @method_task_id(Task.RETRIEVE) 
    def retrieval_iter(self, data, iteration, prev_task_id, task_id=None):
        query_count= 0
        error_count= 0
        for i in tqdm(range(len(data)), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_{Inferrer.dict_labels[task_id]}"):
            try:
                subqueries = data[i][Inferrer.task_label(prev_task_id, iteration)]
                query_count += 1
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue
            
            if(not subqueries):
                error_count += 1
            else:
                context_add = self.retriever.retrieve_info_rag(subqueries)
                flat_context_add = [item for sublist in context_add for item in sublist]
                result = [[d["title"], d["full_text"]] for d in flat_context_add]
                data[i][Inferrer.task_label(task_id, iteration)] = result
        if self.config.logs:
            print(f"{iteration} total queries: {query_count}\nerrors: {error_count}")
        return data
    
    #################################### info_check #################################### 
    
    @method_task_id(Task.INFO_CHECK)    
    def info_check_iter(self, data, iteration, prev_task_id, task_id=None):
        col_name = Inferrer.task_label(task_id, iteration)
        for i in tqdm(range(0, len(data), 1), disable = not self.config.use_tqdm, desc=f"iter_{iteration}_{Inferrer.dict_labels[task_id]}"):
            selected = data[i]
            
            if iteration != 0 and Inferrer.task_label(prev_task_id, iteration-1) not in selected.keys():
                # the sample is skipped since there's no previous iteration.
                continue
            
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            context_str = utils.context_to_string(context)
            prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context_str)

            grammar = None
            if self.config.enforce_grammar:
                grammar = Regex(self.prompts_and_tools[task_id.value]["pattern"])
            
            #TODO batch this
            llm_res = self._call_llm(prompt, tools=tools, grammar=grammar, enable_thinking=False, skip_special_tokens=False)
            
            if llm_res["error"]:
                if "call_error" not in data[i].keys():
                    data[i]["call_error"] = []
                data[i]["call_error"].append(Inferrer.task_label(task_id, iteration))
                
            predicted_m = llm_res["completion_decoded"]

            if self.config.add_onehop and iteration==0:
                predicted_o = self._one_hop(query, context)
            try:
                data[i][col_name] = predicted_m
                data = self._append_llm_res(data, llm_res, i, iteration, task_id)
                if self.config.add_onehop and iteration==0:
                    data[i]["onehop"] = predicted_o
            except Exception as e:
                print(f"exception at {i}")
        return data
    
    ### short_answer ###
    def _one_hop(self, query, context, thinking=False):
        context_str = utils.context_to_string(context)
        prompt = utils.get_prompts(self.prompts_and_tools, Task.SHORT_ANSWER, query=query, context=context_str)
        llm_res = self._call_llm(prompt, enable_thinking=thinking, skip_special_tokens=True)
        return llm_res["completion_decoded"]
    
    def _append_llm_res(self, data, llm_res, data_num, iteration, task_id):
        i = data_num
        labels = ["prompt", "prompt_ids", "prompt_mask", "thought_and_completion_ids",
                  "completion_decoded",
                  ]
        for label in labels:
            if label not in data[i].keys():
                data[i][label] = dict()
            data[i][label].update({
                Inferrer.task_label(task_id, iteration): llm_res[label],
            })          
        return data    
    
    def _check_done(self, response, task_id):
        return utils.information_judgement(self.prompts_and_tools, response, task_id)[0]
    
    #################################### provide_answer ####################################
    @method_task_id(Task.PROVIDE_ANSWER)
    def provide_answer_iter(self, data, iteration, prev_task_id, task_id=None):
        return self.info_check_iter(data, iteration, prev_task_id, task_id=task_id)
    
    #################################### verify_or_deny ####################################
    @method_task_id(Task.VERIFY_OR_DENY)
    def verify_or_deny_iter(self, data, iteration, prev_task_id, task_id=None):
        col_name = Inferrer.task_label(task_id, iteration)
        for i in tqdm(range(0, len(data), 1), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_{Inferrer.dict_labels[task_id]}"):
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            info_pattern = self.prompts_and_tools[prev_task_id.value]["pattern"]
            answer_group = self.prompts_and_tools[prev_task_id.value]["answer_group"]
            context = self._get_deduplicated_context(context, iteration, selected)
            try:
                prev_response = selected[Inferrer.task_label(prev_task_id, iteration)]
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue
            answer_match = re.search(info_pattern, prev_response)
            if(not self._check_done(prev_response, prev_task_id)):
                # response is no
                data[i][col_name] = self.prompts_and_tools[task_id.value]["negative_tag"]
            elif not answer_match or not answer_match.group(answer_group):
                data[i][col_name] = self.prompts_and_tools[task_id.value]["negative_tag"]
            else:    
                response = answer_match.group(answer_group)
                context_str = utils.context_to_string(context)
                prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context_str, response=response)

                grammar = None
                if self.config.enforce_grammar:
                    grammar = Regex(self.prompts_and_tools[task_id.value]["pattern"])
                
                #TODO batch this
                llm_res = self._call_llm(prompt, tools=tools, grammar=grammar, enable_thinking=False, skip_special_tokens=False)
                
                if llm_res["error"]:
                    if "call_error" not in data[i].keys():
                        data[i]["call_error"] = []
                    data[i]["call_error"].append(Inferrer.task_label(task_id, iteration))

                predicted_m = llm_res["completion_decoded"]

                try:
                    data[i][col_name] = predicted_m
                    data = self._append_llm_res(data, llm_res, i, iteration, task_id)
                except Exception as e:
                    print(f"exception at {i}")
        return data
                
    
    #################################### subq_construct ####################################
    
    @method_task_id(Task.SUBQUERY_CONSTRUCT)
    def subq_construct_iter(self, data, iteration, prev_task_id, task_id=None):
        for i in tqdm(range(0, len(data), 1), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_{Inferrer.dict_labels[task_id]}"):
            subq_json = self.prompts_and_tools[task_id.value]
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            
            try:
                prev_response = selected[Inferrer.task_label(prev_task_id, iteration)]
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue

            if(self._check_done(prev_response, prev_task_id)):
                continue
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            context_str = utils.context_to_string(context)
            prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context_str)
            
            grammar = None
            if self.config.enforce_grammar:
                grammar = Regex(subq_json["pattern"])
            
            #TODO batch this
            llm_res = self._call_llm(prompt, grammar=grammar, tools=tools)
            
            if llm_res["error"]:
                if "call_error" not in data[i].keys():
                    data[i]["call_error"] = []
                data[i]["call_error"].append(Inferrer.task_label(task_id, iteration))
                
            llm_output = llm_res["completion_decoded"]
            pattern = subq_json["pattern"]
            regex_groups = subq_json["regex_groups"]
            match = re.search(pattern, llm_output)
            subqueries = []
            data = self._append_llm_res(data, llm_res, i, iteration, task_id)
            if match:
                for g in regex_groups:
                    if match.group(g):
                        subqueries.append(match.group(g))
            else:
                #FIXME warning
                print(f"The following query attempt is malformed:\n{llm_output}.")
                continue
            try:
                data[i][Inferrer.task_label(task_id, iteration)] = subqueries
            except Exception as e:
                print(f"exception at {i}")
        return data
    
    #################################### subq_construct_history ####################################
    
    @method_task_id(Task.SUBQUERY_CONSTRUCT_WITH_HISTORY)
    def subq_construct_history_iter(self, data, iteration, prev_task_id, task_id=None):
        for i in tqdm(range(0, len(data), 1), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_{Inferrer.dict_labels[task_id]}"):
            subqh_json = self.prompts_and_tools[task_id.value]
            selected = data[i]
            query = selected["question"]
            init_context = self._get_deduplicated_context(selected["context"], iteration)
            
            try:
                prev_response = selected[Inferrer.task_label(prev_task_id, iteration)]
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue

            if(self._check_done(prev_response, prev_task_id)):
                continue
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            context_str = utils.context_to_string(init_context)
            
            if iteration == 0:
                add_context_str = "\n"
                past_queries_str = "\n"
            else:  
                add_context = self._get_deduplicated_context([], iteration, selected)
                add_context_str = utils.context_to_string(add_context)
                past_queries = []
                for k in range(0, iteration):
                    past_queries.extend(selected[Inferrer.task_label(task_id, k)])
                past_queries_str = utils.list_to_numbered(past_queries)
                
            prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context_str, past_queries=past_queries_str, additional_context=add_context_str)
            
            grammar = None
            if self.config.enforce_grammar:
                grammar = Regex(subqh_json["pattern"])
            
            #TODO batch this
            llm_res = self._call_llm(prompt, grammar=grammar, tools=tools)
            
            if llm_res["error"]:
                if "call_error" not in data[i].keys():
                    data[i]["call_error"] = []
                data[i]["call_error"].append(Inferrer.task_label(task_id, iteration))
                
            llm_output = llm_res["completion_decoded"]
            pattern = subqh_json["pattern"]
            regex_groups = subqh_json["regex_groups"]
            match = re.search(pattern, llm_output)
            subqueries = []
            data = self._append_llm_res(data, llm_res, i, iteration, task_id)
            if match:
                for g in regex_groups:
                    if match.group(g):
                        subqueries.append(match.group(g))
            else:
                #FIXME warning
                print(f"The following query attempt is malformed:\n{llm_output}.")
                continue
            try:
                data[i][Inferrer.task_label(task_id, iteration)] = subqueries
            except Exception as e:
                print(f"exception at {i}")
        return data
    
    #################################### internal ####################################
    
    def _call_llm(self, prompt, grammar = None, tools = None, skip_special_tokens=False, enable_thinking=False):
        try:
            text = self.tokenizer.apply_chat_template(
                prompt,
                tools = tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = enable_thinking)
            
            prompt_mask = None
            
            try:
                text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                inputs = self.tokenizer(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    # add_special_tokens=False,
                ).to(self.config.device)  
            except Exception as e:
                print(prompt)
                print(tools)
                print(text)
                raise e
            
            inputs = self._prepare_inputs(inputs)
            prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
            inputs = {}
            inputs["input_ids"], inputs["attention_mask"] = prompt_ids, prompt_mask
            
            if grammar:
                outlines_wrapped = OutlinesWrapper(self.model, self.tokenizer)
                output_type = grammar
                outputs = outlines_wrapped.generate(text, inputs, output_type=output_type, generation_config=self.config.generation_config, disable_compile=True)
                
            else:
                outputs = self.model.generate(
                    **inputs,
                    generation_config = self.config.generation_config, disable_compile=True
                )
            
            # Note: thinking is disabled for now
            if(enable_thinking):
                raise NotImplementedError("Thinking is currently disabled.")

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
                "error": False
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "prompt": "",
                "prompt_mask": [],
                "prompt_ids": [],
                "thought_and_completion_ids": [],
                "thought_decoded": "",
                "completion_decoded": "",
                "error": True
            }
    
    def _get_deduplicated_context(self, context, iteration, selected_datum=None):
        context = context.copy()
        if selected_datum:
            if not (iteration == 0):
                try:
                    for k in range(0, iteration):
                        context.extend(selected_datum[Inferrer.task_label(Task.RETRIEVE, k)])
                except KeyError:
                    # FIXME warning
                    pass
        unique = {}
        for k, v in context:
            unique[k] = v 
        context = [[k, v] for k, v in unique.items()]
        return context
    
    # Transplanted from transformers Trainer._prepare_input    
    def _prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.config.device}
            return data.to(**kwargs)
        return data
    
    # Transplanted from transformers Trainer._prepare_inputs 
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        return inputs
    
    #################################### finalize ####################################
        
    def finalize_data_basic(self, data, iterations):
        raise NotImplementedError()
    
    def finalize_data_vod(self, data, iterations):
        raise NotImplementedError()
    
    def finalize_data_hist(self, data, iterations):
        raise NotImplementedError()

    def finalize_data_vod_hist(self, data, iterations):
        data = self.finalize_data(data)
        info_pattern = self.prompts_and_tools[Task.PROVIDE_ANSWER.value]["pattern"]
        answer_group = self.prompts_and_tools[Task.PROVIDE_ANSWER.value]["answer_group"]
        
        pr_label = Inferrer.dict_labels[Task.PROVIDE_ANSWER]
        sc_label = Inferrer.dict_labels[Task.SUBQUERY_CONSTRUCT_WITH_HISTORY]
        rv_label = Inferrer.dict_labels[Task.RETRIEVE]
        vd_label = Inferrer.dict_labels[Task.VERIFY_OR_DENY]

        for d in data:
            d["pr_calls"] = sum(1 for key in d["prompt"] if key.startswith(pr_label))
            d["sc_calls"] = sum(1 for key in d["prompt"] if key.startswith(sc_label))
            d["vd_calls"] = sum(1 for key in d["prompt"] if key.startswith(vd_label))
            d["rv_calls"] = sum(1 for key in d if key.startswith(rv_label)) + 1
            d[f"multihop{iterations}"] = ""
            d["error"] = ""

            if Inferrer.task_label(Task.PROVIDE_ANSWER, 0) not in d:
                d["error"] = "llm_start"
                continue
            # check from iteration_max down to iteration_0
            for i in range(iterations, -1, -1):
                pr_key = f"{pr_label}_{i}"
                query_key= f"{sc_label}_{i}"
                vod_key = f"{vd_label}_{i}"
                ret_key= f"{rv_label}_{i}"
                if pr_key in d and isinstance(d[pr_key], str):
                    d["last_iter"] = i
                    if ret_key in d:
                        d["error"] = "generation"  
                    elif query_key in d:
                        d["error"] = "retrieval"
                    elif vod_key in d:
                        d["error"] = "subquery"
                    elif i < iterations:
                        d["error"] = "vod"
                    else:
                        enough, malformed = utils.information_judgement(self.prompts_and_tools, d[pr_key], Task.PROVIDE_ANSWER)
                        if malformed:
                            d["error"] = "format"
                        elif i == iterations and not enough:
                            d["error"] = "info"
                        elif enough:
                            match = re.search(info_pattern, d[pr_key])
                            if match.group(answer_group):
                                d[f"multihop{iterations}"] = match.group(answer_group)
                            else:
                                d["error"] = "format"
                        else:
                            d["error"] = "unknown"
                    break
                if i == 0 and not d[f"multihop{iterations}"]:
                    d["error"] = "unknown"    
        return data
    
    #################################### generic main ####################################
        
    def finalize_data(self, data, iterations = 2):
        for d in data:
            d["llm_calls"] = len(d["prompts"])
        return data
    
    def infer(self, task_list, data, start_iter=0, finalize_method=None):
        # example:
        # task_list = {
        #     "before": [Task.RETRIEVE_INIT],
        #     "main": [Task.INFO_CHECK, Task.SUBQUERY_CONSTRUCT, Task.RETRIEVE],
        #     "after": [Task.INFO_CHECK]
        # }
        timer_start = time.time()
        
        if start_iter == 0:
            for task in task_list["before"]:
                data = self.task_func(task)(data, -1, None)
                
        iterations_processed = 0
        for i in range(start_iter, self.config.iterations):
            if self.no_cache and self.config.device == "cuda":
                torch.cuda.empty_cache()
            for idx, task in enumerate(task_list["main"]):
                data = self.task_func(task)(data, i, prev_task_id=task_list["main"][idx-1])
            iterations_processed += 1
        
        for task in task_list["after"]:
            data = self.task_func(task)(data, self.config.iterations, prev_task_id=task["main"][-1])
        if finalize_method:
            data = finalize_method(data, self.config.iterations)
        else:    
            data = self.finalize_data(data, self.config.iterations)
        if self.config.remove_tensors:
            data = utils.remove_tensors(data)
        if self.config.remove_intermediate_steps:
            data = utils.remove_intermediate_steps(data)
        timer_end = time.time()
        
        elapsed_time = timer_end - timer_start
        hours = int(elapsed_time // 3600)
        minutes = int(elapsed_time % 3600 // 60)
        seconds = int(elapsed_time % 60)
        if self.config.calculate_time:
            print(f"inference execution time: {hours}h {minutes}m {seconds}s")
        return data
    
    #################################### main ####################################
    
    def infer_basic(self, data, start_iter=0):
        task_list = {
            "before": [Task.RETRIEVE_INIT],
            "main": [Task.INFO_CHECK, Task.SUBQUERY_CONSTRUCT, Task.RETRIEVE],
            "after" : [Task.INFO_CHECK]
            }
        return self.infer(task_list, data, start_iter=start_iter, finalize_method=self.finalize_data_basic)
    
    def infer_vod(self, data, start_iter=0):
        task_list = {
            "before": [Task.RETRIEVE_INIT],
            "main": [Task.PROVIDE_ANSWER, Task.VERIFY_OR_DENY, Task.SUBQUERY_CONSTRUCT, Task.RETRIEVE],
            "after" : [Task.PROVIDE_ANSWER]
            }
        return self.infer(task_list, data, start_iter=start_iter, finalize_method=self.finalize_data_vod)
    
    def infer_hist(self, data, start_iter=0):
        task_list = {
            "before": [Task.RETRIEVE_INIT],
            "main": [Task.INFO_CHECK, Task.SUBQUERY_CONSTRUCT_WITH_HISTORY, Task.RETRIEVE],
            "after" : [Task.INFO_CHECK]
            }
        return self.infer(task_list, data, start_iter=start_iter, finalize_method=self.finalize_data_hist)
    
    def infer_vod_hist(self, data, start_iter=0):
        task_list = {
            "before": [Task.RETRIEVE_INIT],
            "main": [Task.PROVIDE_ANSWER, Task.VERIFY_OR_DENY, Task.SUBQUERY_CONSTRUCT_WITH_HISTORY, Task.RETRIEVE],
            "after" : [Task.PROVIDE_ANSWER]
            }
        return self.infer(task_list, data, start_iter=start_iter, finalize_method=self.finalize_data_vod_hist)
    
    def reconfigure(self, inferrer_config):
        self.config = inferrer_config