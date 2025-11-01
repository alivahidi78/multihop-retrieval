import torch, re, json, time, os
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig
from multihop_retrieval.utils import utils
from multihop_retrieval.utils.outlines_transformers import OutlinesWrapper
from .utils import Task, method_task_id
from .retrieval import Retriever
from outlines.types import Regex
from collections.abc import Mapping

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
            "iterations": 3,
            "use_tqdm": False,
            "logs": False,
            "add_onehop": False,
            "calculate_time": False,
            "remove_intermediate_steps": False,
            "device": "cuda"
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
    
    def __init__(self, retriever, model, tokenizer, prompts_and_tools, inferrer_config=None):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.prompts_and_tools = prompts_and_tools
        self.config = inferrer_config if inferrer_config else InferrerConfig()
    
    @classmethod    
    def create_with_retriever(cls, wiki_path, embedder, index, metadata, model, tokenizer, prompts_and_tools, inferrer_config=None):
        return cls(Retriever(wiki_path, embedder, index, metadata), model, tokenizer, prompts_and_tools, inferrer_config) 
      
    #################################### retrieval ####################################  
    
    @method_task_id(Task.RETRIEVE_INIT)
    def retrieval_init(self, data, task_id=None):
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
        for i in tqdm(range(len(data)), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_2_ret"):
            try:
                subqueries = data[i][f"{Inferrer.dict_labels[prev_task_id]}_{iteration}"]
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
                data[i][f"{Inferrer.dict_labels[task_id]}_{iteration}"] = result
        if self.config.logs:
            print(f"{iteration} total queries: {query_count}\nerrors: {error_count}")
        return data
    
    #################################### info_check #################################### 
    
    @method_task_id(Task.INFO_CHECK)    
    def info_check_iter(self, data, iteration, prev_task_id, task_id=None):
        prompt_list = []
        response_list = []
        col_name = f"{Inferrer.dict_labels[task_id]}_{iteration}"
        for i in tqdm(range(0, len(data), 1), disable = not self.config.use_tqdm, desc=f"iter_{iteration}_0_gen"):
            selected = data[i]
            
            if iteration != 0 and f"{Inferrer.dict_labels[prev_task_id]}_{iteration - 1}" not in selected.keys():
                # the sample is skipped since there's no previous iteration.
                continue
            
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context)
            prompt_list.append(prompt)

            grammar = None
            if self.config.enforce_grammar:
                grammar = Regex(self.prompts_and_tools[task_id.value]["pattern"])
            
            llm_res = self._call_llm(prompt, tools=tools, grammar=grammar, enable_thinking=False, skip_special_tokens=False)
            predicted_m = llm_res["completion_decoded"]
            # TODO deal with thought
            response_list.append(predicted_m)

            if self.config.add_onehop and iteration==0:
                # TODO deal with onehop training
                predicted_o = self._one_hop(query, context)
            try:
                data[i][col_name] = predicted_m
                data = self._append_llm_res(data, llm_res, i, iteration, Inferrer.dict_labels[task_id])
                if self.config.add_onehop and iteration==0:
                    data[i]["onehop"] = predicted_o
            except Exception as e:
                print(f"exception at {i}")
        return data
    
    ### short_answer ###
    def _one_hop(self, query, context, thinking=False):
        prompt = utils.get_prompts(self.prompts_and_tools, Task.SHORT_ANSWER, query=query, context=context)
        llm_res = self._call_llm(prompt, enable_thinking=thinking, skip_special_tokens=True)
        #TODO deal with thought
        return llm_res["completion_decoded"]
    
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
        return data    
    
    def _check_done(self, response, task_id):
        positive_tag = self.prompts_and_tools[task_id.value]["positive_tag"]
        negative_tag = self.prompts_and_tools[task_id.value]["negative_tag"]
        tag_group = self.prompts_and_tools[task_id.value]["tag_group"]
        pattern = self.prompts_and_tools[task_id.value]["pattern"]
        match = re.search(pattern, response) 
        if match:
            group_text = match.group(tag_group)
            if group_text is None:
                pass
            elif positive_tag in match.group(tag_group):
                return True
            elif negative_tag in match.group(tag_group):
                return False
        print(f"Response malformed: {response}")
        return False
    
    #################################### provide_answer ####################################
    @method_task_id(Task.PROVIDE_ANSWER)
    def provide_answer_iter(self, data, iteration, prev_task_id, task_id=None):
        pass
    
    #################################### verify_or_deny ####################################
    @method_task_id(Task.VERIFY_OR_DENY)
    def verify_or_deny_iter(self, data, iteration, prev_task_id, task_id=None):
        pass
    
    #################################### subq_construct ####################################
    
    @method_task_id(Task.SUBQUERY_CONSTRUCT)
    def subq_construct_iter(self, data, iteration, prev_task_id, task_id=None):
        prompt_list = []
        response_list = []
        for i in tqdm(range(0, len(data), 1), disable= not self.config.use_tqdm, desc=f"iter_{iteration}_1_2q"):
            subq_json = self.prompts_and_tools[task_id.value]
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            
            try:
                prev_response = selected[f"{Inferrer.dict_labels[prev_task_id]}_{iteration}"]
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue

            if(self._check_done(prev_response, prev_task_id)):
                continue
            tools = utils.get_tools(self.prompts_and_tools, task_id)
            prompt = utils.get_prompts(self.prompts_and_tools, task_id, query=query, context=context)
            prompt_list.append(prompt)
            
            grammar = None
            if self.config.enforce_grammar:
                grammar = Regex(subq_json["pattern"])
            
            llm_res = self._call_llm(prompt, grammar=grammar, tools=tools)
            llm_output = llm_res["completion_decoded"]
            pattern = subq_json["pattern"]
            regex_groups = subq_json["regex_groups"]
            match = re.search(pattern, llm_output)
            subqueries = []
            if match:
                for g in regex_groups:
                    if match.group(g):
                        subqueries.append(match.group(g))
            else:
                #TODO warning
                print(f"The following query attempt is malformed:\n{llm_output}.")
                continue
            try:
                data[i][f"{Inferrer.dict_labels[task_id]}_{iteration}"] = subqueries
                data = self._append_llm_res(data, llm_res, i, iteration, Inferrer.dict_labels[task_id])
            except Exception as e:
                print(f"exception at {i}")
        return data
    
    #################################### subq_construct_history ####################################
    
    @method_task_id(Task.SUBQUERY_CONSTRUCT_WITH_HISTORY)
    def subq_construct_history_iter(self, data, iteration, prev_task_id, task_id=None):
        pass
    
    #################################### finalize ####################################
        
    def finalize_data_basic(self, data, iterations = 3):
        positive_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["positive_tag"]
        negative_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["negative_tag"]
        info_pattern = self.prompts_and_tools[Task.INFO_CHECK.value]["pattern"]
        answer_group = self.prompts_and_tools[Task.INFO_CHECK.value]["answer_group"]
        
        ic_label = Inferrer.dict_labels[Task.INFO_CHECK]
        sc_label = Inferrer.dict_labels[Task.SUBQUERY_CONSTRUCT]
        rv_label = Inferrer.dict_labels[Task.RETRIEVE]

        for d in data:
            d["ic_calls"] = sum(1 for key in d if key.startswith(ic_label))
            d["sc_calls"] = sum(1 for key in d if key.startswith(sc_label))
            d["rv_calls"] = sum(1 for key in d if key.startswith(rv_label)) + 1
            d["llm_calls"] = d["ic_calls"] + d["sc_calls"]
            d[f"multihop{iterations}"] = ""
            d["error"] = ""

            if f"{ic_label}_{0}" not in d:
                d["error"] = "llm_start"
                continue
            # check from iteration_max down to iteration_0
            for i in range(iterations, -1, -1):
                key = f"{ic_label}_{i}"
                query_key= f"{sc_label}_{i}"
                ret_key= f"{rv_label}_{i}"
                if key in d and isinstance(d[key], str):
                    d["last_iter"] = i
                    content = d[key]
                    
                    if ret_key in d:
                        d["error"] = "generation"
                        
                    if query_key in d:
                        d["error"] = "retrieval"
                        break

                    if negative_tag in content:
                        if i == iterations:
                            d["error"] = "info"
                        else:
                            d["error"] = "subquery"
                        break
                    
                    match = re.search(info_pattern, content)
                    if not match:
                        d["error"] = "format"
                    elif match.group(answer_group):
                        d[f"multihop{iterations}"] = match.group(answer_group)
                    else:
                        d["error"] = "format"
                    break  # stop after first valid key
        return data
    
    #################################### internal ####################################
    
    def _call_llm(self, prompt, grammar = None, tools = None, skip_special_tokens=False, enable_thinking=False):
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
        ).to(self.config.device)  
        
        inputs = self._prepare_inputs(inputs)
        prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
        inputs = {}
        inputs["input_ids"], inputs["attention_mask"] = prompt_ids, prompt_mask
        
        if grammar:
            outlines_wrapped = OutlinesWrapper(self.model, self.tokenizer)
            output_type = grammar
            outputs = outlines_wrapped.generate(text, inputs, output_type=output_type, generation_config=self.config.generation_config)
            
        else:
            outputs = self.model.generate(
                **inputs,
                generation_config = self.config.generation_config
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
        }
    
    def _get_deduplicated_context(self, context, iteration, selected_datum):
        context = context.copy()
        if not (iteration == 0):
            try:
                for k in range(0, iteration):
                    context.extend(selected_datum[f"{Inferrer.dict_labels[Task.RETRIEVE]}_{k}"])
            except KeyError:
                # TODO warning
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
    
    #################################### main ####################################
    
    def infer_basic(self, data, start_iter=0):
        timer_start = time.time()
        if start_iter == 0:
            data = self.retrieval_init(data)
        iterations_processed = 0
        for i in range(start_iter, self.config.iterations):
            data = self.info_check_iter(data, i, prev_task_id=Task.RETRIEVE)
            data = self.subq_construct_iter(data, i, prev_task_id=Task.INFO_CHECK)
            data = self.retrieval_iter(data, i, prev_task_id=Task.SUBQUERY_CONSTRUCT)
            iterations_processed += 1
        data = self.info_check_iter(data, self.config.iterations, prev_task_id=Task.RETRIEVE)
        data = self.finalize_data_basic(data, self.config.iterations)
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
    
    def reconfigure(self, inferrer_config):
        self.config = inferrer_config