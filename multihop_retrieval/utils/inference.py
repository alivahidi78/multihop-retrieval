import torch, re, json, time, os
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig
from multihop_retrieval.utils import utils
from multihop_retrieval.utils.outlines_transformers import CustomizedTransformers
from .utils import Task
from .retrieval import Retriever
from outlines.types import Regex
from collections.abc import Mapping

class Inferrer:
    
    dict_labels = {
        "retrieval_iter": "nothink_retrieve_iteration",
        "info_check_iter": "nothink_gen_iteration",
        "subq_construct_iter": "nothink_query_iteration"
    }
    
    task_labels = {
        "info_check_iter": "info",
        "subq_construct_iter": "subq"
    }
    
    def __init__(self, retriever, model, tokenizer, prompts_and_tools, device="cuda"):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.prompts_and_tools = prompts_and_tools
        self.device = device
    
    @classmethod    
    def create_with_retriever(cls, wiki_path, embedder, index, metadata, model, tokenizer, prompts_and_tools):
        return cls(Retriever(wiki_path, embedder, index, metadata), model, tokenizer, prompts_and_tools) 
      
    #################################### retrieval ####################################  
      
    def retrieval_init(self, data, use_tqdm=True, logs=True):
        query_count= 0
        error_count= 0
        for i in tqdm(range(len(data)), disable=not use_tqdm, desc=f"init_ret"):
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
                data[i][f"context"] = result
        if logs:
            print(f"init total queries: {query_count}\nerrors: {error_count}")
        return data
    
    def retrieval_iter(self, data, iteration, use_tqdm=True, logs=True):
        query_count= 0
        error_count= 0
        for i in tqdm(range(len(data)), disable= not use_tqdm, desc=f"iter_{iteration}_2_ret"):
            try:
                subqueries = data[i][f"{Inferrer.dict_labels["subq_construct_iter"]}_{iteration}"]
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
                data[i][f"{Inferrer.dict_labels["retrieval_iter"]}_{iteration}"] = result
        if logs:
            print(f"{iteration} total queries: {query_count}\nerrors: {error_count}")
        return data
    
    #################################### info_check #################################### 
    
    def one_hop(self, query, context, generation_config, thinking=False):
        prompt = utils.get_prompts(self.prompts_and_tools, Task.SHORT_ANSWER, query=query, context=context)
        llm_res = self._call_llm(generation_config, prompt, enable_thinking=thinking, skip_special_tokens=True)
        #TODO deal with thought
        return llm_res["completion_decoded"]
    
    def info_check_iter(self, data, iteration, generation_config, enforce_grammar=True, add_onehop = False, ignore_ids=False, use_tqdm = True):
        prompt_list = []
        response_list = []
        col_name = f"{Inferrer.dict_labels["info_check_iter"]}_{iteration}"
        for i in tqdm(range(0, len(data), 1), disable = not use_tqdm, desc=f"iter_{iteration}_0_gen"):
            selected = data[i]
            
            if iteration != 0 and f"{Inferrer.dict_labels["retrieval_iter"]}_{iteration - 1}" not in selected.keys():
                # the sample is skipped since there's no previous iteration.
                continue
            
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            tools = utils.get_tools(self.prompts_and_tools, Task.INFO_CHECK)
            prompt = utils.get_prompts(self.prompts_and_tools, Task.INFO_CHECK, query=query, context=context)
            prompt_list.append(prompt)

            grammar = None
            if enforce_grammar:
                grammar = Regex(self.prompts_and_tools[Task.INFO_CHECK.value]["pattern"])
            
            
            llm_res = self._call_llm(generation_config, prompt, tools=tools, grammar=grammar, enable_thinking=False, skip_special_tokens=False)
            predicted_m = llm_res["completion_decoded"]
            # TODO deal with thought
            response_list.append(predicted_m)

            if add_onehop:
                # TODO deal with onehop training
                predicted_o = self.one_hop(query, context, generation_config)
            try:
                data[i][col_name] = predicted_m
                if not ignore_ids:
                    data = self._append_llm_res(data, llm_res, i, iteration, Inferrer.task_labels["info_check_iter"])
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
        return data    
    
    def _check_done(self, response):
        positive_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["positive_tag"]
        negative_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["negative_tag"]
        return positive_tag in response
    
    #################################### subq_construct ####################################
    
    def subq_construct_iter(self, data, iteration, generation_config, enforce_grammar=True, ignore_ids=False, use_tqdm=True):
        prompt_list = []
        response_list = []
        for i in tqdm(range(0, len(data), 1), disable= not use_tqdm, desc=f"iter_{iteration}_1_2q"):
            subq_json = self.prompts_and_tools[Task.SUBQUERY_CONSTRUCT.value]
            selected = data[i]
            query = selected["question"]
            context = selected["context"]
            context = self._get_deduplicated_context(context, iteration, selected)
            
            try:
                prev_response = selected[f"{Inferrer.dict_labels["info_check_iter"]}_{iteration}"]
            except KeyError:
                # the sample is skipped since there's no previous iteration.
                continue

            if(self._check_done(prev_response)):
                continue
            tools = utils.get_tools(self.prompts_and_tools, Task.SUBQUERY_CONSTRUCT)
            prompt = utils.get_prompts(self.prompts_and_tools, Task.SUBQUERY_CONSTRUCT, query=query, context=context)
            prompt_list.append(prompt)
            
            grammar = None
            if enforce_grammar:
                grammar = Regex(subq_json["pattern"])
            
            llm_res = self._call_llm(generation_config, prompt, grammar=grammar, tools=tools)
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
                raise ValueError(f"the query output is malformed: {llm_output}.")
            try:
                data[i][f"{Inferrer.dict_labels["subq_construct_iter"]}_{iteration}"] = subqueries
                if not ignore_ids:
                    data = self._append_llm_res(data, llm_res, i, iteration, Inferrer.task_labels["subq_construct_iter"])
            except Exception as e:
                print(f"exception at {i}")
        return data, prompt_list, response_list
    
    #################################### finalize ####################################
        
    def finalize_data(self, data, iterations = 3):
        positive_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["positive_tag"]
        negative_tag = self.prompts_and_tools[Task.INFO_CHECK.value]["negative_tag"]
        info_pattern = self.prompts_and_tools[Task.INFO_CHECK.value]["pattern"]
        regex_group = self.prompts_and_tools[Task.INFO_CHECK.value]["regex_group"]

        for d in data:
            d[f"multihop{iterations}"] = ""
            d["error"] = ""

            # check from iteration_3 down to iteration_0
            for i in range(iterations, -1, -1):
                key = f"{Inferrer.dict_labels["info_check_iter"]}_{i}"
                query_key= f"{Inferrer.dict_labels["subq_construct_iter"]}_{i}"
                if key in d and isinstance(d[key], str):
                    content = d[key]

                    if query_key in d:
                        d["error"] = "query"
                        break

                    if negative_tag in content:
                        if i == iterations:
                            d["error"] = "info"
                        else:
                            d["error"] = "unknown"
                        break
                    
                    match = re.search(info_pattern, content)
                    if not match:
                        d["error"] = "format"
                    elif match.group(regex_group):
                        d[f"multihop{iterations}"] = match.group(regex_group)
                    else:
                        d["error"] = "format"
                    break  # stop after first valid key
        return data
    
    #################################### internal ####################################
    
    def _get_deduplicated_context(self, context, iteration, selected_datum):
        context = context.copy()
        if not (iteration == 0):
            try:
                for k in range(0, iteration):
                    context.extend(selected_datum[f"{Inferrer.dict_labels["retrieval_iter"]}_{k}"])
            except KeyError:
                # TODO warning
                pass
        unique = {}
        for k, v in context:
            unique[k] = v 
        context = [[k, v] for k, v in unique.items()]
        return context
    
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
        ).to(self.device)  
        
        inputs = self._prepare_inputs(inputs)
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
            raise NotImplementedError("Thinking is currently disabled.")
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
            kwargs = {"device": self.device}
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
    
    #TODO put some of the parameters inside config
    def infer(self, data, generation_config, enforce_grammar=True, iterations=3, start_iter=0, use_tqdm=False, logs=False, add_onehop=False, calculate_time=False, ignore_ids=False):
        timer_start = time.time()
        if start_iter == 0:
            data = self.retrieval_init(data, use_tqdm=use_tqdm, logs=logs)
        iterations_processed = 0
        for i in range(start_iter, iterations):
            data, _, __ = self.info_check_iter(data, i, generation_config, enforce_grammar=enforce_grammar, add_onehop=(add_onehop and i == 0), ignore_ids=ignore_ids, use_tqdm=use_tqdm)
            data, _, __ = self.subq_construct_iter(data, i, generation_config, enforce_grammar=enforce_grammar, ignore_ids=ignore_ids, use_tqdm=use_tqdm)
            data = self.retrieval_iter(data, i, use_tqdm=use_tqdm, logs=logs)
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