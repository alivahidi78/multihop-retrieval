from multihop_retrieval.utils.inference_utils import Inferrer, InferrerConfig
from multihop_retrieval.utils import generic_utils as utils
from multihop_retrieval.utils.generic_utils import Task
from transformers.training_args import OptimizerNames

from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd
from trl.data_utils import is_conversational
from trl.extras.profiling import profiling_context
from trl.trainer.utils import pad, shuffle_sequence_dict
import traceback

from accelerate.utils import gather_object
from itertools import accumulate
import copy, json, os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext           

class MultihopGRPOTrainer(GRPOTrainer):
    
    def __init__(self, model, retriever, prompts_and_tools, reward_funcs=None, args = None, iterations = 2, enforce_grammar=True, train_dataset = None, eval_dataset = None, processing_class = None, reward_processing_classes = None, callbacks = None, optimizers = (None, None), peft_config = None, unbundled_batching = None, no_cache=False, inference_mode="basic"):
        self.current_gradient_accumulation_steps = args.gradient_accumulation_steps
        self.iterations = iterations
        self.retriever = retriever
        self.unbundled_batching = unbundled_batching
        self.prompts_and_tools = prompts_and_tools
        self.enforce_grammar = enforce_grammar
        self.no_cache = no_cache
        self.inference_mode = inference_mode
        
        if reward_funcs == None:
            reward_funcs = MultihopGRPOTrainer.get_default_reward_functions(self.prompts_and_tools)
            
        super().__init__(model, reward_funcs=reward_funcs, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=processing_class, reward_processing_classes=reward_processing_classes, callbacks=callbacks, optimizers=optimizers, peft_config=peft_config)
    
    @staticmethod
    def get_default_reward_functions(prompts_and_tools):
        
        def compute_exact(data, final_answers, bundle_lengths, **kwargs):
            bundled_rewards = [0]* len(final_answers)
            golden_answers = [example["answer"] for example in data]
            for i, a in enumerate(final_answers):
                bundled_rewards[i] = utils.compute_exact(golden_answers[i], a)
            rewards = [row for row, n in zip(bundled_rewards, bundle_lengths) for _ in range(n)]
            return rewards
            
        def compute_f1(data, final_answers, bundle_lengths, **kwargs):
            bundled_rewards = [0]* len(final_answers)
            golden_answers = [example["answer"] for example in data]
            for i, a in enumerate(final_answers):
                bundled_rewards[i] = utils.compute_f1(golden_answers[i], a)
            rewards = [row for row, n in zip(bundled_rewards, bundle_lengths) for _ in range(n)]
            return rewards
        return [compute_exact, compute_f1]
        
    #Overridden
    def _generate_and_score_completions(self, inputs):
        if self.args.num_generations != self.args.per_device_train_batch_size:
            raise NotImplementedError("num_generations should be equal to per_device_train_batch_size.")
        
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        data = copy.deepcopy(inputs)
        
        if self.use_vllm:
            raise NotImplementedError("use_vllm is not currently supported.")
                
        if self.use_transformers_paged:
            
            raise NotImplementedError("use_transformers_paged is not currently supported.")
        else:
            # Context transplated from GRPOTrainer
            with(
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):  
                inferrer_config = InferrerConfig(generation_config = self.generation_config, iterations = self.iterations, enforce_grammar = self.enforce_grammar)
                inferrer = Inferrer(self.retriever, unwrapped_model, self.processing_class, self.prompts_and_tools, inferrer_config = inferrer_config, no_cache=self.no_cache)
                if self.inference_mode == "basic":
                    data = inferrer.infer_basic(data)
                elif self.inference_mode == "hist":
                    data = inferrer.infer_hist(data)
                elif self.inference_mode == "vod":
                    data = inferrer.infer_vod(data)
                elif self.inference_mode == "vod_hist":
                    data = inferrer.infer_vod_hist(data)
                else:
                    raise ValueError(f"inference mode {self.inference_mode} is unknown.")
        
        final_answers = [d[f"multihop{self.iterations}"] for d in data]
        errors = [d[f"error"] for d in data]
        
        prompts_bundled = [list(d["prompt"].values()) for d in data]
        prompt_ids_bundled = [list(d["prompt_ids"].values()) for d in data]  
        prompt_masks_bundled = [list(d["prompt_mask"].values()) for d in data]
        completion_ids_bundled = [list(d["thought_and_completion_ids"].values()) for d in data]
        completions_bundled = [list(d["completion_decoded"].values()) for d in data]
        
        prompt_ids, prompt_ids_counts = utils.unbundle(prompt_ids_bundled)
        prompt_mask, prompt_mask_counts = utils.unbundle(prompt_masks_bundled)
        completion_ids, completion_ids_counts = utils.unbundle(completion_ids_bundled)
        original_prompts, original_prompts_counts = utils.unbundle(prompts_bundled)
        completions, completions_counts = utils.unbundle(completions_bundled)
        
        assert prompt_ids_counts == prompt_mask_counts == completion_ids_counts == original_prompts_counts == completions_counts
        
        bundle_lengths = prompt_ids_counts
        
        # Now we have all the prompts and completions and the way they should be grouped together
        # for the purpose of them being part of the same multi-step trajectory
        
        prompts = copy.deepcopy(original_prompts)
        
        # Prompts padded on the left.
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        # Save original completions lengths for padding the masks later
        original_completions_lengths = torch.tensor([len(t) for t in completion_ids], device=device)
        # Completions padded on the right
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        
        # Mask everything after the first EOS token  
        # Transplanted from GRPOTrainer  
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # If no eos, still exclude padding
        default_completions_mask = (sequence_indices < original_completions_lengths.unsqueeze(1)).int()
        completion_mask = default_completions_mask * completion_mask

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward
        # function, avoiding the need to re-tokenize completions if the reward is computed 
        # from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask (transplanted from parent class)
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(1)

        old_per_token_logps = None
        ref_per_token_logps = None
        
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                # This means generation is misaligned
                mode = "train" if self.model.training else "eval"
                if self.unbundled_batching:
                    batch_size = self.unbundled_batching if mode == "train" else None
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
                
            if self.use_vllm and self.vllm_importance_sampling_correction:
                # This means generation is misaligned
                raise NotImplementedError("This option is not currently supported.")
            
            if self.beta != 0.0:
                # Reference model should be used
                raise NotImplementedError("This option is not currently supported.")

        # Process the generated completions
        if is_conversational(inputs[0]):
            processed_completions = []
            for prompt, completion in zip(prompts, completions):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                processed_completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            processed_completions = completions

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across 
        # all processes. This is important because rewards will be normalized per group, and
        # completions are distributed. We will later slice rewards_per_func to extract each
        # process's subset.
        
        completions_rebundled = utils.rebundle(processed_completions, bundle_lengths)
        completion_ids_list_rebundled = utils.rebundle(completion_ids_list, bundle_lengths)
        rewards_unbundled_func = self._calculate_rewards(data, final_answers, bundle_lengths)
        # Apply weights to each reward function's output and sum
        rewards_unbundled = (rewards_unbundled_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # rewards_bundled = (rewards_bundled_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        ### This entire section is incorrect if more than 1 step is accumulated
        ### Also it is incorrect since we use averages of all llm-calls
        ##########################################################################################
        # Compute grouped-wise rewards
        # mean_grouped_rewards = rewards_unbundled.view(-1, sum(bundle_lengths)).mean(dim=1)
        # std_grouped_rewards = rewards_unbundled.view(-1, sum(bundle_lengths)).std(dim=1)
        # is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
        
        # # Normalize the rewards to compute the advantages
        # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(sum(bundle_lengths), dim=0)
        # std_grouped_rewards = std_grouped_rewards.repeat_interleave(sum(bundle_lengths), dim=0)
        # advantages = rewards_unbundled - mean_grouped_rewards
        # if self.scale_rewards:
        #     advantages = advantages / (std_grouped_rewards + 1e-4)
        ###########################################################################################
        indices = [0] + list(accumulate(bundle_lengths))
        bundled_rewards_mean = torch.tensor([rewards_unbundled[indices[i]:indices[i+1]].mean() for i in range(len(bundle_lengths))], device="cuda")
        mean_grouped_rewards = bundled_rewards_mean.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = bundled_rewards_mean.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        mean_bundle_grouped_rewards = mean_grouped_rewards.repeat_interleave(torch.tensor(bundle_lengths, device="cuda"), dim=0)
        std_bundle_grouped_rewards = std_grouped_rewards.repeat_interleave(torch.tensor(bundle_lengths, device="cuda"), dim=0)
        
        # mean_bundle_rewards_repeated = mean_bundle_rewards.repeat_interleave(torch.tensor(bundle_lengths, device="cuda"), dim=0)
        # advantages = mean_bundle_rewards_repeated - mean_bundle_grouped_rewards
        advantages = rewards_unbundled - mean_bundle_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_bundle_grouped_rewards + 1e-6)
        
        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Calculate mean reward per function, but only for samples where the function 
        # was applied (non-NaN values)
        # Note: We average unbundled rewards for this
        for i, reward_func_name in enumerate(self.reward_func_names):
            
            mean_rewards = torch.nanmean(rewards_unbundled_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            
            indices = [0] + list(accumulate(bundle_lengths))
            bundled_mean_rewards = torch.tensor([rewards_unbundled_func[indices[j]:indices[j+1], i].mean() for j in range(len(bundle_lengths))], device=rewards_unbundled_func.device)
            logged_mean = torch.nanmean(bundled_mean_rewards).item()
            self._metrics[mode][f"rewards/{reward_func_name}/bundled_mean"].append(logged_mean)
            
            std_rewards = nanstd(rewards_unbundled_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
            
            logged_std = nanstd(bundled_mean_rewards).item()
            self._metrics[mode][f"rewards/{reward_func_name}/bundled_std"].append(logged_std)
            
        self._metrics[mode]["generation_count"].append(sum(bundle_lengths) / len(bundle_lengths))
        self._metrics[mode]["reward"].append(mean_bundle_grouped_rewards.mean().item())
        self._metrics[mode]["reward_bundled"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_bundle_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std_bundled"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        # FIXME replace this with user query instead? 
        # Note: probably irrelevant for training
        self._logs["prompt"].extend(gather_object(prompts_bundled))
        self._logs["completion"].extend(gather_object(completions_bundled))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_unbundled_func[:, i].tolist())
        self._logs["advantages"].extend(advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
    
    #Overridden
    def _calculate_rewards(self, data, final_answers, bundle_lengths):
        from torch import nn
        import warnings
        from accelerate.utils import gather
        device = self.accelerator.device
        rewards_per_func = torch.zeros(sum(bundle_lengths), len(self.reward_funcs), device=device)

        reward_kwargs = {}
        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  
                    raise NotImplementedError("models as reward functions are not currently supported.")
                else:
                    output_reward_func = reward_func(
                        data, final_answers=final_answers, bundle_lengths=bundle_lengths, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        # if torch.isnan(rewards_per_func).all(dim=1).any():
        #     nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
        #     row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
        #     row_reward_kwargs["prompt"] = prompts[nan_row_idx]
        #     row_reward_kwargs["completion"] = completions[nan_row_idx]
        #     warnings.warn(
        #         f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
        #         "Please ensure that at least one reward function returns a valid reward."
        #     )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_unbundled = gather(rewards_per_func)
        # rewards_unbundled = torch.cat([row.unsqueeze(0).repeat(n, 1) for row, n in zip(rewards_bundled, bundle_lengths)], dim=0)
        return rewards_unbundled
    
    #Overridden
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, pixel_values=None, image_grid_thw=None, pixel_attention_mask=None, image_sizes=None):
        # Note: this is meant solely for compute_loss calls within the training step.
        # A cleaner implementation might be possible
        mode = "train" if self.model.training else "eval"
        if self.unbundled_batching:
            batch_size = self.unbundled_batching if mode == "train" else None
        return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy, pixel_values, image_grid_thw, pixel_attention_mask, image_sizes)

    def training_step(self, model, inputs, num_items_in_batch = None):
        if self.no_cache:
            torch.cuda.empty_cache()  
        
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            try:
                inputs = self._prepare_inputs(inputs)
            except Exception as e:
                loss = torch.tensor(0.0, device="cuda")
                kwargs = {}
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()
                if self.args.n_gpu > 1:
                    loss = loss.mean() 
                self.accelerator.backward(loss, **kwargs)
                traceback.print_exc()
                return loss
                
            batch_size = self.unbundled_batching or 512
            total_items = 0
            total_loss = 0.0
            for i in range(0, len(inputs["prompt_ids"]), batch_size):
                sub_inputs = {k: v[i:i+batch_size] for k, v in inputs.items() if k != "num_items_in_batch"}
                sub_items = len(sub_inputs["prompt_ids"])
                
                #########################################################################
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, sub_inputs, num_items_in_batch=sub_items)
                #########################################################################
                
                if (
                    self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0
                ):
                    torch.cuda.empty_cache()
                kwargs = {}
                # For LOMO optimizers you need to explicitly use the learning rate
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                ##################################################################
                self.accelerator.backward(loss, **kwargs)
                ##################################################################
                
                sub_loss = loss.detach()
                total_loss += sub_loss * sub_items
                total_items += sub_items
            del inputs
            loss = total_loss / total_items
                            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (
                not self.model_accepts_loss_kwargs or num_items_in_batch is None
            ) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps
            
            return loss
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
    
    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        inputs = self._generate_and_score_completions(generation_batch)
        if mode == "train":
            inputs = shuffle_sequence_dict(inputs)
            self._step += 1
        return inputs