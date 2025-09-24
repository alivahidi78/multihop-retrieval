from multihop_retrieval.utils.inference import Inferrer

from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd
from trl.data_utils import is_conversational
from trl.extras.profiling import profiling_context
from accelerate.utils import gather_object
import copy
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext        
          
class MultihopGRPOTrainer(GRPOTrainer):
    
    def __init__(self, model, reward_funcs, retriever, prompts_path, tools_path, args = None, iterations = 3, train_dataset = None, eval_dataset = None, processing_class = None, reward_processing_classes = None, callbacks = None, optimizers = (None, None), peft_config = None):
        self.iterations = iterations
        self.retriever = retriever
        self.prompts_path = prompts_path
        self.tools_path = tools_path
        super().__init__(model, reward_funcs, args, train_dataset, eval_dataset, processing_class, reward_processing_classes, callbacks, optimizers, peft_config)
    
    #Overridden
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        data = copy.deepcopy(inputs)
        
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):  
            input_preparation_func = super(GRPOTrainer, self)._prepare_inputs
            
            inferrer = Inferrer(self.retriever, unwrapped_model, self.processing_class, self.prompts_path, self.tools_path)
            
            data = inferrer.infer(data, self.generation_config, self.iterations, input_preparation_func=input_preparation_func)
        
        prompt_ids = [lst for d in data for lst in d["prompt_ids"].values()]
        prompt_mask = [lst for d in data for lst in d["prompt_mask"].values()]
        completion_ids = [lst for d in data for lst in d["thought_and_completion_ids"].values()]
        original_prompts = [lst for d in data for lst in d["prompts"].values()]
        prompt_ids = torch.stack(prompt_ids)
        prompt_mask = torch.stack(prompt_mask)
        completion_ids = torch.stack(completion_ids)
        
        prompts = copy.deepcopy(original_prompts)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        #TODO ??
        old_per_token_logps = None
        ref_per_token_logps = None
        
        # with torch.no_grad():
        #     # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
        #     # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
        #     # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
        #     # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
        #     # old_per_token_logps to None.
        #     generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
        #     if self.args.gradient_accumulation_steps % generate_every != 0:
        #         old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
        #             self.model,
        #             prompt_completion_ids,
        #             attention_mask,
        #             logits_to_keep,
        #             batch_size,
        #             pixel_values=prompt_inputs.get("pixel_values"),
        #             image_grid_thw=prompt_inputs.get("image_grid_thw"),
        #             pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
        #             image_sizes=prompt_inputs.get("image_sizes"),
        #         )
        #     else:
        #         old_per_token_logps = None

        #     # Compute the per-token log probabilities for the reference model
        #     if self.beta != 0.0:
        #         if self.ref_model is not None:
        #             ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
        #                 self.ref_model,
        #                 prompt_completion_ids,
        #                 attention_mask,
        #                 logits_to_keep,
        #                 batch_size=batch_size,
        #                 pixel_values=prompt_inputs.get("pixel_values"),
        #                 image_grid_thw=prompt_inputs.get("image_grid_thw"),
        #                 pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
        #                 image_sizes=prompt_inputs.get("image_sizes"),
        #             )
        #         else:
        #             with self.accelerator.unwrap_model(self.model).disable_adapter():
        #                 ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
        #                     self.model,
        #                     prompt_completion_ids,
        #                     attention_mask,
        #                     logits_to_keep,
        #                     batch_size=batch_size,
        #                     pixel_values=prompt_inputs.get("pixel_values"),
        #                     image_grid_thw=prompt_inputs.get("image_grid_thw"),
        #                     pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
        #                     image_sizes=prompt_inputs.get("image_sizes"),
        #                 )
        #     else:
        #         ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        prompts_text = prompts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
    
    #Overridden
    def _compute_loss(self, model, inputs):
        return super()._compute_loss(model, inputs)
    
    #Overridden
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, batch_size)
    
    #Overridden
    def _prepare_inputs(self, generation_batch):
        return super()._prepare_inputs(generation_batch)