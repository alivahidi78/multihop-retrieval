from trl import GRPOTrainer
class MultihopGRPOTrainer(GRPOTrainer):
    
    def __init__(self, model, reward_funcs, args = None, iterations = 3, train_dataset = None, eval_dataset = None, processing_class = None, reward_processing_classes = None, callbacks = None, optimizers = ..., peft_config = None):
        self.iterations = iterations
        super().__init__(model, reward_funcs, args, train_dataset, eval_dataset, processing_class, reward_processing_classes, callbacks, optimizers, peft_config)
    
    def _generate_and_score_completions(self, inputs):
        return super()._generate_and_score_completions(inputs)
    
    def _compute_loss(self, model, inputs):
        return super()._compute_loss(model, inputs)
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, batch_size)
    
    def _prepare_inputs(self, generation_batch):
        return super()._prepare_inputs(generation_batch)