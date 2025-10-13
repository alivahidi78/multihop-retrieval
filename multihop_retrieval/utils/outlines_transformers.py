from outlines.models import Transformers

class CustomizedTransformers(Transformers):
    def generate(self, prompts, inputs, output_type = None, **inference_kwargs):
        logits_processor = self.type_adapter.format_output_type(output_type)

        #TODO review the code for this method
        # is_encoder_decoder
        generated_ids = self._generate_output_seq(
            prompts,
            inputs,
            logits_processor=logits_processor,
            **inference_kwargs,
        )
        # required for multi-modal models that return a 2D tensor even when
        # num_return_sequences is 1
        num_samples = inference_kwargs.get("num_return_sequences", 1)
        if num_samples == 1 and len(generated_ids.shape) == 2:
            generated_ids = generated_ids.squeeze(0)

        return generated_ids
    
    def generate_batch(self, prompts, inputs, output_type = None, **inference_kwargs):
        logits_processor = self.type_adapter.format_output_type(output_type)

        generated_ids = self._generate_output_seq(
            prompts, inputs, logits_processor=logits_processor, **inference_kwargs
        )

        # if there are multiple samples per input, convert generated_id to 3D
        num_samples = inference_kwargs.get("num_return_sequences", 1)
        if num_samples > 1:
            generated_ids = generated_ids.view(len(prompts), num_samples, -1)

        return generated_ids