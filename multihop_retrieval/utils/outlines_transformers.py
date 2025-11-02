from outlines.models import Transformers
from outlines.types.dsl import python_types_to_terms, to_regex
from outlines.backends import get_cfg_logits_processor, get_json_schema_logits_processor, get_regex_logits_processor
from outlines.types import CFG, JsonSchema

class OutlinesWrapper(Transformers):
    def generate(self, prompts, inputs, output_type = None, **inference_kwargs):
        backend_name = None
        model = self
        term = python_types_to_terms(output_type)
        if isinstance(term, CFG):
            cfg_string = term.definition
            logits_processor = get_cfg_logits_processor(
                backend_name,
                model,
                cfg_string,
            )
        elif isinstance(term, JsonSchema):
            logits_processor = get_json_schema_logits_processor(
                backend_name,
                model,
                term.schema,
            )
        else:
            regex_string = to_regex(term)
            logits_processor = get_regex_logits_processor(
                backend_name,
                model,
                regex_string,
            )
        logits_processor = self.type_adapter.format_output_type(logits_processor)

        generated_ids = self._generate_output_seq(
            prompts,
            inputs,
            logits_processor=logits_processor,
            **inference_kwargs,
        )

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
    
    def _generate_output_seq(self, prompts, inputs, **inference_kwargs):
        input_ids = inputs["input_ids"]
        output_ids = self.model.generate(
            **inputs,
            **inference_kwargs,
        )
        return output_ids