import evals
import evals.metrics
import random
import openai
import logging
import json
from evals.record import record_each_sample_match

openai.api_key = ""

class PromptTest(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)
        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, test_sample, rng: random.Random):
        prompt = test_sample["input"]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=100
        )
        sampled = result.get_completions()[0]

        logging.info(f"Output: {sampled}")
        logging.info(f"Expected Output: {test_sample['ideal']}")

        # Convert the JSON string to a Python dictionary
        sampled_dict = json.loads(sampled.replace("'", "\""))
        expected_dict = test_sample["ideal"]

        sampled_lowercase_dict = {key.lower(): value.lower() if isinstance(value, str) else [item.lower() for item in value] for key, value in sampled_dict.items()}
        expected_lowercase_dict = {key.lower(): value.lower() if isinstance(value, str) else [item.lower() for item in value] for key, value in expected_dict.items()}

        sample_accuracy = evals.metrics.get_each_sample_accuracy(sampled_lowercase_dict, expected_lowercase_dict)

        logging.info(f"sample_accuracy: {sample_accuracy}")
        
        record_each_sample_match(sample_accuracy)