import evals
import evals.metrics
import random
import openai
import logging
import json
from evals.record import record_each_sample_match
# import numpy as np
# import sentence_transformers

openai.api_key = ""

# def convert_words_to_vectors(words, word_embedding_model):
#     """Converts a list of words into vectors using a word embedding model."""
#     vectors = []
#     for word in words:
#         if word in word_embedding_model:
#             vectors.append(word_embedding_model[word])
#         else:
#             vectors.append(np.zeros(word_embedding_model.vector_size))
#     return vectors

# def vec():
#    # Load the Word2Vec model
#     word_embedding_model = sentence_transformers.WordEmbedding.from_pretrained("word2vec_model.bin")

#     # Convert the words "cat" and "dog" into vectors
#     words = ["cat", "dog"]
#     vectors = word_embedding_model(words)

#     # Print the vectors
#     print(vectors)

#     # Calculate the similarity between the words "cat" and "dog"
#     similarity = word_embedding_model.similarity("cat", "dog")

#     # Print the similarity
#     logging.info(similarity)
#     print(similarity)


# def cosine_similarity(v1, v2):
#     """Calculates the cosine similarity between two vectors."""
#     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# def main():
#     # Create two vectors
#     v1 = np.array([1, 2, 3])
#     v2 = np.array([4, 5, 6])

#     # Calculate the cosine similarity between the two vectors
#     cosine_similarity_value = cosine_similarity(v1, v2)

#     # Print the cosine similarity
#     logging.info(f"fasdf*************: {cosine_similarity_value}")
#     print(cosine_similarity)



class PromptTest(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl

    def run(self, recorder):
        # vec()
        # main()
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