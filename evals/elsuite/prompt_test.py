import evals
import evals.metrics
import random
import json
import time
import openai
from evals.record import record_each_sample_match

openai.api_key = ""


class PromptTest(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        test_samples_length = len(test_samples)
        self.eval_all_samples(recorder, test_samples)


    def eval_sample(self, test_sample, rng: random.Random):
        # system_input = {"role": "system", "content": "You are an educator who has been given the task of classifying and identifying tags for the questions that are asked in Indian competitive examinations. You want to add the subject tag and drill down on the hierarchical topics and sub-topics of the questions. You need to provide three levels of tags: Level 1, Level 2, and Level 3. The answer should be in JSON format, like {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. Select the subject tag only from the below-mentioned 'Subject-Tags'. Subject tags are mentioned inside four quotes. Subject-Tags: '''' History, Information and Communication Technology, Judiciary and Law, Legal Reasoning, Mathematics, Numerical Aptitude, Physics, Polity, Reasoning, Static General Knowledge, Geography, Strategic Management, Teaching Methodology, Agriculture, Aviation, Accountancy, Indian Art and Culture, Statistics, Biology, Business Management, Child development and pedagogy, Disaster Management, Economy, Financial Management, General Awareness, Chemistry, English, Environment ''''. Identify subject tags based on these points: 1. Mathematics: Mathematics encompasses a wide range of mathematical concepts, theories, and principles, including both practical and abstract mathematics. It can become highly complex, involving advanced topics that require a deep understanding of mathematical theory and principles. 2. Numerical Aptitude: Numerical aptitude tests typically include tasks such as arithmetic calculations, percentages, ratios, data interpretation, and solving numerical word problems. Its tasks are usually of moderate complexity and involve basic to intermediate mathematical operations. 3. Reasoning: Reasoning tests encompass a wide range of tasks, including logical puzzles, pattern recognition,  deciphering a code language, deductive reasoning, analogies, and abstract reasoning. Its tasks can vary in complexity, from relatively simple puzzles to more complex abstract reasoning challenges. They do not necessarily involve mathematical calculations. 4. English subject: Questions typically involve reading passages, correcting sentences, identifying synonyms and antonyms, and assessing your comprehension and writing skills. 5. General Awareness: Questions may cover a wide range of topics, including current affairs, history, geography, science, politics, and culture. Some example topics wise questions of reasoning: Coding-Decoding: In a certain code language, 'mee muk pic' is 'roses are yellow', 'nil dic' is 'white flowers', and 'pic muk dic' is 'flowers are fruits'. What is the code for 'white' in that code language? Arrangement and Deductive: Hitesh, Sunny, Vicky, Nitin, and Bharat are arranged in ascending order of height from the top. Hitesh is in third place. Bharat is between Nitin and Hitesh while Nitin is not at the bottom. Who has the maximum height among them? Direction: A and B start from the same point. A cycle 10 km South, then turns to her right and cycles 2 km. B cycles 2 km North. NOTE: ''Provide subject tags only from the above mentioned 'Subject-Tags' inside four quotes. The answer should be in JSON format, like {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. Do not provide the answer to the question.''"}
        # user_input = test_sample["input"]
        # prompt = [system_input, user_input]
        # result = self.completion_fn(
        #     prompt=prompt,
        #     max_tokens=500
        # )
        # response = result.get_completions()[0]
        # expected = test_sample["ideal"]

        response_sample = "{'level 1': 'physics', 'level 2': 'optics', 'level 3': 'mirrors'}"

        response_sample = response_sample.replace("'", '"')

        tag_tree = evals.metrics.get_tag_tree(response_sample)

        if tag_tree == []:
            print("Tag-Tree", tag_tree)
            return tag_tree

        with open("tag_tree.json", "a") as json_file:
            json.dump(tag_tree, json_file)

        system_input_2 = {"role": "system", "content": f"Tag-Tree-List: {tag_tree}. ""Given a 'Tag-Tree-List' consisting of nested dictionaries representing hierarchical tags, your task as an educator is to identify and classify the correct tag tree from this list. You should then drill down into this selected tag tree to extract hierarchical topics and sub-topics of questions. The tag trees in the 'Tag-Tree-List' can have multiple levels, and you need to provide these levels in your output. The levels should be labeled as 'Level 1', 'Level 2', and so on, with the last level being the deepest level found in the selected tag tree. Your response should be in JSON format, structured as follows: {\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"} For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. In case none of the tag trees in the 'Tag-Tree-List' is correct, your answer should be 'None.' You should only provide levels that exist within the selected tag tree; do not add any extra levels. Your response should strictly adhere to the structure of the provided 'Tag-Tree-List.'"}

        user_input_2 = test_sample["input"]
        prompt = [system_input_2, user_input_2]
        result_2 = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response_2 = result_2.get_completions()[0]

        print("response_2", response_2)

        # with open('Prompt_result.txt', 'a') as file:
        #     file.write("#######################################################\n")
        #     file.write(f"{json.dumps(expected)}\n")
        #     file.write(f"{response}\n")
        #     file.write(f"{response_2}\n")
        #     file.write("#######################################################\n\n\n")
        