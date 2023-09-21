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
        system_input = {"role": "system", "content": "As an educator responsible for classifying and identifying tags for questions in Indian competitive examinations, your goal is to add a subject tag and delve into the hierarchical topics and sub-topics of the questions. You should provide three levels of tags: Level 1, Level 2, and Level 3, in JSON format, structured as follows: {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. You should select the subject tag only from the list of 'Subject-Tags' provided: ['History', 'Information and Communication Technology', 'Judiciary and Law', 'Legal Reasoning', 'Mathematics', 'Numerical Aptitude', 'Physics', 'Polity', 'Reasoning', 'Static General Knowledge', 'Geography', 'Strategic Management', 'Teaching Methodology', 'Agriculture', 'Aviation', 'Accountancy', 'Indian Art and Culture', 'Statistics', 'Biology', 'Business Management', 'Child development and pedagogy', 'Disaster Management', 'Economy', 'Financial Management', 'General Awareness', 'Chemistry', 'English', 'Environment']. To identify subject tags, consider the following criteria: 1. Mathematics: This subject encompasses a wide range of mathematical concepts, theories, and principles, including both practical and abstract mathematics. It can become highly complex, involving advanced topics that require a deep understanding of mathematical theory and principles. 2. Numerical Aptitude: Numerical aptitude tests typically include tasks such as arithmetic calculations, percentages, ratios, data interpretation, and solving numerical word problems. Its tasks are usually of moderate complexity and involve basic to intermediate mathematical operations. 3. Reasoning: Reasoning tests encompass a wide range of tasks, including logical puzzles, pattern recognition, deciphering a code language, deductive reasoning, analogies, and abstract reasoning. Its tasks can vary in complexity, from relatively simple puzzles to more complex abstract reasoning challenges. They do not necessarily involve mathematical calculations. 4. English subject: Questions typically involve reading passages, correcting sentences, identifying synonyms and antonyms, and assessing your comprehension and writing skills. 5. General Awareness: Questions may cover a wide range of topics, including current affairs, history, geography, science, politics, and culture. Some example topics wise questions of reasoning subject: Coding-Decoding: In a certain code language, 'mee muk pic' is 'roses are yellow', 'nil dic' is 'white flowers', and 'pic muk dic' is 'flowers are fruits'. What is the code for 'white' in that code language? Arrangement and Deductive: Hitesh, Sunny, Vicky, Nitin, and Bharat are arranged in ascending order of height from the top. Hitesh is in third place. Bharat is between Nitin and Hitesh while Nitin is not at the bottom. Who has the maximum height among them? Direction: A and B start from the same point. A cycle 10 km South, then turns to her right and cycles 2 km. B cycles 2 km North. Please provide subject tags only from the list of 'Subject-Tags' provided, inside double quotes. Do not provide answers to specific questions."}
        user_input = test_sample["input"]
        prompt = [system_input, user_input]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response = result.get_completions()[0]
        expected = test_sample["ideal"]

        expected_str = json.dumps(expected).replace("[", "").replace("]", "")
        # response_sample = "{'level 1': 'physics', 'level 2': 'optics', 'level 3': 'mirrors'}"

        # response_sample = response_sample.replace("'", '"')

        tag_tree = evals.metrics.get_tag_tree(response)

        if tag_tree == []:
            return tag_tree

        with open("tag_tree.json", "a") as json_file:
            json.dump(tag_tree, json_file)

        # system_input_2 = {"role": "system", "content": f"Tag-Tree-List: {tag_tree}. ""Given a 'Tag-Tree-List' consisting of nested dictionaries representing hierarchical tags, your task as an educator is to identify and classify the correct tag tree from this list. You should then drill down into this selected tag tree to extract hierarchical topics and sub-topics of questions. The tag trees in the 'Tag-Tree-List' can have multiple levels, and you need to provide these levels in your output. The levels should be labeled as 'Level 1', 'Level 2', and so on, with the last level being the deepest level found in the selected tag tree. Your response should be in JSON format, structured as follows: {\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"} For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. In case none of the tag trees in the 'Tag-Tree-List' is correct, your answer should be 'None.' You should only provide levels that exist within the selected tag tree; do not add any extra levels. Your response should strictly adhere to the structure of the provided 'Tag-Tree-List.'"}

        system_input_2 = {"role": "system", "content": f"Tag-Tree-List: {tag_tree}. ""Given a 'Tag-Tree-List' consisting of nested dictionaries representing hierarchical tags, your role as an educator is to identify and classify the correct tag tree from this list. Subsequently, you should navigate through the chosen tag tree to discern hierarchical topics and sub-topics of questions. It's important to note that the tag trees within the 'Tag-Tree-List' can contain multiple levels, and your response should encompass these levels, denoted as 'Level 1,' 'Level 2,' and so forth. The potentially deepest level should reflect the context of the given question, which may or may not align with the absolute deepest level within the selected tag tree. Your response should be structured in JSON format, organized as follows: {\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For instance: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. In the event that more than one tag tree within the 'Tag-Tree-List' corresponds to the question, or if you find yourself uncertain between multiple tag trees, you should provide multiple JSON formats within a list to cover all potential interpretations. Each JSON format should accurately reflect the context of the question. Additionally, if you have selected a tag tree for the question and, during drilling down, you discover that after a certain node, the tag is no longer relevant or suitable for addressing the question, the last relevant node should be considered the deepest level of that suggestion, and you should return that only."}

        user_input_2 = test_sample["input"]
        prompt = [system_input_2, user_input_2]
        result_2 = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response_2 = result_2.get_completions()[0]

        print("response_2", response_2)

        # expected_levels = json.dumps(expected_str)
        # print("-----------------", type(expected_levels))
        # print("-----------------", type(expected_str))
        # print("-----------------", type(expected))
        response_levels = json.loads(response)
        # response_2_levels = json.loads(response_2)
        
        level_1, level_2, level_3 = expected['Level 1'][0], expected['Level 2'][0], expected["Level 3"][0]

        res_1_level_1, res_1_level_2, res_1_level_3 = response_levels.get('Level 1', ''), response_levels.get('Level 2', ''), response_levels.get('Level 3', '')

        # res_2_level_1, res_2_level_2, res_2_level_3, res_2_level_4, res_2_level_5, res_2_level_6, res_2_level_7, res_2_level_8, res_2_level_9, res_2_level_10 = response_2_levels.get('Level 1', ''), response_2_levels.get('Level 2', ''), response_2_levels.get('Level 3', ''), response_2_levels.get('Level 4', ''), response_2_levels.get('Level 5', ''), response_2_levels.get('Level 6', ''), response_2_levels.get('Level 7', ''), response_2_levels.get('Level 8', ''), response_2_levels.get('Level 9', ''), response_2_levels.get('Level 10', '')


        with open('Prompt_result.txt', 'a') as file:
            file.write("#######################################################\n")
            file.write(user_input["content"])
            file.write(f"\nexpected: {expected_str}\n")
            file.write(f"response 1: {response}\n")
            file.write(f"response 2: {response_2}\n")
            file.write("\n-----------------------------------------------------\n")
            # file.write(f"expected:\nLevel 1: {level_1}\nLevel 2: {level_2}\nLevel 3: {level_3}\n\n")
            # file.write(f"response 1:\nLevel 1: {res_1_level_1}\nLevel 2: {res_1_level_2}\nLevel 3: {res_1_level_3}\n\n")
            # file.write(f"response 2:\nLevel 1: {res_2_level_1}\nLevel 2: {res_2_level_2}\nLevel 3: {res_2_level_3}\nLevel 4: {res_2_level_4}\nLevel 5: {res_2_level_5}\nLevel 6: {res_2_level_6}\nLevel 7: {res_2_level_7}\nLevel 8: {res_2_level_8}\nLevel 9: {res_2_level_9}\nLevel 10: {res_2_level_10}\n")
            file.write("#######################################################\n\n")
        