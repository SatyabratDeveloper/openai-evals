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
        system_input = {"role": "system", "content": "You are an educator who has been given the task of classifying and identifying tags for the questions that are asked in Indian competitive examinations. You want to add the subject tag and drill down on the hierarchical topics and sub-topics of the questions. You need to provide three levels of tags: Level 1, Level 2, and Level 3. The answer should be in JSON format, like {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. Select the subject tag only from the below-mentioned 'Subject-Tags'. Subject tags are mentioned inside four quotes. Subject-Tags: '''' History, Information and Communication Technology, Judiciary and Law, Legal Reasoning, Mathematics, Numerical Aptitude, Physics, Polity, Reasoning, Static General Knowledge, Geography, Strategic Management, Teaching Methodology, Agriculture, Aviation, Accountancy, Indian Art and Culture, Statistics, Biology, Business Management, Child development and pedagogy, Disaster Management, Economy, Financial Management, General Awareness, Chemistry, English, Environment ''''. Find subject-tags based on these points: 1. Key differences between 'Numerical Aptitude' and 'Mathematics' subjects: Numerical aptitude focuses on practical, real-world math skills and problem-solving, while mathematics as a subject encompasses a broader range of mathematical concepts, theories, and principles. Mathematics subject can become highly complex, involving advanced topics such as calculus, abstract algebra, and differential equations, while numerical aptitude typically deals with more straightforward mathematical tasks. Numerical aptitude tests often assess skills such as: Basic calculations, percentages, discounts, increases, ratios and proportions, data interpretation(analyzing and drawing conclusions from numerical data presented in tables, charts, or graphs), and word problems(solving mathematical problems presented in a textual format). 2. English subject: Look for keywords such as grammar, vocabulary, comprehension, fill in the blanks or literature. NOTE: ''Provide subject tags only from the above mentioned 'Subject-Tags' inside four quotes''."}
        user_input = test_sample["input"]
        prompt = [system_input, user_input]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response = result.get_completions()[0]
        expected = test_sample["ideal"]

        # response_sample = '{"Level 1": "Numerical Aptitude", "Level 2": "Mensuration", "Level 3": ""}'

        tag_tree = evals.metrics.get_tag_tree(response)

        with open("tag_tree.json", "a") as json_file:
            json.dump(tag_tree, json_file)

        system_input_2 = {"role": "system", "content": f"You are an educator who has been given the task of classifying and identifying tags for the questions that are asked in Indian competitive examinations. You want to drill down on the hierarchical topics and sub-topics of the questions from the below given 'Tag-Tree-List'. Tag-Tree-List: {tag_tree}. ""Tag-Tree-List is a list of dictionaries in which every dictionary is a tag tree. A tag tree is a nested dictionary which is a hierarchical tree structure with a set of connected nodes. Tag-Tree-List might have more than one dictionary (tag trees). You need to classify the correct tag tree from the Tag-Tree-List and drill down the hierarchical topics and sub-topics of the questions only from the above mentioned Tag-Tree-List. You need to provide some levels of tags: Level 1, Level 2, Level 3. The answer should be in JSON format, like {\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. Note: ''None of the tag trees may be correct, which is mentioned in Tag-Tree-List. The answer of that prompt should be: 'None''."}

        user_input_2 = test_sample["input"]
        prompt = [system_input_2, user_input_2]
        result_2 = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response_2 = result_2.get_completions()[0]

        print("response_2", response_2)
        