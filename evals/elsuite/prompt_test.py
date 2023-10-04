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
        self.eval_all_samples(recorder, test_samples)


    def eval_sample(self, test_sample, rng: random.Random):
        system_input = {"role": "system", "content": "As an educator responsible for classifying and identifying tags for questions in Indian competitive examinations, your goal is to add a subject tag and delve into the hierarchical topics and sub-topics of the questions. You should provide three levels of tags: Level 1, Level 2, and Level 3, in JSON format, structured as follows: {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. You should select the subject tag only from the list of 'Subject-Tags' provided, which are: ['History', 'Information and Communication Technology', 'Judiciary and Law', 'Legal Reasoning', 'Mathematics', 'Numerical Aptitude', 'Physics', 'Polity', 'Reasoning', 'Static General Knowledge', 'Geography', 'Strategic Management', 'Teaching Methodology', 'Agriculture', 'Aviation', 'Accountancy', 'Indian Art and Culture', 'Statistics', 'Biology', 'Business Management', 'Child development and pedagogy', 'Disaster Management', 'Economy', 'Financial Management', 'General Awareness', 'Chemistry', 'English', 'Environment']. To identify subject tags accurately, consider the following criteria and specific keywords: Mathematics: This subject encompasses a wide range of advanced mathematical concepts, theories, and principles, including both practical and abstract mathematics. Questions in this category typically involve complex mathematical calculations, proofs, and abstract mathematical concepts. Numerical Aptitude: Numerical aptitude tests primarily assess practical numerical skills. Questions in this category focus on basic to intermediate mathematical operations. Numerical Aptitude keywords: number system, LCM and HCF, divisibility, decimal fractions, geometry, mensuration, compound interest, simple interest, area, shapes and perimeter, percentage, allegation and mixture, ratio and proportion, partnership, time and work, speed, time and distance, boat and stream, train, profit and loss, average, equation, age, probability, algebra, clock and calendar, series, progression, mean, median, mode, variance, and standard deviation, pie charts, tabular data interpretation, graphical data interpretation, simplification, and approximation. Reasoning: Reasoning tests evaluate logical and analytical thinking skills. Unlike mathematics, reasoning questions do not involve mathematical calculations but emphasise problem-solving and logical deduction. Reasoning keywords: series, alphanumeric series, position, analogies, artificial language, blood relation, calendar, cause and effect, clock, coding, decoding, critical path, cube and cuboid, data sufficiency, decision making, deductive/statement analysis, dice, direction, figure matrix, input and output, odd one out, picture series/sequences, paper folding, puzzles, pattern series/sequences, order and ranking, seating arrangement, statement and assumption, statement and conclusion, and syllogism. English: The English subject involves language proficiency and comprehension. Questions typically revolve around reading passages, correcting sentences, identifying synonyms and antonyms, assessing comprehension, and evaluating writing skills. It focuses on language usage, grammar, and reading comprehension, distinct from mathematical or logical content. English keywords: reading comprehension, cloze test, fill in the blanks, tenses rules, rearrangement, jumbled sentences/words, error detection, preposition rules, paragraph completion, idioms and phrases, meaningful sentences. Generally, the subject tag of a question that is related to the passage and rearrangement of a sentence/passage is English. General Awareness: General awareness questions cover a broad spectrum of topics, including current affairs, history, geography, science, politics, culture, and more. They assess a candidate's knowledge and awareness of various subjects. Unlike the other categories, general awareness does not require mathematical or logical reasoning but relies on factual knowledge. Additionally, for questions that involve logical reasoning, deductive analysis, and problem-solving without direct mathematical calculations, you should classify them under the 'Reasoning' subject tag. These questions assess a candidate's logical and analytical thinking skills and do not require mathematical or numerical operations. Here are some example questions that fall under the 'Reasoning' subject tag: 1. A and B start from the same point. A cycle 10 km South, then turns to her right and cycles 2 km. B cycles 2 km North, then turns West and cycles 15 km, then turns to her left and cycles 12 km. Where is B with respect to A now? 2. In a line of boys Aman is 12th from the top and Baman is 18th from the bottom. If there are 6 boys between Aman and Baman, then how many boys are there in the row? 3. In a row of 74 girls, Shweta is 27th from the left end. Palak is 7th to the right of Shweta. What is Palak�s position from the right end of the row? 4. Select the correct alternative to indicate the arrangement of the following words in a logical and meaningful order. 5. Six years ago Parvez's age was the same as the present age of Manish. If the present age of Parvez is one-fourth more than that of Manish's present age, then in how many years will Parvez's age become double of Manish's present age? 6. If 2 # 3 = 6; 15 # 3 = 2; 60 # 4 = 32; then what is the value of 27 # 3 = ? 7. In a certain code language, �+� represents �ג, ��� represents �+�, �ג represents ��� and ��� represents ���. What is the answer to the following question? To further assist you in identifying the subject tag, consider the instructions provided within each question. In many cases, you can determine the subject tag by following the instructions and the nature of the question itself. These added instructions emphasize the importance of considering the question's instructions and nature when determining the subject tag. Please provide subject tags only from the list of 'Subject-Tags' provided, inside double quotes in JSON format only. Do not provide answers to specific questions."}
        user_input = test_sample["input"]
        prompt = [system_input, user_input]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        # Open AI Response
        response = result.get_completions()[0]

        # Expected Response
        expected = test_sample["ideal"]
        expected_response = {"Level 1": expected['Level 1'][0].lower(), "Level 2": expected['Level 2'][0].lower(), "Level 3": [item.lower() for item in expected['Level 3']]}
        print("expected_response", expected_response)
        print("response", response)
        
        # response_sample = '{"Level 1": "Numerical Aptitude", "Level 2": "Simple Interest and Compound Interest", "Level 3": "Comparison of Simple Interest and Compound Interest"}'

        tag_tree = evals.metrics.get_tag_tree(response)

        if tag_tree == []:
            return tag_tree

        with open("tag_tree.json", "a") as json_file:
            json.dump(tag_tree, json_file)

        system_input_2 = {"role": "system", "content": f"Tag-Tree-List: {tag_tree}. ""Within the 'Tag-Tree-List,' you will find nested dictionaries representing hierarchical tags. As an educator, your task is to accurately identify and classify the relevant tag tree(s) for the given question. Once you've determined the appropriate tag tree(s), your goal is to extract the specific hierarchical topics and sub-topics that directly relate to the question. Please pay close attention to the following guidelines: 1. Precision of Tags: Your response should only include tags that exist within the provided 'Tag-Tree-List.' Do not introduce additional tags that are not present in the list. 2. Consistency within Branches: Tags should come from the same branch of the tag tree. Do not mix and match tags from different branches within a single response. 3. Relevance to the Question: Tags should directly relate to the question. Avoid including all possible combinations of levels present in the 'Tag-Tree-List,' as this can lead to irrelevant tags in the response. 4. Multiple Tag Trees: If the question is applicable to multiple tag trees within the 'Tag-Tree-List,' provide separate JSON responses for each applicable tag tree. Each JSON response should accurately represent the context of the question. To structure your response, use a list of JSON structures as follows: [{\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}], and so on. For example: [{\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}]. To illustrate, if a question is applicable to both 'Simple Interest' and 'Compound Interest' on different branches of the provided tag tree, your JSON response should look like this: [{\"Level 1\": \"Simple Interest\", \"Level 2\": \"Terminology of SI\"}, {\"Level 1\": \"Compound Interest\", \"Level 2\": \"Terminology of CI\"}]. Ensure that your responses strictly adhere to these guidelines to provide accurate and contextually relevant tags. Your response should be a list of JSON structures only as follows: [{\"Level 1\": \"Topic tag\", \"Level 2\": \"Sub-Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}]."}
        user_input_2 = test_sample["input"]
        prompt = [system_input_2, user_input_2]
        result_2 = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response_2 = result_2.get_completions()[0]

        print("response_2", response_2)

        final_tree_nodes = evals.metrics.get_final_tree_nodes(response_2)

        # with open('Prompt_result.txt', 'a') as file:
        #     file.write("#######################################################\n")
        #     file.write(user_input["content"])
        #     file.write("\n-----------------------------------------------------\n")
        #     file.write(f"expected: {expected_response}\n")
        #     file.write(f"response 1: {response}\n")
        #     file.write(f"response 2: {response_2}\n")
        #     file.write("#######################################################\n\n")
        