import evals
import evals.metrics
import random
import openai
from evals.record import record_each_sample_match

openai.api_key = ""


class PromptTest(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl
        self.prompt_accuracy = 0
        self.prompt_subject_accuracy = 0
        self.correct_level_1 = 0

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        test_samples_length = len(test_samples)
        self.eval_all_samples(recorder, test_samples)

        prompt_accuracy = self.prompt_accuracy / test_samples_length
        prompt_subject_accuracy = self.prompt_subject_accuracy / test_samples_length
        prompt_correct_level_1 = self.correct_level_1

        file_accuracy = f"prompt_accuracy: {prompt_accuracy}\nprompt_subject_accuracy: {prompt_subject_accuracy}\nprompt_correct_level_1: {prompt_correct_level_1}\n"

        with open('evals_output.txt', 'a') as file:
            file.write(f"\n\n################################################\n")
            file.write(file_accuracy)
            file.write(f"################################################\n")

        print(f"file_accuracy: {file_accuracy}")


    def eval_sample(self, test_sample, rng: random.Random):
        # system_input = {"role": "system", "content": "As an educator responsible for classifying and identifying tags for questions in Indian competitive examinations, your goal is to add a subject tag and delve into the hierarchical topics and sub-topics of the questions. You should provide three levels of tags: Level 1, Level 2, and Level 3, in JSON format, structured as follows: {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. You should select the subject tag only from the list of 'Subject-Tags' provided: ['History', 'Information and Communication Technology', 'Judiciary and Law', 'Legal Reasoning', 'Mathematics', 'Numerical Aptitude', 'Physics', 'Polity', 'Reasoning', 'Static General Knowledge', 'Geography', 'Strategic Management', 'Teaching Methodology', 'Agriculture', 'Aviation', 'Accountancy', 'Indian Art and Culture', 'Statistics', 'Biology', 'Business Management', 'Child development and pedagogy', 'Disaster Management', 'Economy', 'Financial Management', 'General Awareness', 'Chemistry', 'English', 'Environment']. To identify subject tags accurately, consider the following criteria: Mathematics: This subject encompasses a wide range of mathematical concepts, theories, and principles, including both practical and abstract mathematics. It involves advanced topics that require a deep understanding of mathematical theory and principles. Questions in this category typically include advanced mathematical calculations, proofs, and abstract mathematical concepts. Numerical Aptitude: Numerical aptitude tests primarily assess practical numerical skills. Questions in this category involve arithmetic calculations, percentages, ratios, data interpretation, and solving numerical word problems. These tasks are usually of moderate complexity and focus on basic to intermediate mathematical operations without requiring advanced mathematical theory. Reasoning: Reasoning tests evaluate logical and analytical thinking skills. They include tasks such as logical puzzles, pattern recognition, deciphering a code language, deductive reasoning, analogies, and abstract reasoning. Unlike mathematics, reasoning questions do not involve mathematical calculations but instead emphasize problem-solving and logical deduction. English Language: The English subject involves language proficiency and comprehension. Questions typically revolve around reading passages, correcting sentences, identifying synonyms and antonyms, assessing comprehension, and evaluating writing skills. It focuses on language usage, grammar, and reading comprehension, distinct from mathematical or logical content. General Awareness: General awareness questions cover a broad spectrum of topics, including current affairs, history, geography, science, politics, culture, and more. They assess a candidate's knowledge and awareness of various subjects. Unlike the other categories, general awareness does not require mathematical or logical reasoning but relies on factual knowledge. Please provide subject tags only from the list of 'Subject-Tags' provided, inside double quotes. Do not provide answers to specific questions."}

        # system_input = {"role": "system", "content": "As an educator responsible for classifying and identifying tags for questions in Indian competitive examinations, your goal is to add a subject tag and delve into the hierarchical topics and sub-topics of the questions. You should provide three levels of tags: Level 1, Level 2, and Level 3, in JSON format, structured as follows: {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. To minimize confusion between subjects, you should select the subject tag only from the list of 'Subject-Tags' provided, which are: ['History', 'Information and Communication Technology', 'Judiciary and Law', 'Legal Reasoning', 'Mathematics', 'Numerical Aptitude', 'Physics', 'Polity', 'Reasoning', 'Static General Knowledge', 'Geography', 'Strategic Management', 'Teaching Methodology', 'Agriculture', 'Aviation', 'Accountancy', 'Indian Art and Culture', 'Statistics', 'Biology', 'Business Management', 'Child development and pedagogy', 'Disaster Management', 'Economy', 'Financial Management', 'General Awareness', 'Chemistry', 'English', 'Environment']. To identify subject tags accurately, consider the following refined criteria: Mathematics: This subject encompasses a wide range of advanced mathematical concepts, theories, and principles, including both practical and abstract mathematics. Questions in this category typically involve complex mathematical calculations, proofs, and abstract mathematical concepts. Numerical Aptitude: Numerical aptitude tests primarily assess practical numerical skills, including arithmetic calculations, percentages, ratios, data interpretation, and solving numerical word problems. Questions in this category focus on basic to intermediate mathematical operations. Reasoning: Reasoning tests evaluate logical and analytical thinking skills. They include tasks such as logical puzzles, pattern recognition, deciphering a code language, deductive reasoning, analogies, and abstract reasoning. Unlike mathematics, reasoning questions do not involve mathematical calculations but emphasize problem-solving and logical deduction. English Language: The English subject involves language proficiency and comprehension. Questions typically revolve around reading passages, correcting sentences, identifying synonyms and antonyms, assessing comprehension, and evaluating writing skills. It focuses on language usage, grammar, and reading comprehension, distinct from mathematical or logical content. General Awareness: General awareness questions cover a broad spectrum of topics, including current affairs, history, geography, science, politics, culture, and more. They assess a candidate's knowledge and awareness of various subjects. Unlike the other categories, general awareness does not require mathematical or logical reasoning but relies on factual knowledge. By refining the criteria and providing clear distinctions between subjects, you can minimize confusion and improve the accuracy of subject classification in the context of competitive examinations. Please provide subject tags only from the list of 'Subject-Tags' provided, inside double quotes. Do not provide answers to specific questions."}


        # Please provide subject tags based on the predominant content and context of the question, and use your discretion for questions that do not fit into a single subject category.

        system_input = {"role": "system", "content": "As an educator responsible for classifying and identifying tags for questions in Indian competitive examinations, your goal is to add a subject tag and delve into the hierarchical topics and sub-topics of the questions. You should provide three levels of tags: Level 1, Level 2, and Level 3, in JSON format, structured as follows: {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example: {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. To minimize confusion between subjects, you should select the subject tag only from the list of 'Subject-Tags' provided, which are: ['History', 'Information and Communication Technology', 'Judiciary and Law', 'Legal Reasoning', 'Mathematics', 'Numerical Aptitude', 'Physics', 'Polity', 'Reasoning', 'Static General Knowledge', 'Geography', 'Strategic Management', 'Teaching Methodology', 'Agriculture', 'Aviation', 'Accountancy', 'Indian Art and Culture', 'Statistics', 'Biology', 'Business Management', 'Child development and pedagogy', 'Disaster Management', 'Economy', 'Financial Management', 'General Awareness', 'Chemistry', 'English', 'Environment']. To identify subject tags accurately, consider the following refined criteria and specific keywords: Mathematics: This subject encompasses a wide range of advanced mathematical concepts, theories, and principles, including both practical and abstract mathematics. Questions in this category typically involve complex mathematical calculations, proofs, and abstract mathematical concepts. Numerical Aptitude: Numerical aptitude tests primarily assess practical numerical skills, including arithmetic calculations, percentages, ratios, data interpretation, and solving numerical word problems. Questions in this category focus on basic to intermediate mathematical operations. Reasoning: Reasoning tests evaluate logical and analytical thinking skills, such as logical puzzles, pattern recognition, deciphering a code language, deductive reasoning, analogies, and abstract reasoning. Unlike mathematics, reasoning questions do not involve mathematical calculations but emphasize problem-solving and logical deduction. English Language: The English subject involves language proficiency and comprehension. Questions typically revolve around reading passages, correcting sentences, identifying synonyms and antonyms, assessing comprehension, and evaluating writing skills. It focuses on language usage, grammar, and reading comprehension, distinct from mathematical or logical content. General Awareness: General awareness questions cover a broad spectrum of topics, including current affairs, history, geography, science, politics, culture, and more. They assess a candidate's knowledge and awareness of various subjects. Unlike the other categories, general awareness does not require mathematical or logical reasoning but relies on factual knowledge. Please provide subject tags only from the list of 'Subject-Tags' provided, inside double quotes. Do not provide answers to specific questions."}

        user_input = test_sample["input"]
        prompt = [system_input, user_input]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response = result.get_completions()[0]
        expected = test_sample["ideal"]

        sample_accuracy = evals.metrics.get_each_sample_accuracy(response, expected, user_input)

        self.prompt_accuracy += sample_accuracy["combined_tag_score"]
        self.prompt_subject_accuracy += sample_accuracy["subject_tags_score"]
        self.correct_level_1 += 1 if sample_accuracy["correct_level_1"] == True else 0
