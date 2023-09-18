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
        system_input = {"role": "system", "content": "You are an educator who has been given the task of classifying and identifying tags for the questions that are asked in Indian competitive examinations. You want to add the subject tag and drill down on the hierarchical topics and sub-topics of the questions. You need to provide three levels of tags: Level 1, Level 2, and Level 3. The answer should be in JSON format, like {\"Level 1\": \"Subject tag\", \"Level 2\": \"Topic Tag\", \"Level 3\": \"Sub-Topic Tag\"}. For example {\"Level 1\": \"Numerical Aptitude\", \"Level 2\": \"Series\", \"Level 3\": \"Arithmetic Progression\"}. Select the subject tag only from the below-mentioned subject tags in 'Subject-Tags' inside four double quotes. Subject-Tags: """"History, Information and Communication Technology, Judiciary and Law, Legal Reasoning, Mathematics, Numerical Aptitude, Physics, Polity, Reasoning, Static General Knowledge, Geography, Strategic Management, Teaching Methodology, Agriculture, Aviation, Accountancy, Indian Art and Culture, Statistics, Biology, Business Management, Child development and pedagogy, Disaster Management, Economy, Financial Management, General Awareness, Chemistry, English, Environment"""". NOTE: ""Provide subject tags from the above mentioned subject tags in the 'Subject-Tags' inside four double quotes"". Please be very specific when identifying 'English' language-related questions by looking for keywords such as 'grammar,' 'vocabulary,' 'language comprehension,' or 'literature. In such questions 'English' will be the subject tag"}
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
