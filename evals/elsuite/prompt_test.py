import evals
import evals.metrics
import random
import openai
import logging
from evals.record import record_each_sample_match

openai.api_key = ""


class PromptTest(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl
        self.prompt_accuracy = 0
        self.prompt_subject_accuracy = 0
        self.prompt_skill_accuracy = 0
        self.correct_level_1 = 0

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        test_samples_length = len(test_samples)
        self.eval_all_samples(recorder, test_samples)

        prompt_accuracy = self.prompt_accuracy / test_samples_length
        prompt_subject_accuracy = self.prompt_subject_accuracy / test_samples_length
        prompt_skill_accuracy = self.prompt_skill_accuracy / test_samples_length
        prompt_correct_level_1 = self.correct_level_1

        file_accuracy = {'prompt_accuracy': prompt_accuracy, 'prompt_subject_accuracy': prompt_subject_accuracy, 'prompt_skill_accuracy': prompt_skill_accuracy, 'prompt_correct_level_1': prompt_correct_level_1}

        print(f"file_accuracy: {file_accuracy}")

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, test_sample, rng: random.Random):
        system_input = {"role": "system", "content": "You are an educator who has been tasked to classify and tag MCQ questions. You want to add subject tags and drilled down hierarchical topics tags to questions. These subjects are up to competitive exams. Choose from one of the following subjects. A question may have tags containing multiple subjects or skills. Subject tag, Level 1, Level 2, Level 3 must be a single string value. Also suggest skill tags as per revised Bloom's taxonomy. For the following question list down the applicable tags. Answer should be single line JSON for e.g. {'Level 1': 'Numerical Aptitude', 'Level 2': 'series', 'Level 3': 'Arithmetic Progression', 'Skills': ['tag1', 'tag2','tag3']}. Skill tag should be attached on the basis of direction given below : {Recognize,In questions which are in the format of present or future questions and also suggested with all general awareness .} {Recalling,In questions which are in the format of past questions.} {Interpreting,Solve/answer the question by understanding the meaning on what question is focusing or highlighting or when words classify, covert, conclude, demonstrate, describe, discuss, explain, identify, illustrate, locate, paraphrase, predict, recognize, report, select, summarize, translate,most appropriate,meaning,substitute,synonyms,antonyms,idioms,phrases are given in question.} {Exemplifying, Questions that ask to give examples of anything.} {Classifying, Questions which ask to classify or categorize the given things/topics.} {Summarizing,If the question is asking to summarize anything or wants any information in a brief manner.} {Inferring,When you have to draw a logical conclusion from the given information or context or when no direction is given to how to solve the given question or problem.} {Comparing, when question is asked to compare or same way as another number,letter or group} {Implementing,If a learned knowledge is implemented to answer any unfamiliar situation/problem to solve the question.} {Organizing,When the question asks to organize the information about any topic or the given material/information in a way that it fits or functions within a structure or when a question is asked to arrange or rearrange the given data.} {Checking,when question is asked to find the incorrect or wrongly spelt or not is given in the main heading or direction} {Knowledge of terminology,this knowledge tag is suggested when a specific term is given or question is asking a specific term related to their field} {Knowledge of specific details and elements, this knowledge tag is suggested when the question is asking for some details or with every general awareness question} {Knowledge of classifications and categories,when question is asked to classify the data on the basis of their category} {Knowledge of principles and generalizations, when any principle or formula is asked in question} {Knowledge of subject specific skills and algorithms, when question is asked to follow a similar pattern related to given data} {Knowledge of subject-specific technique and method, this skill tag should be attached with implementing skill tag or when their is a formula required to applied in queston} Subjects: History, Information and Communication Technology, Judiciary and Law, Legal Reasoning, Mathematics, Numerical Aptitude, Physics, Polity, Reasoning, Static General Knowledge, Geography, Strategic Management, Teaching Methodology, Agriculture, Aviation, Accountancy, Indian Art and Culture, Statistics, Biology, Business Management, Child development and pedagogy, Disaster Management, Economy, Financial Management, General Awareness, Chemistry, English, Environment."}
        user_input = test_sample["input"]
        prompt = [system_input, user_input]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=500
        )
        response = result.get_completions()[0]
        expected = test_sample["ideal"]

        # logging.info(f"Output: {response}")
        # logging.info(f"Expected Output: {expected}")

        sample_accuracy = evals.metrics.get_each_sample_accuracy(response, expected)

        self.prompt_accuracy += sample_accuracy["combined_tag_score"]
        self.prompt_subject_accuracy += sample_accuracy["subject_tags_score"]
        self.prompt_skill_accuracy += sample_accuracy["skill_tags_score"]
        self.correct_level_1 += 1 if sample_accuracy["correct_level_1"] == True else 0

        output = {"Output": response, "Expected Output": expected, "Tag Accuracy": sample_accuracy}
        
        record_each_sample_match(output)
        