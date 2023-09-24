"""
This file defines various common metrics of interest.
"""
import random
from typing import Optional, Sequence, Set
import numpy as np
import json
from evals.record import Event
import re

# Import the ModelEmbeddings class from the misc module
# from string2string.misc import ModelEmbeddings

# Import the CosineSimilarity class from the similarity module
# from string2string.similarity import CosineSimilarity

# Create an instance of the ModelEmbeddings class (if device is not specified, it will be automatically detected)
# bart_model = ModelEmbeddings(
#     model_name_or_path='facebook/bart-large'
# )

# # Create an instance of the CosineSimilarity class
# cosine_similarity = CosineSimilarity()


def get_accuracy(events: Sequence[Event]) -> float:
    num_correct = sum(int(event.data["correct"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total
    
def get_subject_tags_accuracy(subject_tags, expected_subject_tags, user_input) -> float:
    subject_accuracy = 0
    inverted_value = 0
    matched_level_list = []
    level_1 = ''
    level_2 = ''
    level_3 = ''
    expected_level_1 = ''
    expected_level_2 = ''
    expected_level_3 = []

    correct_level_1 = False

    output_list = list(subject_tags.values())
    expected_list = list(expected_subject_tags.values())

    # output_arr_length = len(output_list)
    expected_arr_length = len(expected_list)

    # loop to find inverted tags and assigning the levels values
    for expected_index, expected_value in enumerate(expected_list):
        if expected_index == 0:
            expected_level_1 = expected_value[0]
        if expected_index == 1:
            expected_level_2 = expected_value[0]
        if expected_index == 2:
            expected_level_3 = expected_value
        
        for output_index, output_value in enumerate(output_list):
            if output_index == 0:
                level_1 = output_value
            if output_index == 1:
                level_2 = output_value
            if output_index == 2:
                level_3 = output_value
            
            if expected_index == 0 or expected_index == 1:
                if expected_value[0] == output_value:
                    matched_level_list.append(output_index)
            if expected_index == 2:
                if output_value in expected_value:
                    matched_level_list.append(output_index)

    for element in matched_level_list:
        if element >= inverted_value:
            inverted_value = element
        else:
            inverted_value = 'Inverted Output'
            break

    # Subject Comparison
    if inverted_value != 'Inverted Output':
        if expected_level_1 == level_1:
            correct_level_1 = True
            subject_accuracy += 50 if expected_arr_length == 3 else (60 if expected_arr_length == 2 else 100)
        else:
            with open('incorrect.txt', 'a') as file:
                file.write(f"Question: {user_input['content']}\n\nExpected: Level 1: {expected_level_1}, Level 2: {expected_level_2}, Level 3: {expected_level_3}\nOutput: Level 1: {level_1}, Level 2: {level_2}, Level 3: {level_3}\n\n")
        if expected_level_2 == level_2 or expected_level_2 == level_3:
            subject_accuracy += 20 if expected_arr_length == 3 else 40
        # elif expected_level_2:
        #     similarity_input = [expected_level_2, level_2, level_3]
        #     if None in similarity_input:
        #         similarity_input = [item for item in similarity_input if item is not None]
        #     result = similarity(similarity_input)
        #     if result == True:
        #         subject_accuracy += 20 if expected_arr_length == 3 else 40
        if level_2 in expected_level_3 or level_3 in expected_level_3:
            subject_accuracy += 30
        # elif expected_level_3:
        #     similarity_input = expected_level_3 + [level_2, level_3]
        #     if None in similarity_input:
        #         similarity_input = [item for item in similarity_input if item is not None]
        #     result = similarity(similarity_input)
        #     if result == True:
        #         subject_accuracy += 30
    else:
        if expected_level_1 == level_1:
            subject_accuracy += 50 if expected_arr_length == 3 else 60
        else:
            with open('incorrect.txt', 'a') as file:
                file.write(f"{user_input}\n\nExpected: Level 1: {expected_level_1}, Level 2: {expected_level_2}, Level 3: {expected_level_3}\nOutput: Level 1: {level_1}, Level 2: {level_2}, Level 3: {level_3}\n\n")
    
    return {"subject_accuracy": subject_accuracy, "correct_level_1": correct_level_1}

def get_lowercase_dictionary(response, expected):
    # Convert the JSON string to a Python dictionary
    print("response", response)
    # if '.' in response:
    #     response = response.replace('.', '')
    
    # Regex to find response JSON
    pattern = r'{[^}]+}'

    # Use re.findall to extract all matches
    match = re.findall(pattern, response)
    response_dict = json.loads(match[0])

    # Replace None with an empty string in the data dictionary
    for key, value in response_dict.items():
        if value is None:
            response_dict[key] = ""

    # Lowercase dictionary while handling None values
    response_lowercase_dict = {
        key.lower(): value.lower() if value is not None and isinstance(value, str) else (
            [item.lower() for item in value] if isinstance(value, list) and all(isinstance(item, str) for item in value) else value
        )
        for key, value in response_dict.items()
    }

    expected_lowercase_dict = {
        key.lower(): value.lower() if value is not None and isinstance(value, str) else (
            [item.lower() for item in value] if isinstance(value, list) and all(isinstance(item, str) for item in value) else value
        )
        for key, value in expected.items()
    }

    return response_lowercase_dict, expected_lowercase_dict

def get_subject_and_skill_tags(response_dict, expected_dict):
    subject_tags = {}
    expected_subject_tags = {}

    # Loop to split response subject and skill tags
    for key, value in response_dict.items():
        if key.startswith('level'):
            subject_tags[key] = value
    # Loop to split expected subject and skill tags
    for key, value in expected_dict.items():
        if key.startswith('level'):
            expected_subject_tags[key] = value
    
    return subject_tags, expected_subject_tags

def get_each_sample_accuracy(response, expected, user_input) -> float:
    test_accuracy = 0

    # To get lowercase response and expected
    response_dict, expected_dict = get_lowercase_dictionary(response, expected)

    # To split subject and skill tags
    subject_tags, expected_subject_tags = get_subject_and_skill_tags(response_dict, expected_dict)

    # To get subject and skill accuracy
    subject_accuracy = get_subject_tags_accuracy(subject_tags, expected_subject_tags, user_input)

    # To get test accuracy and set threshold
    test_accuracy = subject_accuracy["subject_accuracy"]
    test_pass = test_accuracy >= 75

    with open('evals_output.txt', 'a') as file:
        file.write(f"======================================================\n")
        file.write(f"Subject tag output: {subject_tags}\n")
        file.write(f"Subject tag expected: {expected_subject_tags}\n")
        file.write(f"******************************************************\n")
        file.write(f"Subject Tags Accuracy: {subject_accuracy['subject_accuracy']}\n")
        file.write(f"Test Accuracy: {test_accuracy}\n")
        file.write(f"Test pass: {test_accuracy}\n")
        file.write(f"======================================================\n\n")
    
    # Logs
    print("======================================================")
    print("Subject tag output", subject_tags)
    print("Subject tag expected", expected_subject_tags)
    print("******************************************************")
    print("Subject Tags Accuracy", subject_accuracy["subject_accuracy"])
    print("Test Accuracy", test_accuracy)
    print("Test pass", test_accuracy)
    print("======================================================")
    
    return {
        'pass': test_pass,
        'subject_tags_score': subject_accuracy["subject_accuracy"],
        'combined_tag_score': test_accuracy,
        'correct_level_1': subject_accuracy["correct_level_1"],
        'reason': 'output matched' if test_pass else 'Output did not matched',
    }


# def similarity_checker(sentences):
#     similarity_result = False
#     embeds = []

#     # Compute the sentence embeddings for each sentence
#     for sentence in sentences:
#         embedding = bart_model.get_embeddings(sentence, embedding_type='mean_pooling')
#         embeds.append(embedding)

#     # Compute the cosine similarity between each pair of embeddings
#     for i in range(1):
#         for j in range(i + 1, len(embeds)):
#             result = cosine_similarity.compute(embeds[i], embeds[j], dim=1).item()
#             # print(f'The cosine similarity between the BART embeddings of Expeected Level: {sentences[i]} and Output Level: {sentences[j]} is {result:.2f}')
#             if result >= 0.82:
#                 similarity_result = True
#                 return similarity_result
#     return similarity_result

def similarity(str1, str2):
    stop_words = [
        "usually", "us", "upon", "until", "under", "use", "relate", "related", "relatively", "regarding",
        "quite", "n", "necessary", "to", "based", "than", "that", "those", "this", "there", "three", "o",
        "of", "one", "or", "on", "a", "after", "an", "any", "and", "are", "accordingly", "among", "all", "as",
        "vs", "v", "via", "very", "versus", "k", "g", "go", "b", "by", "both", "but", "be", "because",
        "between", "h", "how", "w", "was", "why", "what", "when", "where", "while", "whose", "s", "should",
        "said", "so", "some", "such", "since", "p", "l", "less", "ie", "ifs", "if", "i", "is", "in", "f",
        "from", "for", "d", "did", "c", "e", "eg"
    ]

    def word_count_map(s):
        words = re.split(r'[ -/,/]', s)
        word_count = {}
        for w in words:
            w = w.lower()
            if w in stop_words:
                continue
            word_count[w] = word_count.get(w, 0) + 1
        return word_count

    def add_words_to_dictionary(word_count_map, dictionary):
        for key in word_count_map:
            dictionary[key] = True

    def word_map_to_vector(word_count_map, dictionary):
        word_count_vector = []
        for term in dictionary:
            word_count_vector.append(word_count_map.get(term, 0))
        return word_count_vector

    def dot_product(vec_a, vec_b):
        product = 0
        for i in range(len(vec_a)):
            product += vec_a[i] * vec_b[i]
        return product

    def magnitude(vec):
        sum_sq = sum(x * x for x in vec)
        return math.sqrt(sum_sq)

    def cosine_similarity(vec_a, vec_b):
        dot_prod = dot_product(vec_a, vec_b)
        mag_a = magnitude(vec_a)
        mag_b = magnitude(vec_b)

        if mag_a == 0 or mag_b == 0:
            return 0  # Return 0 similarity when one of the vectors has zero magnitude
        else:
            return dot_prod / (mag_a * mag_b)

    def text_cosine_similarity(txt_a, txt_b):
        word_count_a = word_count_map(txt_a)
        word_count_b = word_count_map(txt_b)
        dictionary = {}
        add_words_to_dictionary(word_count_a, dictionary)
        add_words_to_dictionary(word_count_b, dictionary)
        vector_a = word_map_to_vector(word_count_a, dictionary)
        vector_b = word_map_to_vector(word_count_b, dictionary)
        similarity = cosine_similarity(vector_a, vector_b)
        return similarity

    def get_similarity_score(val):
        return round(val * 100)

    similarity_score = text_cosine_similarity(str1, str2)
    # print(f"Similarity Score: {get_similarity_score(similarity_score)}%")

    return get_similarity_score(similarity_score)


def get_bootstrap_accuracy_std(events: Sequence[Event], num_samples: int = 1000) -> float:
    vals = [m.data["correct"] for m in events]
    return np.std([np.mean(random.sample(vals, len(vals) // 2)) for _ in range(num_samples)])


def get_confusion_matrix(
        matches: Sequence[Event], class_labels: Optional[Set] = None
) -> np.ndarray:
    labels = {match.data["expected"] for match in matches}
    if class_labels is None:
        labels = {label: i for i, label in enumerate(sorted(labels))}
    else:
        assert labels.issubset(class_labels)
        labels = {label: i for i, label in enumerate(class_labels)}
    result = np.zeros((len(labels), len(labels) + 1), dtype=int)
    for match in matches:
        i = labels[match.data["expected"]]
        j = labels.get(match.data["picked"], len(labels))
        result[i, j] += 1
    return result


def compute_matthew_corr(confusion_matrix: np.ndarray) -> float:
    assert confusion_matrix.shape == (2, 3), f"Got shape: {confusion_matrix.shape}"
    r = confusion_matrix[:, :2]
    r[:, 0] += confusion_matrix[:, 2]
    return (r[1, 1] * r[0, 0] - r[1, 0] * r[0, 1]) / np.sqrt(
        r[1, :].sum() * r[0, :].sum() * r[:, 0].sum() * r[:, 1].sum()
    )


def compute_precision(confusion_matrix: np.ndarray, idx: int = 0) -> float:
    return confusion_matrix[idx, idx] / confusion_matrix[:, idx].sum()


def compute_recall(confusion_matrix: np.ndarray, idx: int = 0) -> float:
    return confusion_matrix[idx, idx] / confusion_matrix[idx, :].sum()


def compute_f_score(confusion_matrix: np.ndarray, idx: int = 0, beta: float = 1.0) -> float:
    precision = compute_precision(confusion_matrix, idx=idx)
    recall = compute_recall(confusion_matrix, idx=idx)
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def compute_averaged_f_score(confusion_matrix: np.ndarray, beta: float = 1.0, average: str = "macro") -> float:
    assert average in ["macro"]
    f_scores = []
    for i in range(confusion_matrix.shape[0]):
        f_scores.append(compute_f_score(confusion_matrix, idx=i, beta=beta))
    return np.array(f_scores).mean()
