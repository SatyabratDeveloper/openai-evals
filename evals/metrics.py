"""
This file defines various common metrics of interest.
"""
import random
from typing import Optional, Sequence, Set
import numpy as np
import math
import re
import json
from evals.record import Event

# Import the ModelEmbeddings class from the misc module
from string2string.misc import ModelEmbeddings

# Import the CosineSimilarity class from the similarity module
from string2string.similarity import CosineSimilarity

# Create an instance of the ModelEmbeddings class (if device is not specified, it will be automatically detected)
bart_model = ModelEmbeddings(
    model_name_or_path='facebook/bart-large'
)

# Create an instance of the CosineSimilarity class
cosine_similarity = CosineSimilarity()


def get_accuracy(events: Sequence[Event]) -> float:
    num_correct = sum(int(event.data["correct"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total
    
# def get_subject_tags_accuracy(subject_tags, expected_subject_tags, user_input) -> float:
#     subject_accuracy = 0
#     inverted_value = 0
#     matched_level_list = []
#     level_1 = ''
#     level_2 = ''
#     level_3 = ''
#     expected_level_1 = ''
#     expected_level_2 = ''
#     expected_level_3 = []

#     correct_level_1 = False

#     output_list = list(subject_tags.values())
#     expected_list = list(expected_subject_tags.values())

#     # output_arr_length = len(output_list)
#     expected_arr_length = len(expected_list)

#     # loop to find inverted tags and assigning the levels values
#     for expected_index, expected_value in enumerate(expected_list):
#         if expected_index == 0:
#             expected_level_1 = expected_value[0]
#         if expected_index == 1:
#             expected_level_2 = expected_value[0]
#         if expected_index == 2:
#             expected_level_3 = expected_value
        
#         for output_index, output_value in enumerate(output_list):
#             if output_index == 0:
#                 level_1 = output_value
#             if output_index == 1:
#                 level_2 = output_value
#             if output_index == 2:
#                 level_3 = output_value
            
#             if expected_index == 0 or expected_index == 1:
#                 if expected_value[0] == output_value:
#                     matched_level_list.append(output_index)
#             if expected_index == 2:
#                 if output_value in expected_value:
#                     matched_level_list.append(output_index)

#     for element in matched_level_list:
#         if element >= inverted_value:
#             inverted_value = element
#         else:
#             inverted_value = 'Inverted Output'
#             break

#     # Subject Comparison
#     if inverted_value != 'Inverted Output':
#         if expected_level_1 == level_1:
#             correct_level_1 = True
#             subject_accuracy += 50 if expected_arr_length == 3 else (60 if expected_arr_length == 2 else 100)
#         else:
#             with open('incorrect.txt', 'a') as file:
#                 file.write(f"Question: {user_input['content']}\n\nExpected: Level 1: {expected_level_1}, Level 2: {expected_level_2}, Level 3: {expected_level_3}\nOutput: Level 1: {level_1}, Level 2: {level_2}, Level 3: {level_3}\n\n")
#         if expected_level_2 == level_2 or expected_level_2 == level_3:
#             subject_accuracy += 20 if expected_arr_length == 3 else 40
#         elif expected_level_2:
#             similarity_input = [expected_level_2, level_2, level_3]
#             if None in similarity_input:
#                 similarity_input = [item for item in similarity_input if item is not None]
#             result = similarity_checker(similarity_input)
#             if result == True:
#                 subject_accuracy += 20 if expected_arr_length == 3 else 40
#         if level_2 in expected_level_3 or level_3 in expected_level_3:
#             subject_accuracy += 30
#         elif expected_level_3:
#             similarity_input = expected_level_3 + [level_2, level_3]
#             if None in similarity_input:
#                 similarity_input = [item for item in similarity_input if item is not None]
#             result = similarity_checker(similarity_input)
#             if result == True:
#                 subject_accuracy += 30
#     else:
#         if expected_level_1 == level_1:
#             subject_accuracy += 50 if expected_arr_length == 3 else 60
#         else:
#             with open('incorrect.txt', 'a') as file:
#                 file.write(f"{user_input}\n\nExpected: Level 1: {expected_level_1}, Level 2: {expected_level_2}, Level 3: {expected_level_3}\nOutput: Level 1: {level_1}, Level 2: {level_2}, Level 3: {level_3}\n\n")
    
#     return {"subject_accuracy": subject_accuracy, "correct_level_1": correct_level_1}

# def get_skill_tags_accuracy(skill_tags, expected_skill_tags) -> float:
#     skill_accuracy = 0

#     expected_skill_tags_length = len(expected_skill_tags)
#     correct_tags = len(set(skill_tags) & set(expected_skill_tags))

#     # Skill Tags Comparison
#     skill_accuracy = (correct_tags / expected_skill_tags_length) * 100

#     return skill_accuracy

def get_lowercase_dictionary(response):
    # Convert the JSON string to a Python dictionary
    print("response", response)
    response_dict = json.loads(response)

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

    return response_lowercase_dict

def get_subject_and_skill_tags(response_dict, expected_dict):
    subject_tags = {}

    # Loop to split response subject and skill tags
    for key, value in response_dict.items():
        if key.startswith('level'):
            subject_tags[key] = value
        
    return subject_tags

# def get_each_sample_accuracy(response, expected, user_input) -> float:
#     test_accuracy = 0

#     # To get lowercase response and expected
#     response_dict, expected_dict = get_lowercase_dictionary(response, expected)

#     # To split subject and skill tags
#     subject_tags, expected_subject_tags = get_subject_and_skill_tags(response_dict, expected_dict)

#     # To get subject and skill accuracy
#     subject_accuracy = get_subject_tags_accuracy(subject_tags, expected_subject_tags, user_input)
#     # skill_accuracy = get_skill_tags_accuracy(skill_tags, expected_skill_tags)

#     # To get test accuracy and set threshold
#     # test_accuracy = (subject_accuracy["subject_accuracy"] + skill_accuracy) / 2
#     test_accuracy = subject_accuracy["subject_accuracy"]
#     test_pass = test_accuracy >= 75

#     with open('evals_output.txt', 'a') as file:
#         file.write(f"======================================================\n")
#         file.write(f"Subject tag output: {subject_tags}\n")
#         file.write(f"Subject tag expected: {expected_subject_tags}\n")
#         # file.write(f"Skill tag output: {skill_tags}\n")
#         # file.write(f"Skill tag expected: {expected_skill_tags}\n")
#         file.write(f"******************************************************\n")
#         file.write(f"Subject Tags Accuracy: {subject_accuracy['subject_accuracy']}\n")
#         # file.write(f"Skill Tags Accuracy: {skill_accuracy}\n")
#         file.write(f"Test Accuracy: {test_accuracy}\n")
#         file.write(f"Test pass: {test_accuracy}\n")
#         file.write(f"======================================================\n\n")
    
#     # Logs
#     print("======================================================")
#     print("Subject tag output", subject_tags)
#     print("Subject tag expected", expected_subject_tags)
#     # print("Skill tag output", skill_tags)
#     # print("Skill tag expected", expected_skill_tags)
#     print("******************************************************")
#     print("Subject Tags Accuracy", subject_accuracy["subject_accuracy"])
#     # print("Skill Tags Accuracy", skill_accuracy)
#     print("Test Accuracy", test_accuracy)
#     print("Test pass", test_accuracy)
#     print("======================================================")
    
#     return {
#         'pass': test_pass,
#         'subject_tags_score': subject_accuracy["subject_accuracy"],
#         # 'skill_tags_score': skill_accuracy,
#         'combined_tag_score': test_accuracy,
#         'correct_level_1': subject_accuracy["correct_level_1"],
#         'reason': 'output matched' if test_pass else 'Output did not matched',
#     }


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



def get_tag_tree(subject_tags):
    # Output
    level_1, level_2, level_3  = subject_tags.get('level 1', ''), subject_tags.get('level 2', ''), subject_tags.get('level 3', '')
    # print("level 1", level_1, "level 2", level_2, "level 3", level_3)

    parent = ''
    parent_id = ''
    child_1 = []
    child_1_ids = []
    child_2 = []
    child_2_ids = []
    filtered_child_1_ids = []
    filtered_child_2_ids = []
    final_tree_id = []

    # Open the JSON file for reading
    with open('subjectTags.json', 'r') as file:
        # Parse the JSON data into a Python dictionary
        data = json.load(file)

    for key, value in data.items():
        id = value.get('id')
        name = value.get('name').lower()
        height = value.get('height')
        if level_1 == name and height == 0:
            parent = name
            parent_id = id
        if level_2 == name:
            child_1_ids.append(id)
        if level_3 == name:
            child_2_ids.append(id)

    print("parent_id", parent_id)
    print("child_1_ids", child_1_ids)
    print("child_2_ids", child_2_ids)
    
    tree_ids = []

    if parent_id != '':
        if len(child_1_ids) > 0:
            for child_1_id in child_1_ids:
                if len(child_2_ids) > 0:
                    for child_2_id in child_2_ids:
                        if parent_id in data[child_1_id]['ancestor']:
                            if parent_id in data[child_2_id]['ancestor']:
                                if child_1_id in data[child_2_id]['ancestor']:
                                    tree_ids.append([parent_id, child_1_id, child_2_id])
                                else:
                                    tree_ids.append([parent_id, child_1_id])
                                    tree_ids.append([parent_id, child_2_id])
                            else:
                                tree_ids.append([parent_id, child_1_id])
                        elif parent_id in data[child_2_id]['ancestor']:
                            tree_ids.append([parent_id, child_2_id])
                        else:
                            tree_ids.append([parent_id])
                elif parent_id in data[child_1_id]['ancestor']:
                    tree_ids.append([parent_id, child_1_id])
                else:
                    tree_ids.append([child_1_id])
        elif len(child_2_ids) > 0:
            for child_2_id in child_2_ids:
                if parent_id in data[child_2_id]['ancestor']:
                    tree_ids.append([parent_id, child_2_id])
        else:
            tree_ids.append([parent_id])
    else:
        if len(child_1_ids) > 0:
            for child_1_id in child_1_ids:
                if len(child_2_ids) > 0:
                    for child_2_id in child_2_ids:
                        if child_1_id in data[child_2_id]['ancestor']:
                            tree_ids.append([child_1_id, child_2_id])
                tree_ids.append([child_1_id])
        elif len(child_2_ids) > 0:
            for child_2_id in child_2_ids:
                    tree_ids.append([child_2_id])


    print("tree_ids", tree_ids)

    unique_tree_ids = []
    # To remove the duplicate lists from tree_ids
    for item in tree_ids:
        if item not in unique_tree_ids:
            unique_tree_ids.append(item)
    
    print("unique_tree_ids", unique_tree_ids)

    # Find the maximum length among the sublists
    max_length = max(len(sublist) for sublist in unique_tree_ids) if len(unique_tree_ids) > 0 else []

    # Collect all sublists with the maximum length
    max_length_tree = [sublist for sublist in unique_tree_ids if len(sublist) == max_length]    
        
    print("max_length_tree", max_length_tree)

    node_list = {}
    last_node = {}
    for max_tree in max_length_tree:
        tree_node = max_tree[len(max_tree) - 1]
        last_node = data[tree_node]
        for key, value in data.items():
            if 'ancestor' in value:
                if tree_node in value['ancestor'] and value['children'] == []:
                    node_list[key] = value

    print("node_list", node_list)

    with open("node_list.json", "a") as json_file:
        json.dump(node_list, json_file)


    ancestors_list = []

    for key, value in node_list.items():
        print("---------------", value)
        ancestor_list = []
        for item in value['ancestor']:
            name = data[item]['name']
            height = data[item]['height']
            if height >= last_node['height']:
                ancestor_list.append({'name': name, 'height': height})
        ancestor_list.append({'name': value['name'], 'height': value['height']})
        ancestors_list.append(ancestor_list)

    sorted_ancestors_list = [sorted(array, key=lambda x: x['height']) for array in ancestors_list]

    print("sorted_ancestors_list", sorted_ancestors_list)

    with open("sorted_ancestors_list.json", "a") as json_file:
        json.dump(sorted_ancestors_list, json_file)

    nested_object = {}

    for item in sorted_ancestors_list:
        current_level = nested_object  # Start at the root level
    
        for entry in item:
            name, height = entry["name"], entry["height"]
    
            if name not in current_level:
                current_level[name] = {}  # Create a new level if it doesn't exist
    
            current_level = current_level[name]  # Move to the next level
    
    print(json.dumps(nested_object, indent=2))
    with open("nested_object.json", "a") as json_file:
        json.dump(nested_object, json_file)

    return nested_object


def get_prompt_tree(response):
    # To get lowercase response and expected
    response_dict = get_lowercase_dictionary(response)

    tag_tree = get_tag_tree(response_dict)


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
