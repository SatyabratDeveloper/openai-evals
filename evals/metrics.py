"""
This file defines various common metrics of interest.
"""
import random
from typing import Optional, Sequence, Set
import numpy as np
import math
import re
import json
import sys
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

sorted_ancestors_list = []

def get_accuracy(events: Sequence[Event]) -> float:
    num_correct = sum(int(event.data["correct"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total

def get_size(obj, seen=None):
    """Recursively finds the size of a nested object."""
    if seen is None:
        seen = set()
    size = sys.getsizeof(obj)
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, dict):
        size += sum(get_size(key, seen) + get_size(value, seen) for key, value in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(item, seen) for item in obj)
    return size
    

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

def get_parent_child_ids(level_1, level_2, level_3, data):
    parent = ''
    parent_id = ''
    child_1_ids = []
    child_2_ids = []
    child_1_ids_similar = []
    child_2_ids_similar = []

    for key, value in data.items():
        id = value.get('id')
        name = value.get('name').lower()
        height = value.get('height')
        if level_1 == name and height == 0:
            parent = name
            parent_id = id
        if level_2 == name:
            child_1_ids.append(id)
        elif level_2 and similarity(level_2, name) > 75:
            child_1_ids_similar.append(id)
        if level_3 == name:
            child_2_ids.append(id)
        elif level_3 and similarity(level_3, name) > 75:
            child_2_ids_similar.append(id)

    return parent_id, child_1_ids, child_2_ids, child_1_ids_similar, child_2_ids_similar

def get_related_nodes(parent_id, child_1_ids, child_2_ids, data):
    tree_ids = []

    if parent_id != '':
        if len(child_1_ids) > 0:
            for child_1_id in child_1_ids:
                if len(child_2_ids) > 0:
                    for child_2_id in child_2_ids:
                        if 'ancestor' in data[child_1_id] and parent_id in data[child_1_id]['ancestor']:
                            if 'ancestor' in data[child_2_id] and parent_id in data[child_2_id]['ancestor']:
                                if 'ancestor' in data[child_2_id] and child_1_id in data[child_2_id]['ancestor']:
                                    tree_ids.append([parent_id, child_1_id, child_2_id])
                                elif 'ancestor' in data[child_1_id] and child_2_id in data[child_1_id]['ancestor']:
                                    return "Response is inverted"
                                else:
                                    tree_ids.append([parent_id, child_1_id])
                                    tree_ids.append([parent_id, child_2_id])
                            else:
                                tree_ids.append([parent_id, child_1_id])
                        elif 'ancestor' in data[child_2_id] and parent_id in data[child_2_id]['ancestor']:
                            tree_ids.append([parent_id, child_2_id])
                        else:
                            tree_ids.append([parent_id])
                elif 'ancestor' in data[child_1_id] and parent_id in data[child_1_id]['ancestor']:
                    tree_ids.append([parent_id, child_1_id])
                else:
                    tree_ids.append([child_1_id])
        elif len(child_2_ids) > 0:
            for child_2_id in child_2_ids:
                if 'ancestor' in data[child_2_id] and parent_id in data[child_2_id]['ancestor']:
                    tree_ids.append([parent_id, child_2_id])
                else:
                    tree_ids.append([child_2_id])
        else:
            tree_ids.append([parent_id])
    else:
        if len(child_1_ids) > 0:
            for child_1_id in child_1_ids:
                if len(child_2_ids) > 0:
                    for child_2_id in child_2_ids:
                        if 'ancestor' in data[child_2_id] and child_1_id in data[child_2_id]['ancestor']:
                            tree_ids.append([child_1_id, child_2_id])
                        else:
                            tree_ids.append([child_1_id])
                            tree_ids.append([child_2_id])
                else:
                    tree_ids.append([child_1_id])
        elif len(child_2_ids) > 0:
            for child_2_id in child_2_ids:
                    tree_ids.append([child_2_id])

    unique_tree_ids = []
    # To remove the duplicate lists from tree_ids
    for item in tree_ids:
        if item not in unique_tree_ids:
            unique_tree_ids.append(item)

    # Find the maximum length among the sublists
    max_length = max(len(sublist) for sublist in unique_tree_ids) if len(unique_tree_ids) > 0 else []

    # Collect all sublists with the maximum length
    max_length_tree = [sublist for sublist in unique_tree_ids if len(sublist) == max_length]

    return max_length_tree

def get_node_list(max_length_trees, data):
    final_trees = []
    for max_length_tree in max_length_trees:
        node_info = {}
        node_list = {}
        last_tree_node = max_length_tree[-1]
        last_node = data[last_tree_node]
        for key, value in data.items():
            if 'ancestor' in value and last_tree_node in value['ancestor'] and not value['children']:
                node_list[key] = value
        node_info['node_list'] = node_list
        node_info['last_node'] = last_node
        final_trees.append(node_info)

    return final_trees

def get_nested_object(final_tree, data):
    node_list = final_tree['node_list']
    last_node = final_tree['last_node']
    ancestors_list = []
    
    # To make the tree structure
    for key, value in node_list.items():
        ancestor_list = []
        for item in value['ancestor']:
            name = data[item]['name']
            height = data[item]['height']
            if height >= last_node['height']:
                ancestor_list.append({'name': name, 'height': height})
        ancestor_list.append({'name': value['name'], 'height': value['height']})
        ancestors_list.append(ancestor_list)

    # with open("ancestors_list.json", "a") as json_file:
    #     json.dump(ancestors_list, json_file)
    global sorted_ancestors_list
    sorted_ancestors_list = [sorted(array, key=lambda x: x['height']) for array in ancestors_list]

    nested_object = {}

    for item in sorted_ancestors_list:
        current_level = nested_object  # Start at the root level
    
        for entry in item:
            name, height = entry["name"], entry["height"]
    
            if name not in current_level:
                current_level[name] = {}  # Create a new level if it doesn't exist
    
            current_level = current_level[name]  # Move to the next level

    # with open("sorted_ancestors_list.json", "a") as json_file:
    #     json.dump(sorted_ancestors_list, json_file)
    
    return nested_object

def get_nested_object_tree(parent_id, child_1_ids, child_2_ids, data):
    nested_object_list = []

    # To get the related parent and child
    max_length_trees = get_related_nodes(parent_id, child_1_ids, child_2_ids, data)
    print("max_length_trees", max_length_trees)

    # To get list of all posible node list (ancestor nodes to make a tree)
    final_trees = get_node_list(max_length_trees, data)

    # print(final_trees)
    
    # with open("node_list.json", "a") as json_file:
    #     json.dump(final_trees, json_file)

    # To get nested object (Tag Tree)
    for final_tree in final_trees:
        print(final_tree["last_node"]["name"])
        nested_object = get_nested_object(final_tree, data)
        nested_object_list.append(nested_object)

    return nested_object_list, max_length_trees

def get_similarity_check(correct_level, parent_id, child_1_ids, child_1_ids_similar, child_2_ids_similar, data):
    print("**************similarity******************")
    nested_object = {}
    if 'level_2' not in correct_level and 'level_3' not in correct_level:
        nested_object = get_nested_object_tree(parent_id, child_1_ids_similar, child_2_ids_similar, data)
    elif 'level_3' not in correct_level:
        nested_object = get_nested_object_tree(parent_id, child_1_ids, child_2_ids_similar, data)
    return nested_object

def get_trimed_tag_tree(sorted_ancestors_list):
    for ancestor_list in sorted_ancestors_list:
        if len(ancestor_list) > 1:
            ancestor_list.pop()

    nested_object = {}

    for item in sorted_ancestors_list:
        current_level = nested_object  # Start at the root level
    
        for entry in item:
            name, height = entry["name"], entry["height"]
    
            if name not in current_level:
                current_level[name] = {}  # Create a new level if it doesn't exist
    
            current_level = current_level[name]  # Move to the next level
            
    return nested_object

def get_final_trimed_tag_tree(trees_list):
    global sorted_ancestors_list
    trimed_tag_tree = []

    trimed_tree = get_trimed_tag_tree(sorted_ancestors_list)
    trimed_tag_tree.append(trimed_tree)
    
    if len(json.dumps(trimed_tree)) > 3000:
        get_final_trimed_tag_tree(trimed_tree)

    # with open("trimed_tag_tree.json", "a") as json_file:
    #     json.dump(trimed_tag_tree, json_file)

    return trimed_tag_tree

def get_tag_tree(response):
    # To get response in lowercase
    subject_tags = get_lowercase_dictionary(response)

    # Levels of Subject Tags
    level_1, level_2, level_3  = subject_tags.get('level 1', ''), subject_tags.get('level 2', ''), subject_tags.get('level 3', '')

    # Open the JSON file for reading
    with open('subjectTags.json', 'r') as file:
        data = json.load(file)

    # get ids of parent and childs
    parent_id, child_1_ids, child_2_ids, child_1_ids_similar, child_2_ids_similar = get_parent_child_ids(level_1, level_2, level_3, data)

    print("parent_id", parent_id)
    print("child_1_ids", child_1_ids)
    print("child_2_ids", child_2_ids)
    print("child_1_ids_similar", child_1_ids_similar)
    print("child_2_ids_similar", child_2_ids_similar)

    # nested_object_list - tag tree
    # max_length_trees - To get correct levels
    nested_object_list, max_length_trees = get_nested_object_tree(parent_id, child_1_ids, child_2_ids, data)

    correct_levels = []
    # To get correct levels
    for max_length_tree in max_length_trees:
        correct_level = []
        for node in max_length_tree:
            node_name = (data[node]['name']).lower()
            if node_name == level_1:
                correct_level.append("level_1")
            if node_name == level_2:
                correct_level.append("level_2")
            if node_name == level_3:
                correct_level.append("level_3")
        correct_levels.append(correct_level)

    trees_list = []

    nested_object_list_size = len(json.dumps(nested_object_list))
    # print(nested_object_list_size)

    if nested_object_list_size > 3000:
        for nested_object, correct_level in zip(nested_object_list, correct_levels):
            nested_object_similarity_list = get_similarity_check(correct_level, parent_id, child_1_ids, child_1_ids_similar, child_2_ids_similar, data)
            trees_list.append(nested_object_similarity_list)
    else:
        trees_list = nested_object_list

    if len(json.dumps(trees_list)) > 3000:
        final_trimed_tag_tree = get_final_trimed_tag_tree(trees_list)
    else:
        final_trimed_tag_tree = trees_list
    # print(trees_list)

    final_tag_tree = []

    for tag_tree in final_trimed_tag_tree:
        if len(json.dumps(tag_tree)) > 2:
            final_tag_tree.append(tag_tree)

    return final_tag_tree


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
