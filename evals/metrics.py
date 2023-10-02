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

# Open the JSON file for reading
with open('subjectTags.json', 'r', encoding='utf-8') as file:
    subject_tag_tree_data = json.load(file)

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
    

def get_subject_tag_dict(response):
    # Regex to find response JSON
    pattern = r'{[^}]+}'

    # To extract all matches
    matches = re.findall(pattern, response)

    # Convert the JSON string to a Python dictionary
    response_dict = json.loads(matches[0])

    return response_dict

def get_subject_and_skill_tags(response_dict, expected_dict):
    subject_tags = {}

    # Loop to split response subject and skill tags
    for key, value in response_dict.items():
        if key.startswith('level'):
            subject_tags[key] = value
        
    return subject_tags



def get_similarity(str1, str2):
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

def get_subject_tags_id(level_1, level_2, level_3):
    level_1_name = ''
    level_1_id = ''
    level_2_ids = []
    level_3_ids = []
    level_2_similar_ids = []
    level_3_similar_ids = []

    for key, value in subject_tag_tree_data.items():
        id = value['id']
        name = value['name'].lower()
        height = value['height']
        if level_1 == name and height == 0:
            level_1_name = name
            level_1_id = id
        elif level_2 == name:
            level_2_ids.append(id)
        elif level_3 == name:
            level_3_ids.append(id)
        elif level_2 and get_similarity(level_2, name) > 90:
            print("similarity_level_2*****", level_2, name, get_similarity(level_2, name))
            level_2_similar_ids.append(id)
        elif level_3 and get_similarity(level_3, name) > 90:
            print("similarity_level_3*****", level_2, name, get_similarity(level_2, name))
            level_3_similar_ids.append(id)

    return level_1_id, level_2_ids, level_3_ids, level_2_similar_ids, level_3_similar_ids

"""
    To get related nodes
    Comparing nodes(levels) with there ancestors to get its parent
    returning the max length tree
"""
def get_related_nodes(level_1_id, level_2_ids, level_3_ids):
    tree_ids = []
    # Compare levels with there ancestors to get its parent(related nodes)
    if level_1_id:
        if level_2_ids:
            for level_2_id in level_2_ids:
                if level_3_ids:
                    for level_3_id in level_3_ids:
                        if 'ancestor' in subject_tag_tree_data.get(level_2_id, {}):
                            if 'ancestor' in subject_tag_tree_data.get(level_3_id, {}):
                                if level_1_id in subject_tag_tree_data[level_2_id]['ancestor']:
                                    if level_1_id in subject_tag_tree_data[level_3_id]['ancestor']:
                                        if level_2_id in subject_tag_tree_data[level_3_id]['ancestor']:
                                            tree_ids.append([level_1_id, level_2_id, level_3_id])
                                        else:
                                            tree_ids.append([level_1_id, level_3_id])
                                    else:
                                        tree_ids.append([level_1_id, level_2_id])
                                elif level_1_id in subject_tag_tree_data[level_3_id]['ancestor']:
                                    tree_ids.append([level_1_id, level_3_id])
                                else:
                                    tree_ids.append([level_1_id])
                            elif level_1_id in subject_tag_tree_data[level_2_id]['ancestor']:
                                tree_ids.append([level_1_id, level_2_id])
                            else:
                                tree_ids.append([level_1_id])
                        elif 'ancestor' in subject_tag_tree_data.get(level_3_id, {}):
                            if level_1_id in subject_tag_tree_data[level_3_id]['ancestor']:
                                tree_ids.append([level_1_id, level_3_id])
                            else:
                                tree_ids.append([level_1_id])
                elif 'ancestor' in subject_tag_tree_data.get(level_2_id, {}):
                    if level_1_id in subject_tag_tree_data[level_2_id]['ancestor']:
                        tree_ids.append([level_1_id, level_2_id])
                    else:
                        tree_ids.append([level_1_id])
                else:
                    tree_ids.append([level_1_id])
        elif level_3_ids:
            for level_3_id in level_3_ids:
                if 'ancestor' in subject_tag_tree_data.get(level_3_id, {}):
                    if level_1_id in subject_tag_tree_data[level_3_id]['ancestor']:
                        tree_ids.append([level_1_id, level_3_id])
                    else:
                        tree_ids.append([level_1_id])
                else:
                    tree_ids.append([level_1_id])
        else:
            tree_ids.append([level_1_id])
    if level_2_ids and level_3_ids:
        for level_2_id in level_2_ids:
            for level_3_id in level_3_ids:
                if 'ancestor' in subject_tag_tree_data.get(level_3_id, {}):
                    if level_2_id in subject_tag_tree_data[level_3_id]['ancestor']:
                        tree_ids.append([level_2_id, level_3_id])

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

def get_node_list(max_length_trees):
    final_trees = []
    for max_length_tree in max_length_trees:
        node_info = {}
        node_list = {}
        last_tree_node = max_length_tree[-1]
        last_node = subject_tag_tree_data[last_tree_node]
        for key, value in subject_tag_tree_data.items():
            if 'ancestor' in value and last_tree_node in value['ancestor'] and not value['children']:
                node_list[key] = value
        node_info['node_list'] = node_list
        node_info['last_node'] = last_node
        final_trees.append(node_info)

    return final_trees

def get_nested_object(final_tree):
    node_list = final_tree['node_list']
    last_node = final_tree['last_node']
    ancestors_list = []
    
    # To make the tree structure
    for key, value in node_list.items():
        ancestor_list = []
        for item in value['ancestor']:
            name = subject_tag_tree_data[item]['name']
            height = subject_tag_tree_data[item]['height']
            if height >= last_node['height']:
                ancestor_list.append({'name': name, 'height': height})
        ancestor_list.append({'name': value['name'], 'height': value['height']})
        ancestors_list.append(ancestor_list)

    with open("ancestors_list.json", "a") as json_file:
        json.dump(ancestors_list, json_file)
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

def get_nested_object_tree(level_1_id, level_2_ids, level_3_ids):
    nested_object_list = []

    # To get the related parent and child
    max_length_trees = get_related_nodes(level_1_id, level_2_ids, level_3_ids)
    print("max_length_trees", max_length_trees)

    # To get list of all posible node list (ancestor nodes to make a tree)
    final_trees = get_node_list(max_length_trees)
    
    # with open("node_list.json", "a") as json_file:
    #     json.dump(final_trees, json_file)

    # To get nested object (Tag Tree)
    for final_tree in final_trees:
        # print(final_tree["last_node"]["name"])
        nested_object = get_nested_object(final_tree)
        nested_object_list.append(nested_object)

    return nested_object_list, max_length_trees

def get_similarity_check(correct_level, parent_id, child_1_ids, child_1_ids_similar, child_2_ids_similar):
    print("**************similarity******************")
    nested_object = {}
    if 'level_2' not in correct_level and 'level_3' not in correct_level:
        nested_object = get_nested_object_tree(parent_id, child_1_ids_similar, child_2_ids_similar)
    elif 'level_3' not in correct_level:
        nested_object = get_nested_object_tree(parent_id, child_1_ids, child_2_ids_similar)
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
    
    if len(json.dumps(trimed_tree)) > 4000:
        get_final_trimed_tag_tree(trimed_tree)

    # with open("trimed_tag_tree.json", "a") as json_file:
    #     json.dump(trimed_tag_tree, json_file)

    return trimed_tag_tree

"""
    To get tag tree for Prompt 2 from Prompt 1 response
"""
def get_tag_tree(response):
    # To convert response string into dictionary
    subject_tag_dict = get_subject_tag_dict(response)

    # Levels of Subject Tags
    level_1, level_2, level_3  = subject_tag_dict['Level 1'].lower() if subject_tag_dict['Level 1'] else [], subject_tag_dict['Level 2'].lower() if subject_tag_dict['Level 2'] else [], subject_tag_dict['Level 3'].lower() if subject_tag_dict['Level 3'] else []

    # To get ids of parent and childs
    level_1_id, level_2_ids, level_3_ids, level_2_similar_ids, level_3_similar_ids = get_subject_tags_id(level_1, level_2, level_3)

    print("==================================================")
    print("level_1_id", level_1_id)
    print("level_2_ids", level_2_ids)
    print("level_3_ids", level_3_ids)
    print("level_2_similar_ids", level_2_similar_ids)
    print("level_3_similar_ids", level_3_similar_ids)
    print("==================================================")

    # nested_object_list - tag tree list
    # max_length_trees - To get correct levels
    nested_object_list, max_length_trees = get_nested_object_tree(level_1_id, level_2_ids, level_3_ids)

    correct_levels = []
    # To get correct levels
    for max_length_tree in max_length_trees:
        correct_level = []
        for node in max_length_tree:
            node_name = (subject_tag_tree_data[node]['name']).lower()
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

    if nested_object_list_size > 4000:
        for nested_object, correct_level in zip(nested_object_list, correct_levels):
            nested_object_similarity_list = get_similarity_check(correct_level, level_1_id, level_2_ids, level_2_similar_ids, level_3_similar_ids)
            trees_list.append(nested_object_similarity_list)
    else:
        trees_list = nested_object_list

    if len(json.dumps(trees_list)) > 4000:
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
