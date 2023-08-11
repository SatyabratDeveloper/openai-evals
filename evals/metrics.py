"""
This file defines various common metrics of interest.
"""
import random
from typing import Optional, Sequence, Set

import numpy as np
import logging

from evals.record import Event


def get_accuracy(events: Sequence[Event]) -> float:
    num_correct = sum(int(event.data["correct"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total
    
def get_subject_tags_accuracy(subject_tags, expected_subject_tags) -> float:
    subject_accuracy = 0
    inverted_value = 0
    matched_level_list = []

    output_list = list(subject_tags.values())
    expected_list = list(expected_subject_tags.values())

    output_arr_length = len(output_list)
    expected_arr_length = len(expected_list)

    for expected_index, expected_value in enumerate(expected_list):
        for output_index, output_value in enumerate(output_list):
            if expected_index == 0 or expected_index == 1:
                if expected_value[0] == output_value:
                    matched_level_list.append(output_index)
            else:
                if output_value in expected_value:
                    matched_level_list.append(output_index)

    for element in matched_level_list:
        if element >= inverted_value:
            inverted_value = element
        else:
            inverted_value = 'Inverted Output'
            break

    logging.info(f"inverted_value:{inverted_value}")

    # Subject Comparison
    if inverted_value != 'Inverted Output':
        for expected_index, expected_value in enumerate(expected_list):
            for output_index, output_value in enumerate(output_list):
                if expected_index == 0:
                    if expected_value[0] == output_value:
                        print('a')
                        subject_accuracy += 50 if expected_arr_length == 3 and expected_arr_length == output_arr_length else (50 if expected_arr_length == 2 else (100 if expected_arr_length == 1 else 33))
                elif expected_index == 1:
                    if expected_value[0] == output_value:
                        print('b')
                        subject_accuracy += 30 if expected_arr_length == 3 and expected_arr_length == output_arr_length else (50 if expected_arr_length == 2 else 33)
                else:
                    if output_value in expected_value:
                        print('c')
                        subject_accuracy += 20 if expected_arr_length == 3 and expected_arr_length else 33
    
    # print(f"Subject Tag Level 1: {level_1} === Expected Subject Tag--> Level 1: {expected_level_1}, {level_1 in expected_level_1 or level_1 in expected_level_2 or level_1 in expected_level_3}")
    # print(f"Subject Tag Level 2: {level_2} === Expected Subject Tag--> Level 2: {expected_level_2}, {level_2 in expected_level_1 or level_2 in expected_level_2 or level_2 in expected_level_3}")
    # print(f"Subject Tag Level 3: {level_3} === Expected Subject Tag--> Level 3: {expected_level_3}, {level_3 in expected_level_1 or level_2 in expected_level_2 or level_2 in expected_level_3}")
    
    return subject_accuracy

def get_skill_tags_accuracy(skill_tags, expected_skill_tags) -> float:
    skill_accuracy = 0

    expected_skill_tags_length = len(expected_skill_tags)
    correct_tags = len(set(skill_tags) & set(expected_skill_tags))

    # Skill Tags Comparison
    print("Output Skill Tags -->", skill_tags)
    print("Expected Skill Tags -->", expected_skill_tags)
    print("Correct Skill Tags", correct_tags)

    if correct_tags == 0:
        skill_accuracy += 0
    elif correct_tags == 1 and expected_skill_tags_length == 2:
        skill_accuracy += 50
    elif correct_tags == 1 and expected_skill_tags_length == 3:
        skill_accuracy += 33
    elif correct_tags == 2 and expected_skill_tags_length == 3:
        skill_accuracy += 66
    elif correct_tags == expected_skill_tags_length:
        skill_accuracy += 100

    return skill_accuracy

def get_each_sample_accuracy(sampled, expected) -> float:
    print("======================================================")
    test_accuracy = 0
    subject_tags = {}
    skill_tags = []
    expected_subject_tags = {}
    expected_skill_tags = []

    for key, value in sampled.items():
        if key.startswith('level'):
            subject_tags[key] = value
        elif key == 'skills':
            skill_tags = value

    for key, value in expected.items():
        if key.startswith('level'):
            expected_subject_tags[key] = value
        elif key == 'skills':
            expected_skill_tags = value
    
    subject_accuracy = get_subject_tags_accuracy(subject_tags, expected_subject_tags)
    skill_accuracy = get_skill_tags_accuracy(skill_tags, expected_skill_tags)

    test_accuracy = (subject_accuracy + skill_accuracy) / 2
    test_pass = test_accuracy >= 75
    
    # Logs
    print("Subject Tags Accuracy", subject_accuracy)
    print("Skill Tags Accuracy", skill_accuracy)
    print("Test Accuracy", test_accuracy)
    print("======================================================")
    
    return {
        'pass': test_pass,
        'score': test_accuracy / 100,
        'reason': 'output matched' if test_pass else 'Output did not matched',
    }

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
