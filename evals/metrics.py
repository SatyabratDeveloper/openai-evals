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

def get_each_sample_accuracy(sampled, expected) -> float:
    print("======================================================")
    test_accuracy = 0
    subject_accuracy = 0
    skill_accuracy = 0
    
    sampled_length = len(sampled)
    expected_length = len(expected)
    print(f"sampled_length: {sampled_length}, expected_length: {expected_length}")
    
    # Output
    level_1, level_2, level_3  = sampled.get('level 1', ''), sampled.get('level 2', ''), sampled.get('level 3', '')
    skill_tags = sampled.get('skills', [])

    # Expected
    expected_level_1, expected_level_2, expected_level_3 = expected.get('level 1', ''), expected.get('level 2', ''), expected.get('level 3', '')
    expected_skill_tags = expected.get('skills', [])
    
    # Subject Comparison
    if sampled_length > 3 and sampled_length == expected_length:
        print("a")
        if level_1 in expected_level_1:
            subject_accuracy += 50
        if level_2 in expected_level_2:
            subject_accuracy += 30
        if level_3 in expected_level_3:
            subject_accuracy += 20
    elif expected_length == 3:
        print("b")
        if level_1 in expected_level_1 or level_1 in expected_level_2:
            subject_accuracy += 50
        if level_2 in expected_level_1 or level_2 in expected_level_2:
            subject_accuracy += 50
        if level_3 in expected_level_1 or level_3 in expected_level_2:
            subject_accuracy += 50
    elif expected_length == 2:
        if level_1 in expected_level_1 or level_1 in expected_level_2:
            subject_accuracy += 100
    elif sampled_length <=3 and expected_length > 3:
        print("c")
        if level_1 in expected_level_1 or level_1 in expected_level_2 or level_1 in expected_level_3:
            subject_accuracy += 33
        if level_2 in expected_level_1 or level_2 in expected_level_2 or level_2 in expected_level_3:
            subject_accuracy += 33        
    
    print(f"Output Subject Tag Level 1: {level_1} === Expected Subject Tag--> Level 1: {expected_level_1}, {level_1 in expected_level_1 or level_1 in expected_level_2 or level_1 in expected_level_3}")
    print(f"Output Subject Tag Level 2: {level_2} === Expected Subject Tag--> Level 2: {expected_level_2}, {level_2 in expected_level_1 or level_2 in expected_level_2 or level_2 in expected_level_3}")
    # print(f"Output Subject Tag Level 3: {level_3} === Expected Subject Tag--> Level 3: {expected_level_3}, {level_3 in expected_level_3}")
    
    # Skill Tags Comparison
    correct_tags = len(set(skill_tags) & set(expected_skill_tags))
    print("Output Skill Tags -->", skill_tags)
    print("Expected Skill Tags -->", expected_skill_tags)
    print("Correct Skill Tags", correct_tags)

    if correct_tags == 0:
        skill_accuracy += 0
    elif correct_tags == 1 and len(expected_skill_tags) == 2:
        skill_accuracy += 50
    elif correct_tags == 1 and len(expected_skill_tags) == 3:
        skill_accuracy += 33
    elif correct_tags == 2 and len(expected_skill_tags) == 3:
        skill_accuracy += 66
    elif correct_tags == len(expected_skill_tags):
        skill_accuracy += 100
    
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
