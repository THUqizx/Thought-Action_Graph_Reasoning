import json
import os
from utils import load_eval_data, cal_f1, cal_partial, process_multiple_answers


def load_no_answer_ids(no_answer_path: str) -> set:
    """
    Load question IDs that have no valid answers from a JSON file.
    
    Args:
        no_answer_path: Path to the JSON file containing no-answer question IDs.
        
    Returns:
        Set of question IDs with no answers.
    """
    with open(no_answer_path, "r", encoding="utf-8") as f:
        no_answer = json.load(f)
    return set(no_answer)


def process_grailqa_ground_truth(ground_truth_answers: list) -> list:
    """
    Process GrailQA format ground truth answers.
    
    Args:
        ground_truth_answers: List of ground truth answer dictionaries with "answer_type" and
                             either "entity_name" or "answer_argument" keys.
                             
    Returns:
        List of processed answer strings.
        
    Raises:
        ValueError: If an unsupported answer_type is encountered.
    """
    processed_answers = []

    for answer in ground_truth_answers:
        if answer["answer_type"] == "Entity":
            processed_answers.append(answer["entity_name"])
        elif answer["answer_type"] == "Value":
            processed_answers.append(answer["answer_argument"])
        else:
            raise ValueError(f"Unsupported answer_type: {answer['answer_type']}")

    return processed_answers


def is_answer_in_ground_truth(answer: str, ground_truth_answers: list) -> bool:
    """
    Check if an answer is present in the ground truth answers.
    
    Args:
        answer: The answer string to check.
        ground_truth_answers: List of ground truth answer strings.
        
    Returns:
        True if the answer is found in ground truth, False otherwise.
    """
    if answer in ground_truth_answers:
        return True

    for ground_truth_answer in ground_truth_answers:
        if answer in ground_truth_answer:
            return True

    return False


def cal_hits1(answers: list, ground_truth_answers: list) -> int:
    """
    Calculate Hits@1 metric for a set of answers (GrailQA format).
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer dictionaries.
        
    Returns:
        1 if the first answer matches any ground truth, 0 otherwise.
    """
    if len(answers) == 0:
        return 0

    processed_gt = process_grailqa_ground_truth(ground_truth_answers)

    if is_answer_in_ground_truth(answers[0], processed_gt):
        return 1

    return 0


def cal_f1(answers: list, ground_truth_answers: list) -> dict:
    """
    Calculate F1 metrics (precision, recall, F1 score) for a set of answers (GrailQA format).
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer dictionaries.
        
    Returns:
        Dictionary containing precision, recall, and F1 score.
    """
    processed_gt = process_grailqa_ground_truth(ground_truth_answers)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for answer in answers:
        if is_answer_in_ground_truth(answer, processed_gt):
            true_positives += 1
        else:
            false_positives += 1

    for ground_truth in processed_gt:
        matched = False
        for answer in answers:
            if is_answer_in_ground_truth(answer, [ground_truth]):
                matched = True
                break
        if not matched:
            false_negatives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def cal_partial(answers: list, ground_truth_answers: list) -> int:
    """
    Calculate partial match score for answers (GrailQA format).
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer dictionaries.
        
    Returns:
        1 if any partial match is found, 0 otherwise.
    """
    processed_gt = process_grailqa_ground_truth(ground_truth_answers)

    def _answer_in_ground_truth(answer: str, gt_list: list) -> bool:
        for ground_truth_answer in gt_list:
            if answer.lower() in ground_truth_answer.lower():
                return True
            elif ground_truth_answer.lower() in answer.lower():
                return True
        return False

    for answer in answers:
        if _answer_in_ground_truth(answer, processed_gt):
            return 1

    for ground_truth_answer in processed_gt:
        if _answer_in_ground_truth(ground_truth_answer, answers):
            return 1

    return 0


def evaluate_grailqa(
    eval_data_path: str,
    no_answer_ids: set,
    data_type: str = "grailqa"
) -> list:
    """
    Evaluate GrailQA predictions using MAC1, MAC2, and MAC3 metrics.
    
    Args:
        eval_data_path: Path to the evaluation data JSON file.
        no_answer_ids: Set of question IDs to skip (no valid answers).
        data_type: Type of dataset (default: "grailqa").
        
    Returns:
        List of evaluation results for MAC1, MAC2, and MAC3.
    """
    eval_data = load_eval_data(eval_data_path, data_type)

    mac_1 = {"hits1": 0, "precision": 0, "recall": 0, "f1": 0, "partial": 0}
    mac_2 = {"hits1": 0, "precision": 0, "recall": 0, "f1": 0, "partial": 0}
    mac_3 = {"hits1": 0, "precision": 0, "recall": 0, "f1": 0, "partial": 0}

    no_answer_num = 0

    for data in eval_data:
        if data["QuestionId"] in no_answer_ids:
            no_answer_num += 1
            continue

        if data["Answers"] != []:
            ground_truth_answer = data["Ground_Truth_Answers"]
            multi_answer = data["Answers"]

            hits1_mac1, hits1_mac2, hits1_mac3 = cal_multi_answer_hits1(ground_truth_answer, multi_answer)
            f1_1, f1_2, f1_3 = cal_multi_answer_f1(ground_truth_answer, multi_answer)
            partial_1, partial_2, partial_3 = cal_multi_answer_partial(ground_truth_answer, multi_answer)

            mac_1["hits1"] += hits1_mac1
            mac_1["precision"] += f1_1["precision"]
            mac_1["recall"] += f1_1["recall"]
            mac_1["f1"] += f1_1["f1"]
            mac_1["partial"] += partial_1

            mac_2["hits1"] += hits1_mac2
            mac_2["precision"] += f1_2["precision"]
            mac_2["recall"] += f1_2["recall"]
            mac_2["f1"] += f1_2["f1"]
            mac_2["partial"] += partial_2

            mac_3["hits1"] += hits1_mac3
            mac_3["precision"] += f1_3["precision"]
            mac_3["recall"] += f1_3["recall"]
            mac_3["f1"] += f1_3["f1"]
            mac_3["partial"] += partial_3

    total_count = len(eval_data) - no_answer_num
    result = []
    for mac in [mac_1, mac_2, mac_3]:
        result.append({k: v / total_count for k, v in mac.items()})

    return result


def main(
    eval_data_path: str,
    result_path: str,
    no_answer_path: str = None
):
    """
    Main function to evaluate GrailQA predictions and save results.
    
    Args:
        eval_data_path: Path to the evaluation data JSON file.
        result_path: Path to save the evaluation results.
        no_answer_path: Optional path to no-answer JSON file. If None,
                       uses default path from config.
    """
    if no_answer_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        no_answer_path = os.path.join(base_dir, "data", "GrailQA_no_answer.json")

    no_answer_ids = load_no_answer_ids(no_answer_path)
    result = evaluate_grailqa(eval_data_path, no_answer_ids, data_type="grailqa")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    eval_data_path = config["data_paths"]["grailqa_processed_answer"]
    result_path = config["data_paths"]["grailqa_result"]

    main(eval_data_path, result_path)
