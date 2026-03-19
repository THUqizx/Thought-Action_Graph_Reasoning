import json
import os
from utils import load_eval_data, cal_hits1, cal_f1, cal_partial, process_multiple_answers


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


def evaluate_webqsp(
    eval_data_path: str,
    no_answer_ids: set,
    data_type: str = "webqsp"
) -> list:
    """
    Evaluate WebQSP predictions using MAC1, MAC2, and MAC3 metrics.
    
    Args:
        eval_data_path: Path to the evaluation data JSON file.
        no_answer_ids: Set of question IDs to skip (no valid answers).
        data_type: Type of dataset (default: "webqsp").
        
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
            ground_truth_answers = data["Ground_Truth_Answers"]
            answers_1 = data["Answers"][0]
            answers_2 = data["Answers"][1]
            answers_3 = data["Answers"][2]

            hits1_1 = cal_hits1(answers_1, ground_truth_answers)
            hits1_2 = max(
                cal_hits1(answers_1, ground_truth_answers),
                cal_hits1(answers_2, ground_truth_answers),
            )
            hits1_3 = max(
                cal_hits1(answers_1, ground_truth_answers),
                cal_hits1(answers_2, ground_truth_answers),
                cal_hits1(answers_3, ground_truth_answers),
            )

            f1_results_1 = cal_f1(answers_1, ground_truth_answers)
            f1_results_2 = cal_f1(list(set(answers_1 + answers_2)), ground_truth_answers)
            f1_results_3 = cal_f1(list(set(answers_1 + answers_2 + answers_3)), ground_truth_answers)

            partial_1 = cal_partial(answers_1, ground_truth_answers)
            partial_2 = cal_partial(answers_1 + answers_2, ground_truth_answers)
            partial_3 = cal_partial(answers_1 + answers_2 + answers_3, ground_truth_answers)

            mac_1["hits1"] += hits1_1
            mac_1["precision"] += f1_results_1["precision"]
            mac_1["recall"] += f1_results_1["recall"]
            mac_1["f1"] += f1_results_1["f1"]
            mac_1["partial"] += partial_1

            mac_2["hits1"] += hits1_2
            mac_2["precision"] += f1_results_2["precision"]
            mac_2["recall"] += f1_results_2["recall"]
            mac_2["f1"] += f1_results_2["f1"]
            mac_2["partial"] += partial_2

            mac_3["hits1"] += hits1_3
            mac_3["precision"] += f1_results_3["precision"]
            mac_3["recall"] += f1_results_3["recall"]
            mac_3["f1"] += f1_results_3["f1"]
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
    Main function to evaluate WebQSP predictions and save results.
    
    Args:
        eval_data_path: Path to the evaluation data JSON file.
        result_path: Path to save the evaluation results.
        no_answer_path: Optional path to no-answer JSON file. If None, 
                       uses default path from config.
    """
    if no_answer_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        no_answer_path = os.path.join(base_dir, "data", "WebQSP_no_answer.json")

    no_answer_ids = load_no_answer_ids(no_answer_path)
    result = evaluate_webqsp(eval_data_path, no_answer_ids, data_type="webqsp")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    eval_data_path = config["data_paths"]["webqsp_processed_answer"]
    result_path = config["data_paths"]["webqsp_result"]

    main(eval_data_path, result_path)
