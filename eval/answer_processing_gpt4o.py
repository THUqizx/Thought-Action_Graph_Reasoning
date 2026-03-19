import json
import os
from utils import load_eval_data, cal_hits1, cal_f1, process_multiple_answers


def process_answers(
    answer_path: str,
    save_path: str,
    data_type: str = "webqsp"
):
    """
    Process raw answers from GPT-4o model and format them for evaluation.
    
    This function handles answer formatting, including:
    - Extracting first answer from each answer set
    - Ensuring at least 3 answer sets are present (filling missing with "No_Answer")
    
    Args:
        answer_path: Path to the input JSON file containing raw answers.
        save_path: Path to save the processed JSON file.
        data_type: Type of dataset (default: "webqsp").
    """
    eval_data = load_eval_data(answer_path, data_type)

    result = []

    for data in eval_data:
        answer_processing = []
        answers = data["Answers"]

        for answer in answers:
            answer_processing.append(answer[0])

        if len(answer_processing) == 1:
            answer_processing.append(["No_Answer"])
            answer_processing.append(["No_Answer"])
        if len(answer_processing) == 2:
            answer_processing.append(["No_Answer"])

        data["Answers"] = answer_processing
        result.append(data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    answer_path = config["data_paths"]["gpt4o_raw_answer"]
    save_path = config["data_paths"]["gpt4o_processed_answer"]

    process_answers(answer_path, save_path, data_type="webqsp")
