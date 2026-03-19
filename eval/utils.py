import json
import transformers
import torch


def load_llama3_1(model_path: str):
    """
    Load the Llama3.1 model and tokenizer.
    
    Args:
        model_path: Path to the Llama3.1 model directory.
        
    Returns:
        A transformers pipeline for text generation.
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def llama3_1_generate(pipeline, prompt: str) -> str:
    """
    Generate text using the Llama3.1 model.
    
    Args:
        pipeline: The transformers pipeline for text generation.
        prompt: The input prompt for text generation.
        
    Returns:
        The generated text content from the model response.
    """
    messages = [
        {"role": "system", "content": "You are a knowledge graph expert in the field of natural language processing."},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )

    return outputs[0]["generated_text"][-1]["content"]


def load_eval_data(data_path: str, data_type: str):
    """
    Load evaluation data from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing evaluation data.
        data_type: Type of the dataset (e.g., "webqsp", "cwq", "grailqa").
        
    Returns:
        The loaded JSON data as a Python object.
        
    Raises:
        ValueError: If the data_type is not supported.
    """
    if data_type not in ["webqsp", "cwq", "grailqa"]:
        raise ValueError(f"Unsupported data type: {data_type}")

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    return eval_data


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


def process_ground_truth_answers(ground_truth_answers: list) -> list:
    """
    Process ground truth answers and extract entity names or answer arguments.
    
    Args:
        ground_truth_answers: List of ground truth answer dictionaries with "AnswerType" and 
                             either "EntityName" or "AnswerArgument" keys.
                             
    Returns:
        List of processed answer strings.
        
    Raises:
        ValueError: If an unsupported AnswerType is encountered.
    """
    processed_answers = []

    for answers_list in ground_truth_answers:
        for answer in answers_list:
            if answer["AnswerType"] == "Entity":
                processed_answers.append(answer["EntityName"])
            elif answer["AnswerType"] == "Value":
                processed_answers.append(answer["AnswerArgument"])
            else:
                raise ValueError(f"Unsupported AnswerType: {answer['AnswerType']}")

    return processed_answers


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


def cal_hits1(answers: list, ground_truth_answers: list) -> int:
    """
    Calculate Hits@1 metric for a set of answers.
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer lists.
        
    Returns:
        1 if the first answer matches any ground truth, 0 otherwise.
    """
    if len(answers) == 0:
        return 0

    processed_gt = process_ground_truth_answers(ground_truth_answers)

    if is_answer_in_ground_truth(answers[0], processed_gt):
        return 1

    return 0


def cal_f1(answers: list, ground_truth_answers: list) -> dict:
    """
    Calculate F1 metrics (precision, recall, F1 score) for a set of answers.
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer lists.
        
    Returns:
        Dictionary containing precision, recall, and F1 score.
    """
    processed_gt = process_ground_truth_answers(ground_truth_answers)

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
    Calculate partial match score for answers.
    
    Args:
        answers: List of predicted answer strings.
        ground_truth_answers: List of ground truth answer lists.
        
    Returns:
        1 if any partial match is found, 0 otherwise.
    """
    processed_gt = process_ground_truth_answers(ground_truth_answers)

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


def flatten_and_deduplicate(answer_list: list) -> list:
    """
    Flatten a list of answer lists and remove duplicates while preserving order.
    
    Args:
        answer_list: List of lists containing answer strings.
        
    Returns:
        Flattened list with duplicates removed.
    """
    seen = set()
    result = []

    for sublist in answer_list:
        for item in sublist:
            if item not in seen:
                seen.add(item)
                result.append(item)

    return result


def process_multiple_answers(answer_str: str) -> list:
    """
    Process a comma-separated answer string into a list of answers.
    
    Args:
        answer_str: Comma-separated answer string.
        
    Returns:
        List of stripped answer strings.
    """
    answers_list = answer_str.split(",")
    answers_list = [ans.strip() for ans in answers_list]
    return answers_list


def cal_multi_answer_hits1(ground_truth_answer: list, multi_answer: list) -> tuple:
    """
    Calculate Hits@1 for multi-answer scenarios (MAC1, MAC2, MAC3).
    
    Args:
        ground_truth_answer: List of ground truth answers.
        multi_answer: List of answer sets, each containing multiple answer candidates.
        
    Returns:
        Tuple of (hits1_mac1, hits1_mac2, hits1_mac3) representing Hits@1 for
        1, 2, and 3 answers respectively.
    """
    hits1_mac1, hits1_mac2, hits1_mac3 = 0, 0, 0

    for answer in multi_answer:
        if len(answer) == 0:
            continue
        if len(answer) == 1:
            answer.append(answer[0])
            answer.append(answer[0])
        if len(answer) == 2:
            answer.append(answer[1])

        answer_1 = answer[0][0]
        answer_2 = answer[1][0]
        answer_3 = answer[2][0]

        hits1_1 = cal_hits1(answer_1, ground_truth_answer)
        hits1_2 = max(cal_hits1(answer_1, ground_truth_answer), cal_hits1(answer_2, ground_truth_answer))
        hits1_3 = max(
            cal_hits1(answer_1, ground_truth_answer),
            cal_hits1(answer_2, ground_truth_answer),
            cal_hits1(answer_3, ground_truth_answer),
        )

        hits1_mac1 = max(hits1_mac1, hits1_1)
        hits1_mac2 = max(hits1_mac2, hits1_2)
        hits1_mac3 = max(hits1_mac3, hits1_3)

    return hits1_mac1, hits1_mac2, hits1_mac3


def cal_multi_answer_f1(ground_truth_answer: list, multi_answer: list) -> tuple:
    """
    Calculate F1 scores for multi-answer scenarios (MAC1, MAC2, MAC3).
    
    Args:
        ground_truth_answer: List of ground truth answers.
        multi_answer: List of answer sets, each containing multiple answer candidates.
        
    Returns:
        Tuple of (f1_mac1, f1_mac2, f1_mac3) representing F1 scores for
        1, 2, and 3 answers respectively.
    """
    answer_1, answer_2, answer_3 = [], [], []

    for answer in multi_answer:
        if len(answer) == 0:
            continue
        answer_1 = answer_1 + answer[0][0]
        answer_2 = answer_2 + answer[1][0]
        answer_3 = answer_3 + answer[2][0]

    answer_1 = list(set(answer_1))
    answer_2 = list(set(answer_1 + answer_2))
    answer_3 = list(set(answer_1 + answer_2 + answer_3))

    f1_mac1 = cal_f1(answer_1, ground_truth_answer)
    f1_mac2 = cal_f1(answer_2, ground_truth_answer)
    f1_mac3 = cal_f1(answer_3, ground_truth_answer)

    return f1_mac1, f1_mac2, f1_mac3


def cal_multi_answer_partial(ground_truth_answer: list, multi_answer: list) -> tuple:
    """
    Calculate partial match scores for multi-answer scenarios (MAC1, MAC2, MAC3).
    
    Args:
        ground_truth_answer: List of ground truth answers.
        multi_answer: List of answer sets, each containing multiple answer candidates.
        
    Returns:
        Tuple of (partial_1, partial_2, partial_3) representing partial scores for
        1, 2, and 3 answers respectively.
    """
    answer_1, answer_2, answer_3 = [], [], []

    for answer in multi_answer:
        if len(answer) == 0:
            continue
        answer_1 = answer_1 + answer[0][0]
        answer_2 = answer_2 + answer[1][0]
        answer_3 = answer_3 + answer[2][0]

    answer_1 = list(set(answer_1))
    answer_2 = list(set(answer_1 + answer_2))
    answer_3 = list(set(answer_1 + answer_2 + answer_3))

    partial_1 = cal_partial(answer_1, ground_truth_answer)
    partial_2 = cal_partial(answer_2, ground_truth_answer)
    partial_3 = cal_partial(answer_3, ground_truth_answer)

    return partial_1, partial_2, partial_3
