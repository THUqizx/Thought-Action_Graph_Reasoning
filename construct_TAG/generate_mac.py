"""
Generate Meta-Action Chains (MAC) for WebQSP, CWQ, and GrailQA datasets using GPT.

This module provides a unified interface for generating MAC from different datasets
using GPT models. It handles dataset-specific processing and saves results to JSON files.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm

def load_json(path: str) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        Loaded JSON data.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save.
        path: Path to save the JSON file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

class MACGenerator:
    """
    Generate Meta-Action Chains (MAC) for questions using GPT models.
    """
    
    def __init__(self, client, api_key: str, model_name: str = "gpt-4o",
                 max_attempts: int = 5, delay: int = 30):
        """
        Initialize the MAC generator.
        
        Args:
            client: OpenAI client instance.
            api_key: API key for authentication.
            model_name: Name of the GPT model to use.
            max_attempts: Maximum number of retry attempts.
            delay: Delay between retry attempts in seconds.
        """
        self.client = client
        self.api_key = api_key
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.delay = delay
    
    def generate_mac(self, prompt: str, system_message: str = "You are an expert in the field of natural language processing") -> Optional[str]:
        """
        Generate MAC using GPT with retry logic.
        
        Args:
            prompt: Prompt to send to GPT.
            system_message: System message for the model.
            
        Returns:
            Generated MAC string, or None if all attempts fail.
        """
        attempt = 1
        while attempt <= self.max_attempts:
            try:
                mac = self.client.generate_text(
                    prompt,
                    system_message=system_message,
                    model=self.model_name
                )
                return mac
            except Exception as e:
                if attempt == self.max_attempts:
                    print(f"Failed after {self.max_attempts} attempts: {e}")
                    return None
                time.sleep(self.delay)
            attempt += 1
        return None

def process_webqsp(data: Dict[str, Any], generator: MACGenerator, prompt: str, 
                   save_path: str, utils_module) -> None:
    """
    Process WebQSP dataset and generate MAC.
    
    Args:
        data: WebQSP data dictionary.
        generator: MACGenerator instance.
        prompt: Prompt template for MAC generation.
        save_path: Path to save results.
        utils_module: utils module for helper functions.
    """
    from utils import extract_mac_content
    
    result_mac = []
    
    if os.path.exists(save_path):
        result_mac = load_json(save_path)
    
    already_processed = {item["QuestionId"] for item in result_mac}
    
    for sample in tqdm(data["Questions"], desc="Processing WebQSP"):
        question_id = sample["QuestionId"]
        
        if question_id in already_processed:
            continue
        
        question = sample["ProcessedQuestion"]
        
        for parse in sample["Parses"]:
            try:
                answers = [answer["EntityName"] for answer in parse["Answers"]]
                answer_str = "; ".join(answers)
                sparql = parse["Sparql"]
                head_entity = parse["TopicEntityName"]
                tail_entity = "; ".join(answers)
                
                prompt_text = (prompt + "\n" + 
                              f"Question: {question}\n" +
                              f"Answer: {answer_str}\n" +
                              f"SPARQL: {sparql}\n" +
                              f"Head Entity: {head_entity}\n" +
                              f"Tail Entity: {tail_entity}\n" +
                              "meta-action chain:\n")
                
                mac = generator.generate_mac(prompt_text)
                
                if mac:
                    mac_content = extract_mac_content(mac)
                    if mac_content:
                        result_mac.append({
                            "Source": "WebQSP",
                            "QuestionId": question_id,
                            "ProcessedQuestion": question,
                            "TopicEntityMid": parse.get("TopicEntityMid"),
                            "TopicEntityName": head_entity,
                            "Answers": parse["Answers"],
                            "Meta-Action-Chain": mac_content[0]
                        })
                        print(f"Question: {question}")
                        print("=" * 50)
                        print(f"MAC: {mac_content[0]}")
                        print("=" * 50)
                    else:
                        print(f"Could not extract MAC content for question {question_id}")
                else:
                    print(f"Could not generate MAC for question {question_id}")
            except Exception as e:
                print(f"Error processing WebQSP sample {question_id}: {e}")
        
        if len(result_mac) % 50 == 0:
            save_json(result_mac, save_path)
    
    save_json(result_mac, save_path)

def process_cwq(data: List[Dict[str, Any]], generator: MACGenerator, prompt: str,
                save_path: str, utils_module) -> None:
    """
    Process CWQ dataset and generate MAC.
    
    Args:
        data: CWQ data list.
        generator: MACGenerator instance.
        prompt: Prompt template for MAC generation.
        save_path: Path to save results.
        utils_module: utils module for helper functions.
    """
    from utils import extract_mac_content
    
    result_mac = []
    
    if os.path.exists(save_path):
        result_mac = load_json(save_path)
    
    already_processed = {item["QuestionId"] for item in result_mac}
    
    for sample in tqdm(data, desc="Processing CWQ"):
        question_id = sample["ID"]
        
        if question_id in already_processed:
            continue
        
        question = sample["question"]
        
        sample_answer = []
        for answer in sample["answers"]:
            if answer["answer"]:
                sample_answer.append(answer["answer"])
            else:
                sample_answer.append(answer["answer_id"])
        
        sparql = sample["sparql"]
        head_entity = sample["TopicEntityName"]
        tail_entity = "; ".join(sample_answer)
        
        prompt_text = (prompt + "\n" +
                      f"Question: {question}\n" +
                      f"Answer: {'; '.join(sample_answer)}\n" +
                      f"SPARQL: {sparql}\n" +
                      f"Head Entity: {head_entity}\n" +
                      f"Tail Entity: {tail_entity}\n" +
                      "meta-action chain:\n")
        
        mac = generator.generate_mac(prompt_text)
        
        if mac:
            mac_content = extract_mac_content(mac)
            if mac_content:
                result_mac.append({
                    "Source": "CWQ",
                    "QuestionId": question_id,
                    "ProcessedQuestion": question,
                    "TopicEntityMid": sample.get("TopicEntityMid"),
                    "TopicEntityName": head_entity,
                    "Answers": [{"answer": ans["answer"], "answer_id": ans["answer_id"]} for ans in sample["answers"]],
                    "Meta-Action-Chain": mac_content[0]
                })
                print(f"Question: {question}")
                print("=" * 50)
                print(f"MAC: {mac_content[0]}")
                print("=" * 50)
            else:
                print(f"Could not extract MAC content for question {question_id}")
        else:
            print(f"Could not generate MAC for question {question_id}")
        
        if len(result_mac) % 50 == 0:
            save_json(result_mac, save_path)
    
    save_json(result_mac, save_path)

def process_grailqa(data: List[Dict[str, Any]], generator: MACGenerator, prompt: str,
                    save_path: str, utils_module) -> None:
    """
    Process GrailQA dataset and generate MAC.
    
    Args:
        data: GrailQA data list.
        generator: MACGenerator instance.
        prompt: Prompt template for MAC generation.
        save_path: Path to save results.
        utils_module: utils module for helper functions.
    """
    from utils import extract_mac_content, sparql_preprocess
    
    result_mac = []
    
    if os.path.exists(save_path):
        result_mac = load_json(save_path)
    
    already_processed = {item["QuestionId"] for item in result_mac}
    
    for sample in tqdm(data, desc="Processing GrailQA"):
        question_id = sample["qid"]
        
        if question_id in already_processed:
            continue
        
        question = sample["question"]
        
        answer_type = [ans["answer_type"] for ans in sample["answer"]]
        if not all(atype == "Entity" for atype in answer_type):
            continue
        
        sample_answer = [ans["entity_name"] for ans in sample["answer"]]
        sparql = sparql_preprocess(sample["sparql_query"])
        
        try:
            nodes = sample["graph_query"]["nodes"]
            head_entity_id = None
            head_entity_name = None
            for node in nodes:
                if node["node_type"] == "entity":
                    head_entity_id = node["id"]
                    head_entity_name = node["friendly_name"]
                    break
            
            if not head_entity_id:
                continue
                
            head_entity_name = utils_module.entityName(head_entity_id)
        except Exception as e:
            print(f"Error processing GrailQA sample {question_id}: {e}")
            continue
        
        tail_entity = "; ".join(sample_answer)
        
        prompt_text = (prompt + "\n" +
                      f"Question: {question}\n" +
                      f"Answer: {'; '.join(sample_answer)}\n" +
                      f"SPARQL: {sparql}\n" +
                      f"Head Entity: {head_entity_name}\n" +
                      f"Tail Entity: {tail_entity}\n" +
                      "meta-action chain:\n")
        
        mac = generator.generate_mac(prompt_text)
        
        if mac:
            mac_content = extract_mac_content(mac)
            if mac_content:
                result_mac.append({
                    "Source": "GrailQA",
                    "QuestionId": question_id,
                    "ProcessedQuestion": question,
                    "TopicEntityMid": head_entity_id,
                    "TopicEntityName": head_entity_name,
                    "Answers": sample["answer"],
                    "Meta-Action-Chain": mac_content[0]
                })
                print(f"Question: {question}")
                print("=" * 50)
                print(f"MAC: {mac_content[0]}")
                print("=" * 50)
            else:
                print(f"Could not extract MAC content for question {question_id}")
        else:
            print(f"Could not generate MAC for question {question_id}")
        
        if len(result_mac) % 50 == 0:
            save_json(result_mac, save_path)
    
    save_json(result_mac, save_path)

def merge_all_mac(webqsp_path: str, cwq_path: str, grailqa_path: str, merged_path: str) -> None:
    """
    Merge MAC data from WebQSP, CWQ, and GrailQA into a single file.
    
    Args:
        webqsp_path: Path to WebQSP MAC JSON file.
        cwq_path: Path to CWQ MAC JSON file.
        grailqa_path: Path to GrailQA MAC JSON file.
        merged_path: Path to save merged MAC data.
    """
    result = []
    
    if os.path.exists(webqsp_path):
        webqsp_data = load_json(webqsp_path)
        result.extend(webqsp_data)
        print(f"Loaded {len(webqsp_data)} samples from WebQSP")
    
    if os.path.exists(cwq_path):
        cwq_data = load_json(cwq_path)
        result.extend(cwq_data)
        print(f"Loaded {len(cwq_data)} samples from CWQ")
    
    if os.path.exists(grailqa_path):
        grailqa_data = load_json(grailqa_path)
        result.extend(grailqa_data)
        print(f"Loaded {len(grailqa_data)} samples from GrailQA")
    
    print(f"Total samples after merge: {len(result)}")
    
    save_json(result, merged_path)
    print(f"Merged MAC saved to: {merged_path}")

def main():
    """
    Main function to generate MAC for WebQSP, CWQ, and GrailQA datasets.
    
    Usage:
        1. Set the paths in config.json
        2. Set your API key in config.json
        3. Run: python generate_MAC.py
    """
    import chat_with_gpt
    from utils import load_config
    
    config = load_config("config.json")
    
    api_key = config.get("api_key", "YOUR_API_KEY_HERE")
    if api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set your API key in config.json")
    
    model_name = config.get("model_name", "gpt-4o-mini")
    max_attempts = config.get("gpt_settings", {}).get("max_attempts", 5)
    delay = config.get("gpt_settings", {}).get("delay", 30)
    
    paths = config.get("paths", {})
    
    client = chat_with_gpt.OpenAIClient(api_key)
    generator = MACGenerator(client, api_key, model_name, max_attempts, delay)
    
    prompt_path = paths.get("prompt_file")
    if not os.path.exists(prompt_path):
        raise ValueError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    import utils as utils_module
    
    webqsp_path = paths.get("webqsp_data")
    if webqsp_path and os.path.exists(webqsp_path):
        print("\n" + "=" * 50)
        print("Processing WebQSP")
        print("=" * 50)
        webqsp_data = load_json(webqsp_path)
        webqsp_save = paths.get("webqsp_mac", "data/MAC_WebQSP.json")
        process_webqsp(webqsp_data, generator, prompt, webqsp_save, utils_module)
    
    cwq_path = paths.get("cwq_data")
    if cwq_path and os.path.exists(cwq_path):
        print("\n" + "=" * 50)
        print("Processing CWQ")
        print("=" * 50)
        cwq_data = load_json(cwq_path)
        cwq_save = paths.get("cwq_mac", "data/MAC_CWQ.json")
        process_cwq(cwq_data, generator, prompt, cwq_save, utils_module)
    
    grailqa_path = paths.get("grailqa_data")
    if grailqa_path and os.path.exists(grailqa_path):
        print("\n" + "=" * 50)
        print("Processing GrailQA")
        print("=" * 50)
        grailqa_data = load_json(grailqa_path)
        grailqa_save = paths.get("grailqa_mac", "data/MAC_GrailQA.json")
        process_grailqa(grailqa_data, generator, prompt, grailqa_save, utils_module)
    
    # Merge all MAC data
    merged_path = paths.get("merged_mac", "data/MAC_WebQSP_CWQ_GrailQA.json")
    if merged_path:
        print("\n" + "=" * 50)
        print("Merging all MAC data")
        print("=" * 50)
        webqsp_mac = paths.get("webqsp_mac", "data/MAC_WebQSP.json")
        cwq_mac = paths.get("cwq_mac", "data/MAC_CWQ.json")
        grailqa_mac = paths.get("grailqa_mac", "data/MAC_GrailQA.json")
        merge_all_mac(webqsp_mac, cwq_mac, grailqa_mac, merged_path)
    
    print("\n" + "=" * 50)
    print("MAC generation completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
