"""
Reasoning using TAG (Thought-Action Graph) with GPT-4o-mini model for KBQA.
This module implements reasoning over the Thought-Action Graph knowledge representation.
"""

import os
import json
import time
from typing import Dict, List, Any
from tqdm import tqdm

from utils import (
    load_dataset,
    load_prompt,
    get_entity_name,
    filter_graph_by_similarity,
    filter_answer,
    OpenAIClient
)
import fasttext


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the config.json file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def gpt4o_mini_generate(client: OpenAIClient, prompt: str, max_attempts: int = 5, delay: int = 5) -> str:
    """
    Generate text using GPT-4o-mini with retry mechanism.
    
    Args:
        client: OpenAIClient instance.
        prompt: Input prompt for text generation.
        max_attempts: Maximum number of retry attempts.
        delay: Delay between retries in seconds.
        
    Returns:
        Generated text response from GPT-4o-mini.
        
    Raises:
        Exception: If maximum retry attempts reached.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.generate_text(
                prompt,
                system_message="You are an expert in the field of natural language processing",
                model="gpt-4o-mini"
            )
            return response
        except Exception as e:
            if attempt > max_attempts:
                raise Exception(f"Request failed after {max_attempts} attempts, please check network connection or API key. Error: {e}")
            time.sleep(delay)


def reason_by_mac(
    client: OpenAIClient,
    fasttext_embedding_model,
    question: str,
    queries_and_macs: List[Dict],
    topic_entity_name: str,
    topic_entity_mid: str,
    graph: List[List[str]],
    number_of_triples: int,
    predict_type_name_prompt: str,
    generate_meta_action_chain_prompt: str,
    reasoning_prompt: str,
    sparql_endpoint: str,
    max_attempts: int = 5,
    delay: int = 5
) -> tuple:
    """
    Perform reasoning using meta-action-chains on the Thought-Action Graph.
    
    Args:
        client: OpenAIClient instance.
        fasttext_embedding_model: Loaded FastText model for embeddings.
        question: Input question string.
        queries_and_macs: List of query-MAC dictionaries.
        topic_entity_name: Name of the topic entity.
        topic_entity_mid: Freebase ID of the topic entity.
        graph: Graph triples [head, relation, tail].
        number_of_triples: Maximum triples to select.
        predict_type_name_prompt: Prompt for type name prediction.
        generate_meta_action_chain_prompt: Prompt for MAC generation.
        reasoning_prompt: Prompt for reasoning.
        sparql_endpoint: SPARQL endpoint URL.
        max_attempts: Maximum retry attempts for API calls.
        delay: Delay between retries in seconds.
        
    Returns:
        Tuple of (subgraph, macs_of_question, answers).
    """
    macs_of_question = []
    
    if len(queries_and_macs) == 0:
        prompt = generate_meta_action_chain_prompt
        question_example = "what country does turkey trade with"
        topicentity_example = "Turkey"
        meta_action_chain_example = "Turkey-->SELECT(DISTINCT ?x)-->WHERE_TRI_PATTERN(Turkey location.statistical_region.places_exported_to ?y)-->WHERE_TRI_PATTERN(?y location.imports_and_exports.exported_to ?x)-->WHERE_TRI_PATTERN(Turkey location.statistical_region.places_imported_from ?y)-->WHERE_TRI_PATTERN(?y location.imports_and_exports.imported_from ?x)-->WHERE_FILTER(?x != ns:m.01znc_)-->UNION-->Algeria; Albania"
        prompt = prompt + "\nQuestion: {}\nTopic entity: {}\nMeta-action-chain:\n{}\n".format(
            question_example, topicentity_example, meta_action_chain_example
        )
        prompt = prompt + "\nPlease write a meta-action-chain based on the given question and topic entity according to the above examples:\nQuestion: {}\nTopic entity: {}\nTopic Entity Mid: {}\nMeta-action-chain:\n".format(
            question, topic_entity_name, topic_entity_mid
        )
        mac_of_question = gpt4o_mini_generate(client, prompt, max_attempts, delay)
        macs_of_question.append(mac_of_question)
    else:
        for Query_Meta_Action_Chain_Dict in queries_and_macs:
            queries = list(Query_Meta_Action_Chain_Dict.keys())
            meta_action_chains = list(Query_Meta_Action_Chain_Dict.values())
            topicentities = [i.split("-->")[0] for i in meta_action_chains]
            
            prompt = generate_meta_action_chain_prompt
            for query, meta_action_chain, topicentity in zip(queries, meta_action_chains, topicentities):
                prompt = prompt + "\nQuestion: {}\nTopic entity: {}\nMeta-action-chain:\n{}\n".format(
                    query, topicentity, meta_action_chain
                )
            prompt = prompt + "\nPlease write a meta-action-chain based on the given question and topic entity according to the above examples:\nQuestion: {}\nTopic entity: {}\nTopic Entity Mid: {}\nMeta-action-chain:\n".format(
                question, topic_entity_name, topic_entity_mid
            )
            mac_of_question = gpt4o_mini_generate(client, prompt, max_attempts, delay)
            macs_of_question.append(mac_of_question)
    
    graph_filtered = filter_graph_by_similarity(
        fasttext_embedding_model, 
        macs_of_question, 
        graph, 
        number_of_triples=number_of_triples
    )
    
    graph_str = ""
    for triple in graph_filtered:
        graph_str += "[{}, {}, {}],\n".format(triple[0], triple[1], triple[2])
    graph_str = "[" + graph_str.strip() + "]"

    answers = []
    for mac_of_question in macs_of_question:
        prompt = reasoning_prompt
        prompt = prompt + "\n\nThought-Action Graph: {}\nQuestion: {}\nTopic Entity: {}\nMeta-action-chain:\n{}\nThe corresponding answer to the question is:\n".format(
            graph_str, question, topic_entity_name, mac_of_question
        )

        max_attempt = 5
        attempt = 0
        while True:
            attempt = attempt + 1
            try:
                gpt4o_mini_answer = gpt4o_mini_generate(client, prompt, max_attempts, delay)
                print("gpt4o_mini_answer:", gpt4o_mini_answer)
                answer = filter_answer(gpt4o_mini_answer)
                answers.append(answer)
                print("filter answer success!")
                break
            except Exception as e:
                if attempt >= max_attempt:
                    answer = "Filter failure"
                    print(f"Filter failed after {max_attempt} attempts: {e}")
                    break
        
    return graph_filtered, macs_of_question, answers


def main():
    """Main function to run reasoning on the dataset."""
    config_path = "config.json"
    config = load_config(config_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("cuda_devices", "0")
    
    tag_explore_breadth = config["tag_config"]["explore_breadth"]
    tag_explore_depth = config["tag_config"]["explore_depth"]
    number_of_triples = config["tag_config"]["number_of_triples"]
    
    openai_config = config.get("openai", {})
    api_key = openai_config.get("api_key")
    if api_key is None:
        raise ValueError("API key not provided in config. Please set 'openai.api_key' in config.json")
    
    fasttext_model_path = config["models"]["fasttext"]["path"]
    
    input_dataset_path = config["data"]["input_dataset"]
    output_directory = config["data"]["output_directory"]
    
    predict_type_name_prompt_path = config["prompts"]["predict_type_name"]
    generate_meta_action_chain_prompt_path = config["prompts"]["generate_meta_action_chain"]
    reasoning_prompt_path = config["prompts"]["reasoning"]
    
    sparql_endpoint = config["sparql_endpoint"]
    
    os.makedirs(output_directory, exist_ok=True)
    
    webqsp_test = load_dataset(input_dataset_path)
    
    predict_type_name_prompt = load_prompt(predict_type_name_prompt_path)
    generate_meta_action_chain_prompt = load_prompt(generate_meta_action_chain_prompt_path)
    reasoning_prompt = load_prompt(reasoning_prompt_path)
    
    fasttext_embedding_model = fasttext.load_model(fasttext_model_path)
    client = OpenAIClient(api_key=api_key)
    
    result = []
    num = 0
    
    output_file_partial = os.path.join(
        output_directory, 
        f"GPT-4o_mini_Reasoning_by_TAG_eval(WebQSP)_results.json"
    )
    
    finished_question_ids = []
    if os.path.exists(output_file_partial):
        with open(output_file_partial, "r", encoding="utf-8") as f:
            answer_result = json.load(f)
            finished_question_ids = [item["QuestionId"] for item in answer_result]
            result = answer_result
            num = len(result)
            print(f"Resuming from {num} already processed questions")
    
    for data in tqdm(webqsp_test):
        if data["QuestionId"] in finished_question_ids:
            continue
        
        num += 1
        question = data["Question"]
        queries_and_macs = data["Queries_and_MACs"][:tag_explore_depth]
        topic_entity_mid = data["TopicEntityMid"]
        topic_entity_name = get_entity_name(data["TopicEntityMid"], sparql_endpoint)
        graph = data["Graph"]
        
        subgraph, macs_of_question, answers = reason_by_mac(
            client,
            fasttext_embedding_model,
            question,
            queries_and_macs,
            topic_entity_name,
            topic_entity_mid,
            graph,
            number_of_triples,
            predict_type_name_prompt,
            generate_meta_action_chain_prompt,
            reasoning_prompt,
            sparql_endpoint
        )
        
        ground_truth_answers = data["Ground_Truth_Answers"]

        result.append({
            "QuestionId": data["QuestionId"],
            "Question": question,
            "TopicEntityMid": data["TopicEntityMid"],
            "Queries_and_MACs": data["Queries_and_MACs"],
            "MACs_of_Question": macs_of_question,
            "Answers": answers,
            "Ground_Truth_Answers": ground_truth_answers,
            "Graph": subgraph
        })

        if num % 20 == 0:
            with open(output_file_partial, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
    
    with open(output_file_partial, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
