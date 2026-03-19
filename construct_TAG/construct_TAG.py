"""
Construct Thought-Action Graph (TAG) from Meta-Action Chains (MAC).

This module provides functionality to convert MAC data into a dual-layer
Thought-Action Graph structure stored in ThoughtActionGraph format.
"""

import json
import re
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from ThoughtActionGraph import ThoughtActionGraph
from utils import typeName_all, load_config


def option_decompose(option: str) -> Tuple[str, str]:
    """
    Decompose an option string into option and action parts.
    
    Args:
        option: Option string in format "Option(Action)" or just "Option".
        
    Returns:
        Tuple of (option_name, action_name). Action may be None.
    """
    if "(" not in option and ")" not in option:
        return option, option
    
    pattern = r'^([^(]+)\((.*)\)$'
    match = re.match(pattern, option)
    
    if match:
        option_name = match.group(1)
        action = match.group(2)
        return option_name, action
    else:
        return None, None


def mac_to_triples(data_with_mac: Dict[str, Any], sparql_endpoint: str) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]], List[List[Any]]]:
    """
    Convert MAC data to four types of triples:
    1. Thought triples (Ontology-level reasoning chain)
    2. Action triples (Entity-level execution chain)
    3. Ontology-to-Entity triples (Mapping between layers)
    4. Option-to-Action triples (Parameter mapping)
    
    Args:
        data_with_mac: Dictionary containing MAC and related information.
        sparql_endpoint: URL of the SPARQL endpoint for type queries.
        
    Returns:
        Tuple of (ThoughtTriples, ActionTriples, Ontology2EntityTriples, Option2ActionTriples).
    """
    mac_parts = data_with_mac["Meta-Action-Chain"].split('-->')
    option_and_action = mac_parts[1:-1]
    
    thought_triples = []
    action_triples = []
    ontology_to_entity_triples = []
    option_to_action_triples = []
    
    option_chain = []
    action_chain = []

    # Build Ontology-to-Entity triples for head entity
    head_entity = mac_parts[0]
    assert head_entity == data_with_mac["TopicEntityName"]
    ontology_list = typeName_all(data_with_mac["TopicEntityMid"], sparql_endpoint)
    for ontology in ontology_list:
        ontology_to_entity_triples.append([
            ontology, {"Type": "Ontology", "Layer": "Thought"}, 
            "instance_of", 
            head_entity, {"Type": "Entity", "Layer": "Action"}
        ])
    
    # Build Ontology-to-Entity triples for tail entities (answers)
    source = data_with_mac["Source"]
    if source == "CWQ":
        answers = data_with_mac["Answers"]
        for answer in answers:
            tail_entity = answer["answer"]
            ontology_list = typeName_all(answer["answer_id"], sparql_endpoint)
            for ontology in ontology_list:
                ontology_to_entity_triples.append([
                    ontology, {"Type": "Ontology", "Layer": "Thought"}, 
                    "instance_of", 
                    tail_entity, {"Type": "Entity", "Layer": "Action"}
                ])
    elif source == "WebQSP":
        answers = data_with_mac["Answers"]
        for answer in answers:
            tail_entity = answer["EntityName"]
            ontology_list = typeName_all(answer["AnswerArgument"], sparql_endpoint)
            for ontology in ontology_list:
                ontology_to_entity_triples.append([
                    ontology, {"Type": "Ontology", "Layer": "Thought"}, 
                    "instance_of", 
                    tail_entity, {"Type": "Entity", "Layer": "Action"}
                ])
    elif source == "GrailQA":
        answers = data_with_mac["Answers"]
        for answer in answers:
            tail_entity = answer["entity_name"]
            ontology_list = typeName_all(answer["answer_argument"], sparql_endpoint)
            for ontology in ontology_list:
                ontology_to_entity_triples.append([
                    ontology, {"Type": "Ontology", "Layer": "Thought"}, 
                    "instance_of", 
                    tail_entity, {"Type": "Entity", "Layer": "Action"}
                ])
    else:
        raise ValueError(f"Unknown Source: {source}")

    # Build Option-to-Action triples
    for oa in option_and_action:
        option, action = option_decompose(oa)
        assert option is not None
        if option not in option_chain:
            option_chain.append(option)
        if action is None:
            continue
        else:
            action_chain.append(action)
            option_to_action_triples.append([
                option, {"Type": "Option", "Layer": "Thought"}, 
                "has_parameter", 
                action, {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}
            ])
    
    # Build Thought layer triples
    head_ontology_list = typeName_all(data_with_mac["TopicEntityMid"], sparql_endpoint)
    tail_ontology_list = []
    for ans in data_with_mac["Answers"]:
        if source == "CWQ":
            tail_ontology_list.extend(typeName_all(ans["answer_id"], sparql_endpoint))
        elif source == "WebQSP":
            tail_ontology_list.extend(typeName_all(ans["AnswerArgument"], sparql_endpoint))
        elif source == "GrailQA":
            tail_ontology_list.extend(typeName_all(ans["answer_argument"], sparql_endpoint))
        else:
            raise ValueError(f"Unknown Source: {source}")
    tail_ontology_list = list(set(tail_ontology_list))
    
    for head_ontology in head_ontology_list:
        for tail_ontology in tail_ontology_list:
            for step, opt in enumerate(option_chain):
                if step == 0:
                    thought_triples.append([
                        head_ontology, {"Type": "Ontology", "Layer": "Thought"}, 
                        "start", 
                        opt, {"Type": "Option", "Layer": "Thought"}
                    ])
                elif step == len(option_chain) - 1:
                    thought_triples.append([
                        option_chain[step - 1], {"Type": "Option", "Layer": "Thought"}, 
                        "next", 
                        opt, {"Type": "Option", "Layer": "Thought"}
                    ])
                    thought_triples.append([
                        opt, {"Type": "Option", "Layer": "Thought"}, 
                        "end", 
                        tail_ontology, {"Type": "Ontology", "Layer": "Thought"}
                    ])
                else:
                    thought_triples.append([
                        option_chain[step - 1], {"Type": "Option", "Layer": "Thought"}, 
                        "next", 
                        opt, {"Type": "Option", "Layer": "Thought"}
                    ])
    
    # Build Action layer triples
    head_entity = data_with_mac["TopicEntityName"]
    if source == "CWQ":
        tail_entities = [ans["answer"] for ans in data_with_mac["Answers"]]
    elif source == "WebQSP":
        tail_entities = [ans["EntityName"] for ans in data_with_mac["Answers"]]
    elif source == "GrailQA":
        tail_entities = [ans["entity_name"] for ans in data_with_mac["Answers"]]
    else:
        raise ValueError(f"Unknown Source: {source}")
    
    for tail_entity in tail_entities:
        for step, act in enumerate(action_chain):
            if step == 0:
                action_triples.append([
                    head_entity, {"Type": "Entity", "Layer": "Action"}, 
                    "start", 
                    act, {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}
                ])
            elif step == len(action_chain) - 1:
                action_triples.append([
                    action_chain[step - 1], {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}, 
                    "next", 
                    act, {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}
                ])
                action_triples.append([
                    act, {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}, 
                    "end", 
                    tail_entity, {"Type": "Entity", "Layer": "Action"}
                ])
            else:
                action_triples.append([
                    action_chain[step - 1], {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}, 
                    "next", 
                    act, {"Type": "Action", "Layer": "Action", "Question": data_with_mac["ProcessedQuestion"]}
                ])
    
    return thought_triples, action_triples, ontology_to_entity_triples, option_to_action_triples


def construct_tag_from_mac(mac_path: str, tag_path: str, sparql_endpoint: str) -> None:
    """
    Construct Thought-Action Graph from MAC data and save to file.
    
    Args:
        mac_path: Path to MAC JSON file.
        tag_path: Path to save the TAG file.
        sparql_endpoint: URL of the SPARQL endpoint.
    """
    with open(mac_path, 'r', encoding='utf-8') as f:
        mac_data = json.load(f)
    
    tag = ThoughtActionGraph()
    
    for data_with_mac in tqdm(mac_data, desc="Constructing TAG"):
        try:
            thought_triples, action_triples, ontology_to_entity_triples, option_to_action_triples = mac_to_triples(data_with_mac, sparql_endpoint)
            
            for triples in [
                thought_triples, 
                action_triples, 
                ontology_to_entity_triples, 
                option_to_action_triples
            ]:
                for triple in triples:
                    head_name, head_attrs, relation, tail_name, tail_attrs = triple
                    tag.add_triple(head_name, head_attrs, relation, tail_name, tail_attrs)
        except Exception as e:
            print(f"ERROR processing {data_with_mac.get('QuestionId', 'Unknown')}: {e}")
    
    tag.save_to_file(tag_path)


def main():
    """
    Main function to construct TAG from MAC data.
    
    Usage:
        1. Set paths in config.json
        2. Run: python construct_TAG.py
    """
    config = load_config("config.json")
    
    paths = config.get("paths", {})
    sparql_endpoint = config.get("sparql_endpoint")
    
    mac_path = paths.get("mac_path")
    tag_path = paths.get("tag_path")
    
    print(f"Loading MAC from: {mac_path}")
    print(f"Saving TAG to: {tag_path}")
    
    construct_tag_from_mac(mac_path, tag_path, sparql_endpoint)
    
    print(f"\nTAG construction completed! Saved to {tag_path}")


if __name__ == "__main__":
    main()
