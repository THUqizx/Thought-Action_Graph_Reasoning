#!/usr/bin/env python3
"""
Main reasoning script using Thought-Action Graph for KBQA.

This script implements reasoning over the Thought-Action Graph using Llama-3.1
for question answering based on Freebase knowledge graph.

Author: TAG Team
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
import fasttext

from utils import (
    load_llama3_1, 
    llama3_1_generate, 
    load_dataset, 
    load_prompt, 
    load_embeddings, 
    typeName, 
    typeName_all, 
    GraphFilter, 
    filter_answer,
    load_config
)
from thought_action_graph import ThoughtActionGraph
from retrieval import (
    RetrievalFromStart,
    pruning,
    ThoughtNode2ActionNode,
    RetrievalFromStart2End,
    FilterActionChain,
    ActionChain2ThoughtChain,
    MergeMetaActionChain,
    RetrievalSimilarQueries
)

def reason_by_TAG(
    llama3_1,
    question: str,
    topic_entity_mid: str,
    topic_entity_name: str,
    graph: List[List[str]],
    typename_embeddings: Dict[str, np.ndarray],
    query_embeddings: Dict[str, np.ndarray],
    tag: ThoughtActionGraph,
    qwen3_model: SentenceTransformer,
    fasttext_model,
    predict_type_name_prompt: str,
    generate_meta_action_chain_prompt: str,
    reasoning_prompt: str,
    tag_explore_breadth: int = 5,
    tag_explore_depth: int = 3
) -> Tuple[List[Dict], List[str], List[List[str]], List[List[str]]]:
    """
    Perform reasoning using Thought-Action Graph for a given question.
    
    Args:
        llama3_1: Llama-3.1 model pipeline
        question: Input question string
        topic_entity_mid: Topic entity Freebase ID
        topic_entity_name: Topic entity name
        graph: Knowledge graph triples [head, relation, tail]
        typename_embeddings: Answer type name embeddings
        query_embeddings: Query embeddings
        tag: ThoughtActionGraph instance
        qwen3_model: Qwen3 embedding model
        fasttext_model: FastText embedding model
        predict_type_name_prompt: Prompt for predicting answer type
        generate_meta_action_chain_prompt: Prompt for generating meta action chains
        reasoning_prompt: Prompt for final reasoning
        tag_explore_breadth: TAG exploration breadth
        tag_explore_depth: TAG exploration depth
        
    Returns:
        Tuple of (queries_and_macs, macs_of_question, answers, filtered_graph)
    """
    topic_entity_typeName_all = typeName_all(topic_entity_mid, tag.sparql_endpoint)
    topic_entity_typeName_all = [
        typeName for typeName in topic_entity_typeName_all 
        if typeName not in ["Topic", "Abstract", "Inanimate", "Agent", "Non-Agent"]
    ]
    print(f"topic_entity_typeName_all: {topic_entity_typeName_all}")
    
    prompt = predict_type_name_prompt + "\nQuestion: {}\nThe type name of the answer:".format(question)
    answer_typeName_pre = llama3_1_generate(llama3_1, prompt)
    answer_typeNames = RetrievalSimilarQueries(qwen3_model, answer_typeName_pre, typename_embeddings, top_k=tag_explore_breadth)
    print(f"Question: {question}")
    answer_typeNames = [typeName[0] for typeName in answer_typeNames]
    print(f"answer_typeNames: {answer_typeNames}")
    
    queries_and_meta_action_chains_all = {}

    for topic_entity_typeName in topic_entity_typeName_all:
        queries_and_meta_action_chains = retrieval(
            question, 
            query_embeddings, 
            topic_entity_typeName, 
            answer_typeNames,
            tag,
            qwen3_model
        )
        print(f"Queries_and_Meta_Action_Chains: {queries_and_meta_action_chains}")
        for query_mac in queries_and_meta_action_chains:
            for query in list(query_mac.keys()):
                if query not in queries_and_meta_action_chains_all:
                    queries_and_meta_action_chains_all[query] = query_mac[query]
    
    queries_and_meta_action_chains_all_embedding = {
        k: v for k, v in query_embeddings.items() 
        if k in queries_and_meta_action_chains_all
    }
    queries_and_meta_action_chains_list = RetrievalSimilarQueries(
        qwen3_model, 
        question, 
        queries_and_meta_action_chains_all_embedding, 
        top_k=tag_explore_depth
    )
    queries_and_meta_action_chains_list = [query[0] for query in queries_and_meta_action_chains_list]
    
    queries_and_meta_action_chains = []
    for query in queries_and_meta_action_chains_list:
        queries_and_meta_action_chains.append({query: queries_and_meta_action_chains_all[query]})

    print(f"Queries_and_Meta_Action_Chains_len: {len(queries_and_meta_action_chains)}")
    
    macs_of_question = []
    if len(queries_and_meta_action_chains) == 0:
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
        mac_of_question = llama3_1_generate(llama3_1, prompt)
        macs_of_question.append(mac_of_question)
    else:
        for query_meta_action_chain_dict in queries_and_meta_action_chains:
            queries = list(query_meta_action_chain_dict.keys())
            meta_action_chains = list(query_meta_action_chain_dict.values())
            topicentities = [i.split("-->")[0] for i in meta_action_chains]
            prompt = generate_meta_action_chain_prompt
            for query, meta_action_chain, topicentity in zip(queries, meta_action_chains, topicentities):
                prompt = prompt + "\nQuestion: {}\nTopic entity: {}\nMeta-action-chain:\n{}\n".format(
                    query, topicentity, meta_action_chain
                )
            prompt = prompt + "\nPlease write a meta-action-chain based on the given question and topic entity according to the above examples:\nQuestion: {}\nTopic entity: {}\nTopic Entity Mid: {}\nMeta-action-chain:\n".format(
                question, topic_entity_name, topic_entity_mid
            )
            mac_of_question = llama3_1_generate(llama3_1, prompt)
            macs_of_question.append(mac_of_question)

    filtered_graph = GraphFilter(fasttext_model, macs_of_question, graph, number_of_triples=200)
    graph_str = ""
    for triple in filtered_graph:
        graph_str += "[{}, {}, {}],\n".format(triple[0], triple[1], triple[2])
    graph_str = "[" + graph_str.strip() + "]"

    answers = []
    print(f"macs_of_question_len: {len(macs_of_question)}")
    for mac_of_question in macs_of_question:
        prompt = reasoning_prompt
        prompt = prompt + "\n\nKnowledge Graph: {}\nQuestion: {}\nTopic Entity: {}\nMeta-action-chain:\n{}\nThe corresponding answer to the question is:\n".format(
            graph_str, question, topic_entity_name, mac_of_question
        )

        max_attempt = 5
        attempt = 0
        while attempt < max_attempt:
            attempt += 1
            try:
                llama3_1_answer = llama3_1_generate(llama3_1, prompt)
                print(f"llama3_1_answer: {llama3_1_answer}")
                answer = filter_answer(llama3_1_answer)
                answers.append(answer)
                print("filter answer success!")
                break
            except Exception as e:
                print(f"filter answer failure, attempt: {attempt}, error: {e}")
                if attempt >= max_attempt:
                    answer = ["Filter failure"]
                    break
        
    return queries_and_meta_action_chains, macs_of_question, answers, filtered_graph


def retrieval(
    question: str,
    query_embeddings: Dict[str, np.ndarray],
    topic_entity_typeName: str,
    answer_typeNames: List[str],
    tag: ThoughtActionGraph,
    qwen3_model: SentenceTransformer
) -> List[Dict[str, str]]:
    """
    Retrieval for a given question and entity type.
    
    Args:
        question: Input question string
        query_embeddings: Query embeddings dictionary
        topic_entity_typeName: Topic entity type name
        answer_typeNames: List of possible answer type names
        tag: ThoughtActionGraph instance
        qwen3_model: Qwen3 embedding model
        
    Returns:
        List of dictionaries mapping queries to meta-action-chains
    """
    queries_and_meta_action_chains = []

    thought_layer_reason_subgraph = tag.RetrievalFromStart(
        start_name=topic_entity_typeName, 
        start_attrs={"Type": "Ontology", "Layer": "Thought"}
    )
    
    for answer_typeName in answer_typeNames:
        thought_layer_reason_subgraph = pruning(
            thought_layer_reason_subgraph,
            answer_typeName
        )
        
        action_layer_start_nodes = tag.ThoughtNode2ActionNode(
            head_name=topic_entity_typeName, 
            head_attrs={"Type": "Ontology", "Layer": "Thought"}, 
            target_relation="instance_of", 
            tail_attrs={"Type": "Entity", "Layer": "Action"}
        )
        action_layer_end_nodes = tag.ThoughtNode2ActionNode(
            head_name=answer_typeName, 
            head_attrs={"Type": "Ontology", "Layer": "Thought"}, 
            target_relation="instance_of", 
            tail_attrs={"Type": "Entity", "Layer": "Action"}
        )
        action_layer_reason_subgraph = []
        for action_layer_start_node in action_layer_start_nodes:
            for action_layer_end_node in action_layer_end_nodes:
                start_node_name = action_layer_start_node["name"]
                start_node_attrs = action_layer_start_node["attributes"]
                end_node_name = action_layer_end_node["name"]
                end_node_attrs = action_layer_end_node["attributes"]
                action_layer_reason_subgraph.extend(
                    tag.RetrievalFromStart2End(
                        start_name=start_node_name,
                        start_attrs=start_node_attrs,
                        end_name=end_node_name,
                        end_attrs=end_node_attrs
                    )
                )
        
        similar_queries, action_layer_reason_subgraph = FilterActionChain(
            qwen3_model, 
            action_layer_reason_subgraph,
            question, 
            query_embeddings, 
            top_k=5
        )
        
        thought_layer_reason_subgraph = tag.ActionChain2ThoughtChain(action_layer_reason_subgraph)
        
        similar_meta_action_chains = MergeMetaActionChain(action_layer_reason_subgraph, thought_layer_reason_subgraph)
        
        if similar_queries != [] and similar_meta_action_chains != []:
            query_meta_action_chain_dict = {}
            for query, meta_action_chain in zip(similar_queries, similar_meta_action_chains):
                query_meta_action_chain_dict[query] = meta_action_chain
            queries_and_meta_action_chains.append(query_meta_action_chain_dict)

    return queries_and_meta_action_chains


def main():
    """Main function to run reasoning on WebQSP dataset."""
    config_path = "config.json"
    config = load_config(config_path)
    
    paths = config["paths"]
    parameters = config["parameters"]
    model_settings = config["model_settings"]
    sparql_endpoint = config["sparql"]["endpoint"]
    
    print("Loading models and data...")
    
    webqsp_test = load_dataset(paths["webqsp_test_path"])
    predict_type_name_prompt = load_prompt(paths["predict_type_name_prompt_path"])
    generate_meta_action_chain_prompt = load_prompt(paths["generate_meta_action_chain_prompt_path"])
    reasoning_prompt = load_prompt(paths["reasoning_prompt_path"])

    typename_embeddings = load_embeddings(paths["answer_type_name_embeddings_path"])
    query_embeddings = load_embeddings(paths["query_embeddings_path"])

    qwen3_model = SentenceTransformer(paths["qwen3_embedding_model_path"])
    fasttext_model = fasttext.load_model(paths["fasttext_embeddings_path"])
    llama3_1 = load_llama3_1(paths["llama3_1_model_path"])

    tag = ThoughtActionGraph.load_from_file(paths["tag_path"])
    tag.sparql_endpoint = sparql_endpoint
    
    print("Models and data loaded successfully.")
    
    result = []
    num = 0
    
    for data in tqdm(webqsp_test, desc="Processing questions"):
        num += 1
        print(f"Question ID: {data['QuestionId']}")
        question = data["ProcessedQuestion"]
        topic_entity_mid = data["Parses"][0]["TopicEntityMid"]
        topic_entity_name = data["Parses"][0]["TopicEntityName"]
        graph = data["graph"]
        
        queries_and_macs, macs_of_question, answers, filtered_graph = reason_by_TAG(
            llama3_1,
            question, 
            topic_entity_mid,
            topic_entity_name,
            graph, 
            typename_embeddings, 
            query_embeddings,
            tag,
            qwen3_model,
            fasttext_model,
            predict_type_name_prompt,
            generate_meta_action_chain_prompt,
            reasoning_prompt,
            tag_explore_breadth=parameters["tag_explore_breadth"],
            tag_explore_depth=parameters["tag_explore_depth"]
        )
                
        ground_truth_answers = [parse['Answers'] for parse in data['Parses']]
        result.append({
            "QuestionId": data["QuestionId"],
            "Question": question,
            "TopicEntityMid": topic_entity_mid,
            "Queries_and_MACs": queries_and_macs,
            "MACs_of_Question": macs_of_question,
            "Answers": answers,
            "Ground_Truth_Answers": ground_truth_answers,
            "Graph": filtered_graph
        })

        if num % 20 == 0:
            output_path = os.path.join(
                paths["answer_save_path"],
                f"llama3_1_8B_Reasoning_by_TAG_results.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            print(f"Saved intermediate results at step {num}")
        
    final_output_path = os.path.join(
        paths["answer_save_path"],
        "llama3_1_8B_Reasoning_by_TAG_results.json"
    )
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Final results saved to {final_output_path}")


if __name__ == "__main__":
    main()
