"""
Semantic dual-layer retrieval for Thought-Action Graph.

This module provides retrieval functionality for finding similar queries,
filtering action chains, and traversing the Thought-Action Graph.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import deque
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ThoughtActionGraph import ThoughtActionGraph
from utils import load_config, encode_queries, RetrievalSimilarQueries


def retrieval_from_start(graph: ThoughtActionGraph, 
                         start_name: str, 
                         start_attrs: Dict[str, Any]
                         ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Retrieve all complete paths starting from a specific node.
    
    This method finds all paths following the pattern:
    start_node → (start) → node1 → (next) → ... → node_n → (end) → end_node
    
    Args:
        graph: ThoughtActionGraph instance.
        start_name: Name of the start node.
        start_attrs: Attributes of the start node for precise matching.
        
    Returns:
        List of complete paths, where each path is a list of triples.
    """
    start_nodes = graph.find_nodes_by_name_and_attributes(start_name, start_attrs)
    if not start_nodes:
        return []

    all_complete_paths = []

    for start_node in start_nodes:
        start_related_triples = graph.find_triples_with_head_node(
            name=start_node["name"],
            attrs=start_node["attributes"]
        )
        first_hop_triples = [t for t in start_related_triples if t[1] == "start"]

        if not first_hop_triples:
            continue

        for first_hop in first_hop_triples:
            initial_path = [first_hop]
            start_node_id = graph._get_entity_id_by_data(start_node)
            first_tail_node = first_hop[2]
            first_tail_id = graph._get_entity_id_by_data(first_tail_node)

            if start_node_id is None or first_tail_id is None:
                continue
            initial_visited = {start_node_id, first_tail_id}

            graph._traverse_next_end(
                current_node=first_tail_node,
                current_path=initial_path,
                visited_ids=initial_visited,
                all_paths=all_complete_paths
            )

    return all_complete_paths


def prune_paths(path_list: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]], 
                target_tail_name: str
                ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Filter paths to keep only those ending with a specific node name.
    
    Args:
        path_list: List of paths to filter.
        target_tail_name: Target tail node name.
        
    Returns:
        List of paths ending with the target node name.
    """
    filtered_paths = []
    
    for path in path_list:
        if not path:
            continue
        
        last_triple = path[-1]
        tail_node = last_triple[2]
        
        if 'name' not in tail_node:
            continue
        
        if tail_node['name'] == target_tail_name:
            filtered_paths.append(path)
    
    return filtered_paths


def thought_node_to_action_node(graph: ThoughtActionGraph,
                                head_name: str,
                                head_attrs: Dict[str, Any],
                                target_relation: str,
                                tail_attrs: Dict[str, Any]
                                ) -> List[Dict[str, Any]]:
    """
    Find action nodes connected to thought nodes via a specific relation.
    
    Args:
        graph: ThoughtActionGraph instance.
        head_name: Name of the thought node.
        head_attrs: Attributes of the thought node.
        target_relation: Relationship type to follow.
        tail_attrs: Attributes to filter action nodes.
        
    Returns:
        List of matching action nodes.
    """
    matched_heads = graph.find_nodes_by_name_and_attributes(head_name, head_attrs)
    if not matched_heads:
        return []

    head_ids = []
    for head_node in matched_heads:
        head_id = graph._get_entity_id_by_data(head_node)
        if head_id:
            head_ids.append(head_id)

    candidate_tail_ids = set()
    for head_id in head_ids:
        head_relations = graph.get_relations(head_id, relation=target_relation)
        if target_relation not in head_relations:
            continue
        tail_ids_for_relation = head_relations[target_relation]
        candidate_tail_ids.update(tail_ids_for_relation)

    if not candidate_tail_ids:
        return []

    matched_tails = []
    for tail_id in candidate_tail_ids:
        tail_node = graph.get_entity_by_id(tail_id)
        if not tail_node:
            continue

        attr_match = True
        for req_attr, req_value in tail_attrs.items():
            if (req_attr not in tail_node["attributes"] or 
                tail_node["attributes"][req_attr] != req_value):
                attr_match = False
                break

        if attr_match:
            matched_tails.append(tail_node)

    unique_tails = []
    seen_tail_hashes = set()
    for tail in matched_tails:
        tail_hash = (tail["name"], tuple(sorted(tail["attributes"].items())))
        if tail_hash not in seen_tail_hashes:
            seen_tail_hashes.add(tail_hash)
            unique_tails.append(tail)

    return unique_tails


def action_node_to_thought_node(graph: ThoughtActionGraph,
                                tail_name: str,
                                tail_attrs: Dict[str, Any],
                                target_relation: str,
                                head_attrs: Dict[str, Any]
                                ) -> List[Dict[str, Any]]:
    """
    Find thought nodes connected to action nodes via a specific relation.
    
    Args:
        graph: ThoughtActionGraph instance.
        tail_name: Name of the action node.
        tail_attrs: Attributes of the action node.
        target_relation: Relationship type to follow.
        head_attrs: Attributes to filter thought nodes.
        
    Returns:
        List of matching thought nodes.
    """
    matched_tails = graph.find_nodes_by_name_and_attributes(tail_name, tail_attrs)
    if not matched_tails:
        return []

    tail_ids = []
    for tail_node in matched_tails:
        tail_id = graph._get_entity_id_by_data(tail_node)
        if tail_id:
            tail_ids.append(tail_id)

    candidate_head_ids = set()
    for head_id, relations in graph.relations.items():
        if target_relation not in relations:
            continue
        for tail_id in tail_ids:
            if tail_id in relations[target_relation]:
                candidate_head_ids.add(head_id)
                break

    if not candidate_head_ids:
        return []

    matched_heads = []
    for head_id in candidate_head_ids:
        head_node = graph.get_entity_by_id(head_id)
        if not head_node:
            continue

        attr_match = True
        for req_attr, req_value in head_attrs.items():
            if (req_attr not in head_node["attributes"] or 
                head_node["attributes"][req_attr] != req_value):
                attr_match = False
                break

        if attr_match:
            matched_heads.append(head_node)

    unique_heads = []
    seen_head_hashes = set()
    for head in matched_heads:
        head_hash = (head["name"], tuple(sorted(head["attributes"].items())))
        if head_hash not in seen_head_hashes:
            seen_head_hashes.add(head_hash)
            unique_heads.append(head)

    return unique_heads


def retrieval_from_start_to_end(graph: ThoughtActionGraph,
                                start_name: str,
                                start_attrs: Dict[str, Any],
                                end_name: str,
                                end_attrs: Dict[str, Any],
                                max_depth: int = 10
                                ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Find all paths from start node to end node using BFS.
    
    Args:
        graph: ThoughtActionGraph instance.
        start_name: Name of the start node.
        start_attrs: Attributes of the start node.
        end_name: Name of the end node.
        end_attrs: Attributes of the end node.
        max_depth: Maximum path depth to search.
        
    Returns:
        List of complete paths from start to end.
    """
    start_node_ids = set()
    for node in graph.find_nodes_by_name_and_attributes(start_name, start_attrs):
        node_id = graph._get_entity_id_by_data(node)
        if node_id:
            start_node_ids.add(node_id)
    
    end_node_ids = set()
    for node in graph.find_nodes_by_name_and_attributes(end_name, end_attrs):
        node_id = graph._get_entity_id_by_data(node)
        if node_id:
            end_node_ids.add(node_id)
    
    if not start_node_ids or not end_node_ids:
        return []

    all_valid_paths = []
    queue = deque()
    
    for start_id in start_node_ids:
        if start_id not in graph.relations:
            continue
            
        start_relations = graph.relations[start_id].get("start", [])
        for next_id in start_relations:
            if next_id not in graph.entities:
                continue
                
            path = [(start_id, "start", next_id)]
            visited = {start_id, next_id}
            queue.append((next_id, path, visited, 1))

    while queue:
        current_id, path, visited, depth = queue.popleft()
        
        if depth > max_depth:
            continue
            
        if current_id in graph.relations and "end" in graph.relations[current_id]:
            for end_id in graph.relations[current_id]["end"]:
                if end_id in end_node_ids:
                    complete_path = path + [(current_id, "end", end_id)]
                    all_valid_paths.append(complete_path)
        
        if current_id in graph.relations and "next" in graph.relations[current_id]:
            for next_id in graph.relations[current_id]["next"]:
                if next_id not in visited and next_id in graph.entities:
                    new_path = path + [(current_id, "next", next_id)]
                    new_visited = visited | {next_id}
                    queue.append((next_id, new_path, new_visited, depth + 1))

    result_paths = []
    for id_path in all_valid_paths:
        entity_path = []
        for head_id, rel, tail_id in id_path:
            head_entity = graph.entities[head_id]
            tail_entity = graph.entities[tail_id]
            entity_path.append((head_entity, rel, tail_entity))
        result_paths.append(entity_path)
    
    return result_paths


def filter_action_chains(Qwen3Model: SentenceTransformer, 
                         action_chains: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]],
                         text: str, 
                         embedding_dict: Dict[str, Any], 
                         top_k: int = 5
                         ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Filter action chains based on semantic similarity to a query text.
    
    Args:
        Qwen3Model: Qwen3 embedding model instance.
        action_chains: List of action chains to filter.
        text: Query text for similarity matching.
        embedding_dict: Dictionary of pre-computed embeddings.
        top_k: Number of top similar chains to return.
        
    Returns:
        List of top-k similar action chains.
    """
    queries = []
    for chain in action_chains:
        queries.append(chain[0][-1]["attributes"]["Question"])
    queries = list(set(queries))
    
    similar_results = [query[0] for query in RetrievalSimilarQueries(
        Qwen3Model, text, queries, embedding_dict, top_k
    )]
    
    similar_action_chains = []
    for chain in action_chains:
        if chain[0][-1]["attributes"]["Question"] in similar_results:
            similar_action_chains.append(chain)
            similar_results.remove(chain[0][-1]["attributes"]["Question"])
    
    return similar_action_chains


def action_chain_to_thought_chain(graph: ThoughtActionGraph,
                                  action_chains: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]
                                  ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Convert action chains to thought chains by mapping entities to ontologies.
    
    Args:
        graph: ThoughtActionGraph instance.
        action_chains: List of action chains to convert.
        
    Returns:
        List of thought chains.
    """
    thought_chains = []
    
    for action_chain in action_chains:
        thought_chain = []
        for triple in action_chain:
            head_entity, relation, tail_entity = triple
            
            head_ontologies = graph.find_triples_with_tail_node(
                name=head_entity["name"],
                attrs={"Type": "Ontology", "Layer": "Thought"}
            )
            
            tail_ontologies = graph.find_triples_with_tail_node(
                name=tail_entity["name"],
                attrs={"Type": "Ontology", "Layer": "Thought"}
            )
            
            if head_ontologies and tail_ontologies:
                for head_onto in head_ontologies:
                    for tail_onto in tail_ontologies:
                        thought_chain.append((head_onto[0], relation, tail_onto[2]))
        
        thought_chains.append(thought_chain)
    
    return thought_chains


def main():
    """
    Main function for demonstration and testing.
    """
    config = load_config("config.json")
    
    paths = config.get("paths", {})
    
    tag_path = paths.get("tag_path")
    
    print(f"Loading TAG from: {tag_path}")
    tag = ThoughtActionGraph.load_from_file(tag_path)
    
    print(f"TAG loaded successfully!")
    print(f"Number of entities: {len(tag.entities)}")
    print(f"Number of relations: {sum(len(rels) for rels in tag.relations.values())}")
    
    entity_counts = tag.count_entity_types()
    print(f"\nEntity type counts:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")
    
    relation_counts = tag.count_relation_types()
    print(f"\nRelation type counts:")
    for relation_type, count in relation_counts.items():
        print(f"  {relation_type}: {count}")


if __name__ == "__main__":
    main()
