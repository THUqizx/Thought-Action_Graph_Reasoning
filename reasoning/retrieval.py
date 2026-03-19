import os
import re
import pickle
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Set
from tqdm import tqdm
import collections
from sentence_transformers import SentenceTransformer


def RetrievalFromStart(self, 
                       start_name: str, 
                       start_attrs: Dict[str, Any]
                       ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Retrieve all complete paths starting from a specified node.
    
    This method finds all paths with structure:
    Start Node → (start) → Node1 → (next) → ... → NodeN → (end) → End Node
    
    Args:
        start_name: Name of the start node
        start_attrs: Attribute dictionary of the start node (for precise matching)
        
    Returns:
        List of complete paths, where each path is a list of triples.
        Each triple has format: (source_node_data, relation_type, target_node_data)
        Returns empty list if no matching paths found.
    """
    start_nodes = self.find_nodes_by_name_and_attributes(
        name=start_name,
        attrs=start_attrs
    )
    if not start_nodes:
        return []

    all_complete_paths = []

    for start_node in start_nodes:
        start_related_triples = self.find_triples_with_head_node(
            name=start_node["name"],
            attrs=start_node["attributes"]
        )
        first_hop_triples = [t for t in start_related_triples if t[1] == "start"]

        if not first_hop_triples:
            continue

        for first_hop in first_hop_triples:
            initial_path = [first_hop]
            start_node_id = self._get_entity_id_by_data(start_node)
            first_tail_node = first_hop[2]
            first_tail_id = self._get_entity_id_by_data(first_tail_node)

            if start_node_id is None or first_tail_id is None:
                continue
            initial_visited = {start_node_id, first_tail_id}

            self._traverse_next_end(
                current_node=first_tail_node,
                current_path=initial_path,
                visited_ids=initial_visited,
                all_paths=all_complete_paths
            )

    return all_complete_paths


def pruning(
        path_list: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]],
        target_tail_name: str
        ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Filter paths where the last triple's tail node name matches the target name.
    
    Args:
        path_list: List of paths, where each path is a list of triples.
                   Each triple format: (head_node_dict, relation_string, tail_node_dict)
                   Node dict must contain 'name' key (e.g., {'name': 'Algeria', 'attributes': ...})
        target_tail_name: Target tail node name (e.g., 'Algeria')
        
    Returns:
        List of paths matching the condition, or empty list if none found.
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


def ThoughtNode2ActionNode(
    self,
    head_name: str,
    head_attrs: Dict[str, Any],
    target_relation: str,
    tail_attrs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find all tail nodes based on a known head node (name + attributes) and a specific relation,
    filtered by specific attribute conditions.
    
    Args:
        head_name: Head node name (for initial node location)
        head_attrs: Head node attribute dictionary (for precise matching)
        target_relation: Target relation type (e.g., "belongs_to", "has", "professor")
        tail_attrs: Tail node attribute filter conditions (key=attribute name, value=attribute value, exact match required)

    Returns:
        List of tail node dictionaries matching all conditions; empty list if none found.
    """
    matched_heads = self.find_nodes_by_name_and_attributes(head_name, head_attrs)
    if not matched_heads:
        return []

    head_ids = []
    for head_node in matched_heads:
        head_id = self._get_entity_id_by_data(head_node)
        if head_id:
            head_ids.append(head_id)

    candidate_tail_ids = set()
    for head_id in head_ids:
        head_relations = self.get_relations(head_id, relation=target_relation)
        if target_relation not in head_relations:
            continue
        tail_ids_for_relation = head_relations[target_relation]
        candidate_tail_ids.update(tail_ids_for_relation)

    if not candidate_tail_ids:
        return []

    matched_tails = []
    for tail_id in candidate_tail_ids:
        tail_node = self.get_entity_by_id(tail_id)
        if not tail_node:
            continue

        attr_match = True
        for req_attr, req_value in tail_attrs.items():
            if (req_attr not in tail_node["attributes"]) or (tail_node["attributes"][req_attr] != req_value):
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


def _ActionNode2ThoughtNode(
    self,
    tail_name: str,
    tail_attrs: Dict[str, Any],
    target_relation: str,
    head_attrs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find all head nodes based on a known tail node (name + attributes) and a specific relation,
    filtered by specific attribute conditions.
    
    Args:
        tail_name: Tail node name (for initial node location)
        tail_attrs: Tail node attribute dictionary (for precise matching)
        target_relation: Target relation type (e.g., "belongs_to", "has", "professor")
        head_attrs: Head node attribute filter conditions (key=attribute name, value=attribute value, exact match required)

    Returns:
        List of head node dictionaries matching all conditions; empty list if none found.
    """
    matched_tails = self.find_nodes_by_name_and_attributes(tail_name, tail_attrs)
    if not matched_tails:
        return []

    tail_ids = []
    for tail_node in matched_tails:
        tail_id = self._get_entity_id_by_data(tail_node)
        if tail_id:
            tail_ids.append(tail_id)

    candidate_head_ids = set()
    for head_id, relations in self.relations.items():
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
        head_node = self.get_entity_by_id(head_id)
        if not head_node:
            continue

        attr_match = True
        for req_attr, req_value in head_attrs.items():
            if (req_attr not in head_node["attributes"]) or (head_node["attributes"][req_attr] != req_value):
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


def RetrievalFromStart2End(self,
                           start_name: str,
                           start_attrs: Dict[str, Any],
                           end_name: str,
                           end_attrs: Dict[str, Any],
                           max_depth: int = 10
                           ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Find all complete paths from a start node to an end node.
    Path structure must be: Start → start → [multiple next] → end → End
    
    Args:
        start_name: Start node name
        start_attrs: Start node attribute dictionary (key=attribute name, value=attribute value)
        end_name: End node name
        end_attrs: End node attribute dictionary (key=attribute name, value=attribute value)
        max_depth: Maximum depth to prevent infinite recursion
        
    Returns:
        List of all valid complete paths. Each path is a list of triples.
        Example: [
            [
                (start_entity, "start", node_A),
                (node_A, "next", node_B),
                (node_B, "end", end_entity)
            ],
            ...  # other paths
        ]
        Returns empty list if no matching paths found.
    """
    start_node_ids = set()
    for node in self.find_nodes_by_name_and_attributes(start_name, start_attrs):
        node_id = self._get_entity_id_by_data(node)
        if node_id:
            start_node_ids.add(node_id)
    
    end_node_ids = set()
    for node in self.find_nodes_by_name_and_attributes(end_name, end_attrs):
        node_id = self._get_entity_id_by_data(node)
        if node_id:
            end_node_ids.add(node_id)
    
    if not start_node_ids or not end_node_ids:
        return []

    all_valid_paths = []
    queue = collections.deque()
    
    for start_id in start_node_ids:
        if start_id not in self.relations:
            continue
            
        start_relations = self.relations[start_id].get("start", [])
        for next_id in start_relations:
            if next_id not in self.entities:
                continue
                
            path = [(start_id, "start", next_id)]
            visited = {start_id, next_id}
            queue.append((next_id, path, visited, 1))

    while queue:
        current_id, path, visited, depth = queue.popleft()
        
        if depth > max_depth:
            continue
            
        if current_id in self.relations and "end" in self.relations[current_id]:
            for end_id in self.relations[current_id]["end"]:
                if end_id in end_node_ids:
                    complete_path = path + [(current_id, "end", end_id)]
                    all_valid_paths.append(complete_path)
        
        if current_id in self.relations and "next" in self.relations[current_id]:
            for next_id in self.relations[current_id]["next"]:
                if next_id not in visited and next_id in self.entities:
                    new_path = path + [(current_id, "next", next_id)]
                    new_visited = visited | {next_id}
                    queue.append((next_id, new_path, new_visited, depth + 1))

    result_paths = []
    for id_path in all_valid_paths:
        entity_path = []
        for head_id, rel, tail_id in id_path:
            head_entity = self.entities[head_id]
            tail_entity = self.entities[tail_id]
            entity_path.append((head_entity, rel, tail_entity))
        result_paths.append(entity_path)
    
    return result_paths


def _Embedding_by_Qwen3(Qwen3Model, queries):
    """
    Generate embeddings for queries using Qwen3 model.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        queries: List of query strings
        
    Returns:
        List of embeddings
    """
    query_embeddings = Qwen3Model.encode(queries, prompt_name="query")
    return query_embeddings 


def encode_queries(Qwen3Model, queries, save_path, batch_size=10):
    """
    Encode a list of texts using Qwen3-Embedding model and save to file.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        queries: List of text strings, each element is a text
        save_path: File path to save the encoded results
    """
    queries = list(set(queries))
    total_count = len(queries)
    print(f"Total {total_count} unique texts, processing in batches of {batch_size}...")
    
    embedding_dict = {}
    
    for i in range(0, total_count, batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}: {len(batch_queries)} texts")
        
        batch_embeddings = _Embedding_by_Qwen3(Qwen3Model, batch_queries)
        
        for text, embedding in zip(batch_queries, batch_embeddings):
            embedding_dict[text] = embedding
    
    with open(save_path, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    print(f"All texts encoded, saved to {save_path}, total {len(embedding_dict)} texts")
    return embedding_dict


def RetrievalSimilarQueries(Qwen3Model, text, embedding_dict, top_k=5):
    """
    Find the top-K most similar queries to the input text.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        text: Input text
        embedding_dict: Dictionary of saved encoded results
        top_k: Number of most similar texts to return
        
    Returns:
        List of top-K similar texts with their similarity scores
    """
    text_embedding = _Embedding_by_Qwen3(Qwen3Model, text)
    
    similarities = []
    for query, embedding in embedding_dict.items():
        similarity = Qwen3Model.similarity(embedding, text_embedding)
        similarities.append((query, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_similar = similarities[:top_k]
    
    return top_k_similar


def encode_WebQSP_question(Qwen3Model, webqsp_path, save_path, batch_size):
    """
    Encode WebQSP questions using Qwen3-Embedding model.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        webqsp_path: Path to WebQSP dataset file
        save_path: Path to save encoded results
        batch_size: Batch size for encoding
    """
    with open(webqsp_path, "r", encoding="utf-8") as f:
        webqsp = json.load(f)
    queries = []
    for data in webqsp["Questions"]:
        queries.append(data["ProcessedQuestion"])
    encode_queries(Qwen3Model, queries, save_path, batch_size)


def encode_typeName_question(Qwen3Model, typeName_path, save_path, batch_size):
    """
    Encode answer type names using Qwen3-Embedding model.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        typeName_path: Path to answer type names file
        save_path: Path to save encoded results
        batch_size: Batch size for encoding
    """
    with open(typeName_path, "r", encoding="utf-8") as f:
        typeName = [line.strip() for line in f]
    encode_queries(Qwen3Model, typeName, save_path, batch_size)


def FilterActionChain(Qwen3Model, ActionChains, text, embedding_dict, top_k=5):
    """
    Filter Action Chains based on semantic similarity and return top-K results.
    
    Args:
        Qwen3Model: Qwen3-embedding model instance
        ActionChains: Action Chains from Action Layer
        text: Input text
        embedding_dict: Dictionary of queries and their embeddings
        top_k: Number of most similar texts to return
        
    Returns:
        Top-K similar Action Chains
    """
    queries = []
    for chain in ActionChains:
        queries.append(chain[0][-1]["attributes"]["Question"])
    queries = list(set(queries))
    similar_results = [query[0] for query in RetrievalSimilarQueries(Qwen3Model, text, embedding_dict, top_k)]
    
    similar_ActionChain = {}
    for chain in ActionChains:
        if chain[0][-1]["attributes"]["Question"] in similar_results:
            similar_ActionChain[chain[0][-1]["attributes"]["Question"]] = chain
            similar_results.remove(chain[0][-1]["attributes"]["Question"])
    similar_queries = list(similar_ActionChain.keys())
    similar_ActionChain = [similar_ActionChain[q] for q in similar_queries]
    return similar_queries, similar_ActionChain


def ActionChain2ThoughtChain(
        self, 
        ActionChains: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]
        ) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]:
    """
    Convert Action Chains to Thought Chains by replacing each entity with its 
    corresponding head entity via "instance_of" or "has_parameter" relation.
    
    Args:
        ActionChains: List of Action Chains, each chain is a list of triples
        
    Returns:
        List of converted Thought Chains with the same format
    """
    ThoughtChains = []
    for chain in ActionChains:
        chain_temp = []
        for triple in chain:
            head_entity, relation, tail_entity = triple
            
            head_as_tail_triples = self.find_head_nodes_by_tail_and_relation(
                head_entity['name'], head_entity['attributes'], relation="instance_of"
            )
            if not head_as_tail_triples:
                head_as_tail_triples = self.find_head_nodes_by_tail_and_relation(
                    head_entity['name'], head_entity['attributes'], relation="has_parameter"
                )
            new_head_entity = head_as_tail_triples[0]
         
            tail_as_tail_triples = self.find_head_nodes_by_tail_and_relation(
                tail_entity['name'], tail_entity['attributes'], relation="instance_of"
            )
            if not tail_as_tail_triples:
                tail_as_tail_triples = self.find_head_nodes_by_tail_and_relation(
                    tail_entity['name'], tail_entity['attributes'], relation="has_parameter"
                )
            new_tail_entity = tail_as_tail_triples[0]
            
            new_triple = (new_head_entity, relation, new_tail_entity)
            chain_temp.append(new_triple)
        ThoughtChains.append(chain_temp)
    return ThoughtChains


def MergeMetaActionChain(ActionChains, ThoughtChains):
    """
    Merge Action Chains and Thought Chains to form final Meta Action Chains.
    
    Args:
        ActionChains: Action Chains from Action Layer
        ThoughtChains: Thought Chains from Thought Layer
        
    Returns:
        List of merged Meta-Action-Chains as strings
    """
    MetaActionChains = []
    for ActionChain, ThoughtChain in zip(ActionChains, ThoughtChains):
        MetaActionChain = []
        for step, (action_triple, thought_triple) in enumerate(zip(ActionChain, ThoughtChain)):
            if step == 0:
                assert action_triple[1] == "start"
                assert thought_triple[1] == "start"
                MetaActionChain.append(action_triple[0]["name"])
            elif step == len(ActionChain) - 1:
                assert action_triple[1] == "end"
                assert thought_triple[1] == "end"
                MetaActionChain.append(thought_triple[0]["name"]+"("+action_triple[0]["name"]+")")
                MetaActionChain.append(action_triple[-1]["name"])
            else:
                assert action_triple[1] == "next"
                assert thought_triple[1] == "next"
                MetaActionChain.append(thought_triple[0]["name"]+"("+action_triple[0]["name"]+")")
        MetaActionChains.append(MetaActionChain)
    MetaActionChains = ["-->".join(mac) for mac in MetaActionChains]
    return MetaActionChains
