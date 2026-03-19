import os
import re
import json
import pickle
import pandas as pd
import time
import fasttext
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch
from SPARQLWrapper import SPARQLWrapper, JSON


def load_llama3_1(model_path: str):
    """
    Load Llama-3.1 model from the specified path
    
    Args:
        model_path: Path to the Llama-3.1 model directory
        
    Returns:
        transformers pipeline object for text generation
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
    Generate text using Llama-3.1 model
    
    Args:
        pipeline: Llama-3.1 pipeline object
        prompt: Input prompt for text generation
        
    Returns:
        Generated text content from the model
    """
    messages = [
        {"role": "system", "content": "You are a knowledge graph expert in the field of natural language processing."},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=32768,
    )
    
    return outputs[0]["generated_text"][-1]["content"]


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON or Parquet file
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        List of dictionaries containing the dataset
    """
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
        dataset = df.to_dict(orient="records")
        return dataset


def load_embeddings(embeddings_path: str) -> Dict:
    """
    Load embeddings from pickle file
    
    Args:
        embeddings_path: Path to the embeddings file
        
    Returns:
        Dictionary containing embeddings
    """
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_prompt(prompt_path: str) -> str:
    """
    Load prompt from text file
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        String content of the prompt file
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


def execute_sparql(sparql_txt: str, endpoint: str) -> Optional[Dict]:
    """
    Execute SPARQL query against the endpoint
    
    Args:
        sparql_txt: SPARQL query string
        endpoint: SPARQL endpoint URL
        
    Returns:
        Query results as dictionary, or None if error occurs
    """
    try:
        sparql = SPARQLWrapper(endpoint)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(f"SPARQL execution error: {e}")
        return None


def extract_freebase_id(uri: str) -> str:
    """
    Extract Freebase ID from URI
    
    Args:
        uri: Freebase URI string
        
    Returns:
        Extracted Freebase ID
    """
    if uri.startswith("http://rdf.freebase.com/ns/"):
        return uri[len("http://rdf.freebase.com/ns/"):]
    return uri


def id2entity_name_or_type(entity_uri: str, endpoint: str) -> str:
    """
    Get entity name or type from Freebase ID using SPARQL
    
    Args:
        entity_uri: Entity URI
        endpoint: SPARQL endpoint URL
        
    Returns:
        Entity name or "UnName_Entity" if not found
    """
    entity_id = extract_freebase_id(entity_uri)
    sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
    query_txt = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query_txt)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']


def id2entity_name_en(entity_uri: str, endpoint: str) -> Optional[str]:
    """
    Get English entity name from Freebase ID using SPARQL
    
    Args:
        entity_uri: Entity URI
        endpoint: SPARQL endpoint URL
        
    Returns:
        English entity name or None if not found
    """
    entity_id = extract_freebase_id(entity_uri)
    sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
    query_txt = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query_txt)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for item in results["results"]["bindings"]:
        if item["tailEntity"].get("xml:lang") == "en":
            return item["tailEntity"]["value"]
    return None


def sparql2answer_en(sparql_txt: str, endpoint: str) -> Tuple[List[str], List[Optional[str]]]:
    """
    Execute SPARQL query and convert results to English answers
    
    Args:
        sparql_txt: SPARQL query string
        endpoint: SPARQL endpoint URL
        
    Returns:
        Tuple of (answer_ids, answer_names) where answer_names can be None
    """
    id_results = execute_sparql(sparql_txt, endpoint)
    answers_argument = []
    answers_en_name = []
    
    if id_results is None:
        return answers_argument, answers_en_name
    
    for id_result in id_results["results"]["bindings"]:
        if id_result["x"]["type"] != "uri":
            answers_argument.append(id_result["x"]["value"])
            answers_en_name.append(None)
        else:
            answer_uri = id_result["x"]["value"]
            id = extract_freebase_id(answer_uri)
            answer = id2entity_name_en(answer_uri, endpoint)
            answers_argument.append(id)
            answers_en_name.append(answer)
    
    return answers_argument, answers_en_name


def typeName(type_id: str, endpoint: str) -> Optional[str]:
    """
    Get entity type name from Freebase ID using SPARQL
    
    Args:
        type_id: Freebase type ID
        endpoint: SPARQL endpoint URL
        
    Returns:
        English type name or None if not found
    """
    sparql_typeName_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?typeName\nWHERE {{\n  ns:{typeID} ns:common.topic.notable_types ?type .\n  ?type ns:type.object.name ?typeName .\n}}""".format(typeID=type_id)
    typename = execute_sparql(sparql_typeName_txt, endpoint)
    
    if typename is None:
        return None
        
    typename_list = typename["results"]["bindings"]
    for typename in typename_list:
        if 'typeName' in typename and isinstance(typename['typeName'], dict):
            type_name_data = typename['typeName']
            if ('xml:lang' in type_name_data and 
                type_name_data['xml:lang'] == 'en' and 
                'value' in type_name_data):
                return type_name_data['value']
    return None


def typeName_all(type_id: str, endpoint: str) -> List[str]:
    """
    Get all entity type names from Freebase ID using SPARQL
    
    Args:
        type_id: Freebase type ID
        endpoint: SPARQL endpoint URL
        
    Returns:
        List of English type names (deduplicated and ordered)
    """
    sparql_typeName_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?entityType ?typeName\nWHERE {{\n<http://rdf.freebase.com/ns/{typeID}> ns:type.object.type ?entityType .\n?entityType ns:type.object.name ?typeName .}}""".format(typeID=type_id)
    typename = execute_sparql(sparql_typeName_txt, endpoint)
    
    if typename is None:
        return []
    
    typename_all = []
    typename_list = typename["results"]["bindings"]
    for typename in typename_list:
        if 'typeName' in typename and isinstance(typename['typeName'], dict):
            type_name_data = typename['typeName']
            if ('xml:lang' in type_name_data and 
                type_name_data['xml:lang'] == 'en' and 
                'value' in type_name_data):
                typename_all.append(type_name_data['value'])
    
    typename_all = list(dict.fromkeys(typename_all))
    return typename_all


def _FasttextEmbedding(fasttext_model, text: str) -> List[float]:
    """
    Generate FastText embedding for text
    
    Args:
        fasttext_model: FastText model instance
        text: Input text
        
    Returns:
        Embedding vector as list of floats
    """
    embedding = fasttext_model.get_sentence_vector(text.replace("\n", " "))
    return embedding


def cosine_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        array1: First vector
        array2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))


def GraphFilter(embedding_model, meta_action_chains: List[str], 
                graph: List[List[str]], number_of_triples: int) -> List[List[str]]:
    """
    Filter graph triples based on semantic similarity with meta action chains
    
    Args:
        embedding_model: FastText model instance
        meta_action_chains: List of meta action chain strings
        graph: List of triples [head, relation, tail]
        number_of_triples: Maximum number of triples to select per similarity type
        
    Returns:
        Deduplicated list of selected triples
    """
    topic_entity = []
    triple_pattern = []
    pattern = r'WHERE_TRI_PATTERN\(([^)]+)\)'
    
    for mac in meta_action_chains:
        mac_list = mac.split("-->")
        topic_entity.append(mac_list[0])
        for action in mac_list:
            match = re.search(pattern, action)
            if match:
                triple_pattern.append(match.group(1))
    
    selected_triples = []
    
    if topic_entity and graph:
        topic_embeddings = [_FasttextEmbedding(embedding_model, te) for te in topic_entity]
        avg_topic_embedding = np.mean(topic_embeddings, axis=0)
        
        head_similarities = []
        for idx, triple in enumerate(graph):
            head_entity = triple[0]
            head_embedding = _FasttextEmbedding(embedding_model, head_entity)
            similarity = cosine_similarity(avg_topic_embedding, head_embedding)
            head_similarities.append((idx, similarity))

        head_similarities.sort(key=lambda x: x[1], reverse=True)
        top_head_indices = set(idx for idx, _ in head_similarities[:number_of_triples])
        
        tail_similarities = []
        for idx, triple in enumerate(graph):
            tail_entity = triple[2]
            tail_embedding = _FasttextEmbedding(embedding_model, tail_entity)
            similarity = cosine_similarity(avg_topic_embedding, tail_embedding)
            tail_similarities.append((idx, similarity))

        tail_similarities.sort(key=lambda x: x[1], reverse=True)
        top_tail_indices = set(idx for idx, _ in tail_similarities[:number_of_triples])
        
        for idx in top_head_indices.union(top_tail_indices):
            selected_triples.append(graph[idx])

    if triple_pattern and graph:
        for pattern_text in triple_pattern:
            pattern_embedding = _FasttextEmbedding(embedding_model, pattern_text)
            rel_similarities = []
            
            for idx, triple in enumerate(graph):
                relation = triple[1]
                rel_embedding = _FasttextEmbedding(embedding_model, relation)
                similarity = cosine_similarity(pattern_embedding, rel_embedding)
                rel_similarities.append((idx, similarity))
            
            rel_similarities.sort(key=lambda x: x[1], reverse=True)
            top_rel_indices = [idx for idx, _ in rel_similarities[:number_of_triples]]
            
            for idx in top_rel_indices:
                selected_triples.append(graph[idx])

    unique_triples = []
    seen = set()
    for triple in selected_triples:
        triple_tuple = tuple(triple)
        if triple_tuple not in seen:
            seen.add(triple_tuple)
            unique_triples.append(triple)
    
    return unique_triples


def filter_answer(reasoning_result: str) -> List[List[str]]:
    """
    Filter and extract answers from reasoning result
    
    Args:
        reasoning_result: Model output string containing answers
        
    Returns:
        List of answer lists
    """
    answer = re.findall(r'<ANSWER>(.*?)</ANSWER>', reasoning_result)[0]
    filtered_answers = []
    
    if ("[" in answer) and ("]" in answer):
        answer_list = re.findall(r'\[(.*?)\]', answer)[0].split(",")
        answer_list = [a.strip() for a in answer_list]
        answer_list = [a.replace("\"", "") for a in answer_list]
        filtered_answers.append(answer_list)
    else:
        filtered_answers.append([answer])
    
    return filtered_answers


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
