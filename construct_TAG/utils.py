"""
Utility functions for Thought-Action Graph construction.

This module provides various utility functions for:
- SPARQL queries to Freebase
- Entity and type name retrieval
- Query encoding using Qwen3 embedding model
- MAC (Meta-Action-Chain) extraction
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
import pickle


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path to save the configuration JSON file.
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def execute_sparql(sparql_txt: str, sparql_endpoint: str) -> Optional[Dict[str, Any]]:
    """
    Execute a SPARQL query against the Freebase endpoint.
    
    Args:
        sparql_txt: SPARQL query string.
        sparql_endpoint: URL of the SPARQL endpoint.
        
    Returns:
        Query results as dictionary, or None if query fails.
    """
    try:
        sparql = SPARQLWrapper(sparql_endpoint)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(f"SPARQL query failed: {e}")
        return None


def extract_freebase_id(uri: str) -> str:
    """
    Extract Freebase ID from a URI.
    
    Args:
        uri: Freebase URI string.
        
    Returns:
        Extracted Freebase ID.
    """
    if uri.startswith("http://rdf.freebase.com/ns/"):
        return uri[len("http://rdf.freebase.com/ns/"):]
    return uri


def entityName(entity_uri: str, sparql_endpoint: str) -> str:
    """
    Get the English name of a Freebase entity.
    
    Args:
        entity_uri: Freebase entity URI.
        sparql_endpoint: URL of the SPARQL endpoint.
        
    Returns:
        English name of the entity, or "UnName_Entity" if not found.
    """
    entity_id = extract_freebase_id(entity_uri)
    query_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {{
  {{ ?entity ns:type.object.name ?tailEntity . FILTER(?entity = ns:{entity_id}) }}
  UNION
  {{ ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity . FILTER(?entity = ns:{entity_id}) }}
}}"""
    
    results = execute_sparql(query_txt, sparql_endpoint)
    if not results:
        return "UnName_Entity"
    
    entity_name_list = results["results"]["bindings"]
    for entity_name in entity_name_list:
        if 'tailEntity' in entity_name and isinstance(entity_name['tailEntity'], dict):
            tail_entity_data = entity_name['tailEntity']
            if ('xml:lang' in tail_entity_data and 
                tail_entity_data['xml:lang'] == 'en' and 
                'value' in tail_entity_data):
                return tail_entity_data['value']
    return "UnName_Entity"


def typeName(type_id: str, sparql_endpoint: str) -> Optional[str]:
    """
    Get the English name of a Freebase type.
    
    Args:
        type_id: Freebase type ID.
        sparql_endpoint: URL of the SPARQL endpoint.
        
    Returns:
        English name of the type, or None if not found.
    """
    sparql_typeName_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?typeName
WHERE {{
  ns:{type_id} ns:common.topic.notable_types ?type .
  ?type ns:type.object.name ?typeName .
}}"""
    
    results = execute_sparql(sparql_typeName_txt, sparql_endpoint)
    if not results:
        return None
    
    typename_list = results["results"]["bindings"]
    for typename in typename_list:
        if 'typeName' in typename and isinstance(typename['typeName'], dict):
            type_name_data = typename['typeName']
            if ('xml:lang' in type_name_data and 
                type_name_data['xml:lang'] == 'en' and 
                'value' in type_name_data):
                return type_name_data['value']
    return None


def typeName_all(type_id: str, sparql_endpoint: str) -> List[str]:
    """
    Get all English names of a Freebase type.
    
    Args:
        type_id: Freebase type ID.
        sparql_endpoint: URL of the SPARQL endpoint.
        
    Returns:
        List of English names of the type.
    """
    sparql_typeName_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?entityType ?typeName
WHERE {{
  <http://rdf.freebase.com/ns/{type_id}> ns:type.object.type ?entityType .
  ?entityType ns:type.object.name ?typeName .
}}"""
    
    results = execute_sparql(sparql_typeName_txt, sparql_endpoint)
    if not results:
        return []
    
    typename_list = results["results"]["bindings"]
    typename_all = []
    for typename in typename_list:
        if 'typeName' in typename and isinstance(typename['typeName'], dict):
            type_name_data = typename['typeName']
            if ('xml:lang' in type_name_data and 
                type_name_data['xml:lang'] == 'en' and 
                'value' in type_name_data):
                typename_all.append(type_name_data['value'])
    
    typename_all = list(dict.fromkeys(typename_all))
    return typename_all


def encode_queries(Qwen3Model, queries: List[str], batch_size: int = 10) -> Dict[str, Any]:
    """
    Encode a list of queries using the Qwen3 embedding model.
    
    Args:
        Qwen3Model: Qwen3 embedding model instance.
        queries: List of query strings to encode.
        batch_size: Number of queries to process in each batch.
        
    Returns:
        Dictionary mapping each query to its embedding.
    """
    def _embed_by_qwen3(model, query_list):
        return model.encode(query_list, prompt_name="query")
    
    queries = list(set(queries))
    total_count = len(queries)
    print(f"Total {total_count} unique texts, processing in batches of {batch_size}...")
    
    embedding_dict = {}
    
    for i in range(0, total_count, batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}: {len(batch_queries)} texts")
        
        batch_embeddings = _embed_by_qwen3(Qwen3Model, batch_queries)
        
        for text, embedding in zip(batch_queries, batch_embeddings):
            embedding_dict[text] = embedding
    
    return embedding_dict


def extract_mac_content(text: str) -> List[str]:
    """
    Extract content between <MAC> and </MAC> tags.
    
    Args:
        text: Input text string.
        
    Returns:
        List of content found between MAC tags.
    """
    pattern = r'<MAC>(.*?)</MAC>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def sparql_preprocess(sparql: str) -> str:
    """
    Remove SPARQL prefix declarations from a query.
    
    Args:
        sparql: SPARQL query string with prefixes.
        
    Returns:
        SPARQL query string without prefixes.
    """
    prefix_string = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \n"
    return sparql.replace(prefix_string, "")


def save_to_pickle(data: Any, save_path: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save.
        save_path: Path to save the pickle file.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(load_path: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        load_path: Path to the pickle file.
        
    Returns:
        Loaded data.
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)
