import openai
import os
import re
from typing import List, Dict, Optional
import json
import pandas as pd
import time
import pickle
from SPARQLWrapper import SPARQLWrapper, JSON
import torch
import numpy as np
import fasttext


class OpenAIClient:
    """OpenAI API client with retry mechanism for robust API calls."""
    
    def __init__(self, api_key: Optional[str] = None, max_attempts: int = 5, delay: int = 30):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
            max_attempts: Maximum number of retry attempts.
            delay: Delay between retries in seconds.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and OPENAI_API_KEY environment variable is not set")
        
        self.client = openai.OpenAI(api_key=api_key, base_url="")
        self.max_attempts = max_attempts
        self.delay = delay
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Call ChatCompletion API for对话.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys.
            model: Model name to use (default: gpt-4o-mini).
            temperature: Controls randomness (0-1).
            max_tokens: Maximum tokens to generate.
            stream: Whether to use streaming mode.
            
        Returns:
            Generated text response from API.
            
        Raises:
            Exception: If API connection fails, rate limit exceeded, or other API errors.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                return response.choices[0].message.content
                
        except openai.APIConnectionError as e:
            raise Exception(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI API rate limit exceeded: {e}")
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simplified text generation function with retry mechanism.
        
        Args:
            prompt: User input prompt text.
            model: Model name to use.
            system_message: Optional system message to set context.
            **kwargs: Additional arguments for chat_completion.
            
        Returns:
            Generated text response from API.
            
        Raises:
            Exception: If maximum retry attempts reached.
        """
        attempt = 1
        while attempt <= self.max_attempts:
            try:
                messages = []
                
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                messages.append({"role": "user", "content": prompt})
                
                return self.chat_completion(messages, model=model, **kwargs)
            except:
                attempt += 1
                if attempt <= self.max_attempts:
                    time.sleep(self.delay)
                else:
                    raise Exception("Maximum retry attempts reached, operation failed.")


def load_llama3_1(model_path: str):
    """
    Load LLaMA3.1 model and create text generation pipeline.
    
    Args:
        model_path: Path to the LLaMA3.1 model directory.
        
    Returns:
        transformers pipeline for text generation.
    """
    import transformers
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def llama3_1_generate(pipeline, prompt: str) -> str:
    """
    Generate text using LLaMA3.1 pipeline.
    
    Args:
        pipeline: LLaMA3.1 text generation pipeline.
        prompt: Input prompt for text generation.
        
    Returns:
        Generated text content from the model response.
    """
    import transformers
    
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
    Load dataset from JSON or Parquet file.
    
    Args:
        dataset_path: Path to the dataset file (.json or .parquet).
        
    Returns:
        Dataset as a list of dictionaries.
        
    Raises:
        ValueError: If file format is not supported.
    """
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
    elif dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
        dataset = df.to_dict(orient="records")
        return dataset
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}. Supported formats: .json, .parquet")


def load_embeddings(embeddings_path: str) -> any:
    """
    Load embeddings from pickle file.
    
    Args:
        embeddings_path: Path to the pickle file containing embeddings.
        
    Returns:
        Loaded embeddings object.
    """
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_prompt(prompt_path: str) -> str:
    """
    Load prompt template from text file.
    
    Args:
        prompt_path: Path to the prompt text file.
        
    Returns:
        Prompt template string.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


def execute_sparql(sparql_txt: str, sparql_endpoint: str) -> Optional[Dict]:
    """
    Execute SPARQL query on the endpoint.
    
    Args:
        sparql_txt: SPARQL query string.
        sparql_endpoint: SPARQL endpoint URL.
        
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
    Extract Freebase ID from URI.
    
    Args:
        uri: Freebase URI string.
        
    Returns:
        Extracted Freebase ID.
    """
    if uri.startswith("http://rdf.freebase.com/ns/"):
        return uri[len("http://rdf.freebase.com/ns/"):]
    return uri


def get_entity_name(entity_uri: str, sparql_endpoint: str) -> str:
    """
    Get English name of an entity from Freebase.
    
    Args:
        entity_uri: Freebase entity URI.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        Entity name in English, or "UnName_Entity" if not found.
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
    if results is None:
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


def get_type_name(type_id: str, sparql_endpoint: str) -> Optional[str]:
    """
    Get English name of a type from Freebase.
    
    Args:
        type_id: Freebase type ID.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        Type name in English, or None if not found.
    """
    sparql_typeName_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?typeName
WHERE {{
  ns:{type_id} ns:common.topic.notable_types ?type .
  ?type ns:type.object.name ?typeName .
}}"""
    
    results = execute_sparql(sparql_typeName_txt, sparql_endpoint)
    if results is None:
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


def get_type_name_all(type_id: str, sparql_endpoint: str) -> List[str]:
    """
    Get all English names of types for a given type ID from Freebase.
    
    Args:
        type_id: Freebase type ID.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        List of type names in English (with duplicates removed while preserving order).
    """
    sparql_typeName_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?entityType ?typeName
WHERE {{
  <http://rdf.freebase.com/ns/{type_id}> ns:type.object.type ?entityType .
  ?entityType ns:type.object.name ?typeName .
}}"""
    
    results = execute_sparql(sparql_typeName_txt, sparql_endpoint)
    if results is None:
        return []
    
    typename_all = []
    typename_list = results["results"]["bindings"]
    for typename in typename_list:
        if 'typeName' in typename and isinstance(typename['typeName'], dict):
            type_name_data = typename['typeName']
            if ('xml:lang' in type_name_data and 
                type_name_data['xml:lang'] == 'en' and 
                'value' in type_name_data):
                typename_all.append(type_name_data['value'])
    
    typename_all = list(dict.fromkeys(typename_all))
    return typename_all


def get_entity_name_or_type(entity_uri: str, sparql_endpoint: str) -> str:
    """
    Get entity name or type from Freebase.
    
    Args:
        entity_uri: Freebase entity URI.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        Entity name or "UnName_Entity" if not found.
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
    if results is None or len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']


def get_entity_name_en(entity_uri: str, sparql_endpoint: str) -> Optional[str]:
    """
    Get English name of an entity from Freebase.
    
    Args:
        entity_uri: Freebase entity URI.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        English entity name, or None if not found or not in English.
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
    if results is None:
        return None
    
    for item in results["results"]["bindings"]:
        if item["tailEntity"].get("xml:lang") == "en":
            return item["tailEntity"]["value"]
    return None


def sparql_to_answers_en(sparql_txt: str, sparql_endpoint: str) -> tuple:
    """
    Execute SPARQL query and convert results to English answers.
    
    Args:
        sparql_txt: SPARQL query string.
        sparql_endpoint: SPARQL endpoint URL.
        
    Returns:
        Tuple of (answers_argument, answers_en_name) where:
        - answers_argument: List of answer IDs or values
        - answers_en_name: List of English names (or None)
    """
    id_results = execute_sparql(sparql_txt, sparql_endpoint)
    if id_results is None:
        return [], []
    
    answers_argument = []
    answers_en_name = []
    
    for id_result in id_results["results"]["bindings"]:
        if id_result["x"]["type"] != "uri":
            answers_argument.append(id_result["x"]["value"])
            answers_en_name.append(None)
        else:
            answer_uri = id_result["x"]["value"]
            answer_id = extract_freebase_id(answer_uri)
            answer_name = get_entity_name_en(answer_uri, sparql_endpoint)
            answers_argument.append(answer_id)
            answers_en_name.append(answer_name)
    
    return answers_argument, answers_en_name


def get_fasttext_embedding(fasttext_model, text: str) -> List[float]:
    """
    Get FastText embedding for text.
    
    Args:
        fasttext_model: Loaded FastText model.
        text: Input text string.
        
    Returns:
        Embedding vector as list of floats.
    """
    embedding = fasttext_model.get_sentence_vector(text.replace("\n", " "))
    return embedding


def cosine_similarity(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        array1: First vector.
        array2: Second vector.
        
    Returns:
        Cosine similarity value between -1 and 1.
    """
    return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))


def filter_graph_by_similarity(
    fasttext_model,
    meta_action_chains: List[str],
    graph: List[List[str]],
    number_of_triples: int
) -> List[List[str]]:
    """
    Filter graph triples based on similarity to meta action chains.
    
    Args:
        fasttext_model: Loaded FastText model for embeddings.
        meta_action_chains: List of meta action chain strings.
        graph: List of triples [head_entity, relation, tail_entity].
        number_of_triples: Maximum number of triples to select per similarity category.
        
    Returns:
        Filtered and deduplicated list of graph triples.
    """
    pattern = r'WHERE_TRI_PATTERN\(([^)]+)\)'
    
    topic_entity = []
    triple_pattern = []
    for mac in meta_action_chains:
        mac_list = mac.split("-->")
        topic_entity.append(mac_list[0])
        for action in mac_list:
            match = re.search(pattern, action)
            if match:
                triple_pattern.append(match.group(1))

    selected_triples = []
    
    if topic_entity and graph:
        topic_embeddings = [get_fasttext_embedding(fasttext_model, te) for te in topic_entity]
        avg_topic_embedding = np.mean(topic_embeddings, axis=0)
        
        head_similarities = []
        for idx, triple in enumerate(graph):
            head_entity = triple[0]
            head_embedding = get_fasttext_embedding(fasttext_model, head_entity)
            similarity = cosine_similarity(avg_topic_embedding, head_embedding)
            head_similarities.append((idx, similarity))

        head_similarities.sort(key=lambda x: x[1], reverse=True)
        top_head_indices = set(idx for idx, _ in head_similarities[:number_of_triples])
        
        tail_similarities = []
        for idx, triple in enumerate(graph):
            tail_entity = triple[2]
            tail_embedding = get_fasttext_embedding(fasttext_model, tail_entity)
            similarity = cosine_similarity(avg_topic_embedding, tail_embedding)
            tail_similarities.append((idx, similarity))

        tail_similarities.sort(key=lambda x: x[1], reverse=True)
        top_tail_indices = set(idx for idx, _ in tail_similarities[:number_of_triples])
        
        for idx in top_head_indices.union(top_tail_indices):
            selected_triples.append(graph[idx])

    if triple_pattern and graph:
        for pattern_str in triple_pattern:
            pattern_embedding = get_fasttext_embedding(fasttext_model, pattern_str)
            rel_similarities = []
            
            for idx, triple in enumerate(graph):
                relation = triple[1]
                rel_embedding = get_fasttext_embedding(fasttext_model, relation)
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
    Extract and parse answer from reasoning result.
    
    Args:
        reasoning_result: Model reasoning output string.
        
    Returns:
        List containing filtered answers as list of strings.
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
