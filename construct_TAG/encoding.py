"""
Encode questions and answer types for Thought-Action Graph retrieval.

This module provides functionality to encode questions and answer type names
using the Qwen3 embedding model for semantic retrieval.
"""

import json
import pickle
from typing import Dict, List, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import encode_queries, typeName_all, load_config

def encode_questions(Qwen3Model: SentenceTransformer, question_list: List[str], 
                     batch_size: int = 10) -> Dict[str, Any]:
    """
    Encode a list of questions using the Qwen3 embedding model.
    
    Args:
        Qwen3Model: Qwen3 embedding model instance.
        question_list: List of question strings to encode.
        batch_size: Number of questions to process in each batch.
        
    Returns:
        Dictionary mapping each question to its embedding.
    """
    return encode_queries(Qwen3Model, question_list, batch_size)

def encode_answer_types(Qwen3Model: SentenceTransformer, mac_data: List[Dict[str, Any]], 
                        batch_size: int = 10, sparql_endpoint: str = None) -> Dict[str, Any]:
    """
    Encode all answer type names from MAC data.
    
    Args:
        Qwen3Model: Qwen3 embedding model instance.
        mac_data: List of MAC data dictionaries.
        batch_size: Number of type names to process in each batch.
        sparql_endpoint: URL of the SPARQL endpoint.
        
    Returns:
        Dictionary mapping each answer type name to its embedding.
    """
    answer_type_names = []
    
    for data_with_mac in tqdm(mac_data, desc="Extracting answer types"):
        for answer in data_with_mac["Answers"]:
            source = data_with_mac["Source"]
            if source == "CWQ":
                if answer["answer"] is not None:
                    answer_type_names.extend(typeName_all(answer["answer_id"], sparql_endpoint))
            elif source == "WebQSP":
                if answer["EntityName"] is not None:
                    answer_type_names.extend(typeName_all(answer["AnswerArgument"], sparql_endpoint))
            elif source == "GrailQA":
                if answer["entity_name"] is not None:
                    answer_type_names.extend(typeName_all(answer["answer_argument"], sparql_endpoint))
            else:
                raise ValueError(f"Unknown Source: {source}")
    
    answer_type_names = list(set(answer_type_names))
    answer_type_names = [name for name in answer_type_names if name is not None]
    
    return encode_queries(Qwen3Model, answer_type_names, batch_size)

def encode_mac_data(Qwen3Model: SentenceTransformer, mac_path: str, 
                    question_save_path: str, answertypename_save_path: str,
                    batch_size: int = 10, sparql_endpoint: str = None) -> None:
    """
    Encode questions and answer types from MAC data and save to files.
    
    Args:
        Qwen3Model: Qwen3 embedding model instance.
        mac_path: Path to MAC JSON file.
        question_save_path: Path to save question embeddings.
        answertypename_save_path: Path to save answer type embeddings.
        batch_size: Number of items to process in each batch.
        sparql_endpoint: URL of the SPARQL endpoint.
    """
    with open(mac_path, 'r', encoding='utf-8') as f:
        mac_data = json.load(f)
    
    question_list = [sample["ProcessedQuestion"] for sample in mac_data]
    
    print("Encoding questions...")
    question_embeddings_dict = encode_questions(Qwen3Model, question_list, batch_size)
    
    with open(question_save_path, 'wb') as f:
        pickle.dump(question_embeddings_dict, f)
    print(f"Question embeddings saved to {question_save_path}")
    
    print("Encoding answer types...")
    answertypename_embeddings_dict = encode_answer_types(
        Qwen3Model, mac_data, batch_size, sparql_endpoint
    )
    
    with open(answertypename_save_path, 'wb') as f:
        pickle.dump(answertypename_embeddings_dict, f)
    print(f"Answer type embeddings saved to {answertypename_save_path}")

def main():
    """
    Main function to encode MAC data.
    
    Usage:
        1. Set paths in config.json
        2. Run: python encoding.py
    """
    config = load_config("config.json")
    
    paths = config.get("paths")
    model_config = config.get("model")
    sparql_endpoint = config.get("sparql_endpoint")
    
    qwen3_embedding_path = model_config.get("qwen3_embedding_path")
    batch_size = model_config.get("batch_size")
    
    print(f"Loading Qwen3 model from: {qwen3_embedding_path}")
    Qwen3Model = SentenceTransformer(qwen3_embedding_path)
    
    mac_path = paths.get("mac_path")
    question_save_path = paths.get("question_embeddings")
    answertypename_save_path = paths.get("answertypename_embeddings")
    
    encode_mac_data(Qwen3Model, mac_path, question_save_path, answertypename_save_path, batch_size, sparql_endpoint)
    
    print("\nEncoding completed!")

if __name__ == "__main__":
    main()
