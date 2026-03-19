"""
Example usage of Thought-Action Graph (TAG) construction.

This script demonstrates how to use the TAG module to:
1. Generate MAC from questions
2. Construct TAG from MAC
3. Perform semantic retrieval
"""

import json
from sentence_transformers import SentenceTransformer

# Import TAG modules
from utils import load_config, encode_queries, typeName_all
from ThoughtActionGraph import ThoughtActionGraph
from retrieval import retrieval_from_start, retrieval_from_start_to_end
from construct_TAG import construct_tag_from_mac
from generate_MAC import MACGenerator
from chat_with_gpt import OpenAIClient


def example_1_load_config():
    """Example 1: Load configuration from config.json"""
    print("=" * 60)
    print("Example 1: Load Configuration")
    print("=" * 60)
    
    config = load_config("config.json")
    
    print(f"SPARQL Endpoint: {config['sparql_endpoint']}")
    print(f"Model Name: {config['model_name']}")
    print(f"Batch Size: {config['model']['batch_size']}")
    
    print("\nPaths:")
    for key, value in config['paths'].items():
        print(f"  {key}: {value}")
    
    print()


def example_2_load_tag():
    """Example 2: Load and inspect a pre-constructed TAG"""
    print("=" * 60)
    print("Example 2: Load and Inspect TAG")
    print("=" * 60)
    
    # Load TAG from file
    tag = ThoughtActionGraph.load_from_file("data/TAGv2(WebQSP+CWQ+GrailQA,Freebase).pkl")
    
    print(f"Total Entities: {len(tag.entities)}")
    print(f"Total Relations: {len(tag.relations)}")
    
    # Count entity types
    entity_counts = tag.count_entity_types()
    print("\nEntity Type Distribution:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")
    
    # Count relation types
    relation_counts = tag.count_relation_types()
    print("\nRelation Type Distribution:")
    for relation_type, count in list(relation_counts.items())[:10]:  # Show top 10
        print(f"  {relation_type}: {count}")
    
    print()


def example_3_retrieve_paths():
    """Example 3: Retrieve paths from TAG"""
    print("=" * 60)
    print("Example 3: Retrieve Paths from TAG")
    print("=" * 60)
    
    # Load TAG
    tag = ThoughtActionGraph.load_from_file("data/TAGv2(WebQSP+CWQ+GrailQA,Freebase).pkl")
    
    # Retrieve paths from a specific start node
    start_name = "Person"
    start_attrs = {"Type": "Ontology", "Layer": "Thought"}
    
    paths = retrieval_from_start(tag, start_name, start_attrs)
    
    print(f"Found {len(paths)} paths starting from '{start_name}'")
    
    if paths:
        print("\nFirst path:")
        for i, triple in enumerate(paths[0]):
            head, relation, tail = triple
            print(f"  {i}: {head['name']} --[{relation}]--> {tail['name']}")
    
    print()


def example_4_encode_queries():
    """Example 4: Encode questions using Qwen3 model"""
    print("=" * 60)
    print("Example 4: Encode Questions")
    print("=" * 60)
    
    # Load model
    print("Loading Qwen3 model...")
    model = SentenceTransformer("path/to/qwen3/model")
    
    # Example questions
    questions = [
        "What is the capital of France?",
        "Who wrote Harry Potter?",
        "What is the population of China?"
    ]
    
    # Encode questions
    embeddings = encode_queries(model, questions, batch_size=2)
    
    print(f"\nEncoded {len(embeddings)} questions")
    for question, embedding in embeddings.items():
        print(f"  '{question}': {len(embedding)}-dimensional vector")
    
    print()


def example_5_generate_mac():
    """Example 5: Generate MAC using GPT"""
    print("=" * 60)
    print("Example 5: Generate MAC using GPT")
    print("=" * 60)
    
    # Load configuration
    config = load_config("config.json")
    api_key = config['api_key']
    
    # Initialize client and generator
    client = OpenAIClient(api_key)
    generator = MACGenerator(client, api_key, model_name=config['model_name'])
    
    # Example prompt
    prompt = """You are an expert in natural language processing. 
Given a question, answer, and SPARQL query, generate a meta-action chain.
Format: head_entity --> option1(action1) --> option2(action2) --> ... --> tail_entity

Question: What is the capital of France?
Answer: Paris
SPARQL: SELECT ?x WHERE { ns:m.02hj4 ns:location.location.adjoin_s ?x . }
Head Entity: France
Tail Entity: Paris
meta-action chain:
"""
    
    # Generate MAC
    mac = generator.generate_mac(prompt)
    
    if mac:
        print(f"\nGenerated MAC:\n{mac}")
    else:
        print("\nFailed to generate MAC")
    
    print()


def example_6_construct_tag():
    """Example 6: Construct TAG from MAC data"""
    print("=" * 60)
    print("Example 6: Construct TAG from MAC")
    print("=" * 60)
    
    # Load configuration
    config = load_config("config.json")
    sparql_endpoint = config['sparql_endpoint']
    
    # Paths
    mac_path = "data/MAC_WebQSP_CWQ_GrailQA.json"
    tag_path = "data/TAGv2(WebQSP+CWQ+GrailQA,Freebase).pkl"
    
    print(f"Loading MAC from: {mac_path}")
    print(f"Saving TAG to: {tag_path}")
    
    # Construct TAG
    construct_tag_from_mac(mac_path, tag_path, sparql_endpoint)
    
    print(f"\nTAG constructed and saved to {tag_path}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Thought-Action Graph (TAG) Examples")
    print("=" * 60 + "\n")
    
    # Example 1: Load configuration
    try:
        example_1_load_config()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    # Example 2: Load TAG
    try:
        example_2_load_tag()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    # Example 3: Retrieve paths
    try:
        example_3_retrieve_paths()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    # Example 4: Encode queries
    try:
        example_4_encode_queries()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    # Example 5: Generate MAC
    try:
        example_5_generate_mac()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    # Example 6: Construct TAG
    try:
        example_6_construct_tag()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
