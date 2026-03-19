# Thought-Action Graph Reasoning Module

This module provides functionality for reasoning over the Thought-Action Graph (TAG) for Knowledge Base Question Answering (KBQA) tasks.

## Overview

The Thought-Action Graph is a dual-layer graph structure that enables reasoning over knowledge graphs by:
- **Thought Layer**: Abstract ontological reasoning paths
- **Action Layer**: Concrete entity-level action chains

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "paths": {
    "llama3_1_model_path": "/path/to/Llama-3.1-8B-Instruct",
    "tag_path": "/path/to/TAG.pkl",
    "qwen3_embedding_model_path": "/path/to/Qwen3-Embedding-4B/",
    "query_embeddings_path": "/path/to/Question_Embeddings.pkl",
    "answer_type_name_embeddings_path": "/path/to/AnswerTypeName_Embeddings.pkl",
    "fasttext_embeddings_path": "/path/to/FastText/dbpedia.bin",
    "webqsp_test_path": "/path/to/WebQSP/WebQSP.test.json",
    "predict_type_name_prompt_path": "/prompt/predict_typeName_prompt.txt",
    "generate_meta_action_chain_prompt_path": "/prompt/generate_meta_action_chain_prompt.txt",
    "reasoning_prompt_path": "/prompt/reasoning_prompt.txt",
    "answer_save_path": "/path/to/save/results"
  },
  "sparql": {
    "endpoint": "http://your-sparql-endpoint.com/sparql/"
  },
  "parameters": {
    "tag_explore_breadth": 5,
    "tag_explore_depth": 3,
    "max_attempts": 5,
    "delay": 30,
    "batch_size": 10,
    "number_of_triples": 200,
    "cuda_visible_devices": "2,3"
  },
  "model_settings": {
    "llama3_1_dtype": "bfloat16",
    "max_new_tokens": 32768
  }
}
```

## Usage

```python
from reasoning import main_reasoning

# Run reasoning on the dataset
main_reasoning.main()
```

## Module Structure

### Core Components

- **`thought_action_graph.py`**: Defines the `ThoughtActionGraph` class
  - `ThoughtActionGraph`: Main graph class for managing entities and relationships
  - Methods for path retrieval, filtering, and graph operations

- **`retrieval.py`**: Implements dual-layer retrieval methods
  - `RetrievalFromStart`: Retrieve paths from start node
  - `pruning`: Filter paths by target node
  - `ThoughtNode2ActionNode`: Map thought nodes to action nodes
  - `RetrievalFromStart2End`: Find paths between start and end nodes
  - `FilterActionChain`: Filter action chains by similarity
  - `ActionChain2ThoughtChain`: Convert action chains to thought chains
  - `MergeMetaActionChain`: Merge action and thought chains

- **`utils.py`**: Utility functions
  - Model loading functions (Llama-3.1, Qwen3)
  - Dataset and embedding loading
  - SPARQL query execution
  - Graph filtering and answer extraction

- **`main_reasoning.py`**: Main entry point
  - Loads configuration and models
  - Processes questions from dataset
  - Saves results to JSON

## Example Configuration

See `config.json` for a complete example configuration file.
