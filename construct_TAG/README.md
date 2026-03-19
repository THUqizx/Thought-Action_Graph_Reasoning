<<<<<<< HEAD
# Thought-Action Graph (TAG) Construction

This module provides tools for constructing and working with Thought-Action Graphs from Meta-Action Chains (MAC) in knowledge base question answering.

## Overview

The Thought-Action Graph (TAG) is a dual-layer graph structure that represents both:
- **Thought Layer**: Ontology-level reasoning chains
- **Action Layer**: Entity-level execution chains

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "sparql_endpoint": "http://101.6.69.142:3001/sparql/",
  "api_key": "YOUR_API_KEY_HERE",
  "model_name": "gpt-4o",
  
  "paths": {
    "data_dir": "YOUR_DATA_DIR_HERE",
    "tag_dir": "YOUR_TAG_DIR_HERE",
    "embedding_dir": "YOUR_EMBEDDING_DIR_HERE",
    "prompt_file": "YOUR_PROMPT_FILE_HERE",
    
    "webqsp_data": "YOUR_WEBQSP_DATA_PATH_HERE",
    "cwq_data": "YOUR_CWQ_DATA_PATH_HERE",
    "grailqa_data": "YOUR_GRAILQA_DATA_PATH_HERE",
    
    "webqsp_tag": "YOUR_WEBQSP_TAG_PATH_HERE",
    "cwq_tag": "YOUR_CWQ_TAG_PATH_HERE",
    "grailqa_tag": "YOUR_GRAILQA_TAG_PATH_HERE",
    
    "webqsp_cwq_tag": "YOUR_WEBQSP_CWQ_TAG_PATH_HERE",
    "webqsp_cwq_grailqa_tag": "YOUR_WEBQSP_CWQ_GRAILQA_TAG_PATH_HERE",
    
    "mac_path": "YOUR_MAC_PATH_HERE",
    "tag_path": "YOUR_TAG_PATH_HERE",
    
    "question_embeddings": "YOUR_QUESTION_EMBEDDINGS_PATH_HERE",
    "answertypename_embeddings": "YOUR_ANSWERTYPE_EMBEDDINGS_PATH_HERE"
  },
  
  "model": {
    "qwen3_embedding_path": "YOUR_QWEN3_EMBEDDING_PATH_HERE",
    "batch_size": 10
  },
  
  "gpt_settings": {
    "max_attempts": 5,
    "delay": 30,
    "temperature": 0.7,
    "max_tokens": null
  },
  
  "retrieval": {
    "top_k": 5,
    "max_depth": 10
  }
}
```

## Usage

### 1. Generate Meta-Action Chains (MAC)

Generate MAC for WebQSP, CWQ, or GrailQA datasets:

```bash
python generate_MAC.py
```

### 2. Construct Thought-Action Graph

Convert MAC data to TAG:

```bash
python construct_TAG.py
```

### 3. Encode Questions and Answer Types

Encode questions and answer types for retrieval:

```bash
python encoding.py
```

### 4. Compute Statistics

Get statistics about the TAG:

```bash
python tag_statistics.py --tag_path path/to/tag.pkl
```

### 5. Semantic Retrieval

Use the retrieval module for finding similar queries and traversing the graph:

```python
from retrieval import retrieval_from_start, retrieval_from_start_to_end
from ThoughtActionGraph import ThoughtActionGraph

tag = ThoughtActionGraph.load_from_file("path/to/tag.pkl")

# Retrieve paths from start node
paths = retrieval_from_start(tag, "start_node_name", {"Type": "Ontology", "Layer": "Thought"})

# Retrieve paths from start to end node
paths = retrieval_from_start_to_end(
    tag, 
    "start_node_name", 
    {"Type": "Ontology", "Layer": "Thought"},
    "end_node_name", 
    {"Type": "Ontology", "Layer": "Thought"}
)
```

## Module Structure

- `config.json`: Configuration file for paths and settings
- `utils.py`: Utility functions for SPARQL queries, encoding, and file operations
- `ThoughtActionGraph.py`: Core data structure for the dual-layer graph
- `chat_with_gpt.py`: OpenAI client wrapper for GPT interactions
- `generate_MAC.py`: Generate MAC from WebQSP, CWQ, and GrailQA datasets
- `construct_TAG.py`: Construct TAG from MAC data
- `encoding.py`: Encode questions and answer types using Qwen3 embeddings
- `retrieval.py`: Semantic retrieval and graph traversal functions
- `merge_MAC.py`: Merge MAC data from multiple datasets
- `tag_statistics.py`: Compute and display TAG statistics

## Data Flow

1. **Input**: Raw dataset (WebQSP/CWQ/GrailQA) with questions and answers
2. **MAC Generation**: Use GPT to generate Meta-Action Chains
3. **TAG Construction**: Convert MAC to dual-layer Thought-Action Graph
4. **Encoding**: Embed questions and answer types for retrieval
5. **Retrieval**: Use semantic similarity for query matching and path finding

## License

This is a sub-module of the GRAG project. See the main project for license information.
=======
# Thought-Action-Graph-Reasoning
>>>>>>> dcccfba001344ad11c16776012ad5ceb5eddaddc
