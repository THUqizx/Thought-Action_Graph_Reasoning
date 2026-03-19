# Thought-Action Graph (TAG) Reasoning

This module implements reasoning over the Thought-Action Graph (TAG) knowledge representation for Knowledge Base Question Answering (KBQA).

## Overview

The TAG reasoning system uses large language models (LLMs) to generate and execute meta-action-chains for answering questions over knowledge graphs. This implementation supports:

- LLaMA3.1 model inference
- GPT-4o-mini model inference
- FastText-based graph filtering
- SPARQL-based entity type/name lookup

## File Structure

```
reasoning_by_macs/
├── config.json              # Configuration file for paths and parameters
├── utils.py                 # Utility functions and model wrappers
├── reasoning_by_tag_llama3_1.py    # Main script for LLaMA3.1-based reasoning
├── reasoning_by_tag_gpt4o_mini.py  # Main script for GPT-4o-mini-based reasoning
└── README.md                # This file
```

## Installation

### Requirements

- Python 3.8+
- OpenAI API key
- SPARQL endpoint (Freebase)
- LLaMA3.1 model (for llama3_1 version)
- FastText embeddings

### Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install openai pandas transformers torch fasttext sentence-transformers tqdm SPARQLWrapper
```

3. Download required models:
   - LLaMA3.1-8B-Instruct (if using llama3_1 version)
   - FastText dbpedia.bin embeddings
   - Qwen3-Embedding-4B (optional)

## Configuration

Create a `config.json` file in the root directory with the following structure:

```json
{
  "sparql_endpoint": "http://your-sparql-endpoint:port/sparql/",
  "models": {
    "llama3_1": {
      "path": "/path/to/Llama-3.1-8B-Instruct"
    },
    "qwen3_embedding": {
      "path": "/path/to/Qwen3-Embedding-4B/"
    },
    "fasttext": {
      "path": "/path/to/FastText/dbpedia.bin"
    }
  },
  "embeddings": {
    "query_embeddings": "/path/to/query_embeddings.pkl",
    "answer_type_name_embeddings": "/path/to/answer_type_embeddings.pkl"
  },
  "prompts": {
    "predict_type_name": "/path/to/predict_typeName_prompt.txt",
    "generate_meta_action_chain": "/path/to/generate_meta_action_chain_prompt.txt",
    "reasoning": "/path/to/reasoning_promptv2.txt"
  },
  "data": {
    "input_dataset": "/path/to/input_dataset.json",
    "output_directory": "/path/to/output/results"
  },
  "tag_config": {
    "explore_breadth": 5,
    "explore_depth": 3,
    "number_of_triples": 100
  },
  "openai": {
    "api_key": "your-api-key",
    "base_url": "https://api.chatanywhere.tech/v1",
    "default_model": "gpt-3.5-turbo",
    "max_attempts": 5,
    "delay_seconds": 30
  },
  "cuda_devices": "0"
}
```

## Usage

### LLaMA3.1-based Reasoning

```bash
cd /path/to/reasoning_by_macs
python reasoning_by_tag_llama3_1.py
```

### GPT-4o-mini-based Reasoning

```bash
cd /path/to/reasoning_by_macs
python reasoning_by_tag_gpt4o_mini.py
```

## Key Components

### utils.py

Contains all utility functions and model wrappers:

- `OpenAIClient`: Wrapper for OpenAI API with retry mechanism
- `load_llama3_1()`: Load LLaMA3.1 model pipeline
- `llama3_1_generate()`: Generate text with LLaMA3.1
- `load_dataset()`: Load dataset from JSON or Parquet
- `load_prompt()`: Load prompt templates
- `execute_sparql()`: Execute SPARQL queries
- `get_entity_name()`: Get entity names from Freebase
- `get_type_name()`: Get type names from Freebase
- `filter_graph_by_similarity()`: Filter graph triples by similarity
- `filter_answer()`: Extract answers from reasoning results

### Thought-Action Graph (TAG)

The TAG framework represents reasoning paths as sequences of actions over a knowledge graph. Each meta-action-chain (MAC) represents a reasoning path that can be executed on the graph to find answers.

## Output

Results are saved as JSON files in the configured output directory with the following structure:

```json
{
  "QuestionId": "unique_question_id",
  "Question": "question text",
  "TopicEntityMid": "freebase_id",
  "Queries_and_MACs": [...],
  "MACs_of_Question": [...],
  "Answers": [[...]],
  "Ground_Truth_Answers": [...],
  "Graph": [[head, relation, tail], ...]
}
```

## References

For more information about the TAG framework, please refer to the main TAG repository.
