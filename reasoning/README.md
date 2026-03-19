# Thought-Action Graph Reasoning Module

This module provides functionality for reasoning over the Thought-Action Graph (TAG) for Knowledge Base Question Answering (KBQA) tasks.

## Overview

The Thought-Action Graph is a dual-layer graph structure that enables reasoning over knowledge graphs by:
- **Thought Layer**: Abstract ontological reasoning paths
- **Action Layer**: Concrete entity-level action chains

## Installation

```bash
pip install torch transformers sentence-transformers fasttext pandas
```

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "paths": {
    "llama3_1_model_path": "/path/to/Llama-3.1-8B-Instruct",
    "tag_path": "/path/to/TAGv2(WebQSP+CWQ,Freebase).pkl",
    "qwen3_embedding_model_path": "/path/to/Qwen3-Embedding-4B/",
    "query_embeddings_path": "/path/to/WebQSP_CWQ_Question_Embeddings.pkl",
    "answer_type_name_embeddings_path": "/path/to/AnswerTypeName_WebQSP_CWQ_Embeddingsv2.pkl",
    "fasttext_embeddings_path": "/path/to/FastText/dbpedia.bin",
    "webqsp_test_path": "/path/to/WebQSP/WebQSP.test.json",
    "predict_type_name_prompt_path": "/path/to/predict_typeName_prompt.txt",
    "generate_meta_action_chain_prompt_path": "/path/to/generate_meta_action_chain_prompt.txt",
    "reasoning_prompt_path": "/path/to/reasoning_promptv2.txt",
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

### Basic Usage

```python
from reasoning import main_reasoning

# Run reasoning on the dataset
main_reasoning.main()
```

### Advanced Usage

```python
from reasoning import ThoughtActionGraph, load_config, reason_by_TAG

# Load configuration
config = load_config("config.json")

# Load the Thought-Action Graph
tag = ThoughtActionGraph.load_from_file(config["paths"]["tag_path"])

# Load models and data
# ... (load models, embeddings, prompts)

# Perform reasoning
queries_and_macs, macs_of_question, answers, filtered_graph = reason_by_TAG(
    llama3_1=llama3_1,
    question="What is the capital of France?",
    topic_entity_mid="m.02h5n",
    topic_entity_name="France",
    graph=graph,
    typename_embeddings=typename_embeddings,
    query_embeddings=query_embeddings,
    tag=tag,
    qwen3_model=qwen3_model,
    fasttext_model=fasttext_model,
    predict_type_name_prompt=predict_type_name_prompt,
    generate_meta_action_chain_prompt=generate_meta_action_chain_prompt,
    reasoning_prompt=reasoning_prompt
)
```

## Module Structure

### Core Components

- **`thought_action_graph.py`**: Defines the `ThoughtActionGraph` class
  - `ThoughtActionGraph`: Main graph class for managing entities and relationships
  - Methods for path retrieval, filtering, and graph operations

- **`semantic_dual_layer_retrieval.py`**: Implements dual-layer retrieval methods
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

## API Reference

### ThoughtActionGraph

```python
class ThoughtActionGraph:
    def __init__(self): ...
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "ThoughtActionGraph": ...
    
    def add_entity(self, name: str, attributes: Dict[str, Any]) -> str: ...
    
    def add_relation(self, head_id: str, relation: str, tail_id: str) -> None: ...
    
    def find_nodes_by_name_and_attributes(self, name: str, attrs: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    
    def RetrievalFromStart(self, start_name: str, start_attrs: Dict[str, Any]) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]: ...
    
    def RetrievalFromStart2End(self, start_name: str, start_attrs: Dict[str, Any], end_name: str, end_attrs: Dict[str, Any], max_depth: int = 10) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]: ...
    
    def ThoughtNode2ActionNode(self, head_name: str, head_attrs: Dict[str, Any], target_relation: str, tail_attrs: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    
    def ActionChain2ThoughtChain(self, ActionChains: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]) -> List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]: ...
```

### Utility Functions

```python
def load_llama3_1(model_path: str): ...
def llama3_1_generate(pipeline, prompt: str) -> str: ...
def load_dataset(dataset_path: str) -> List[Dict]: ...
def load_embeddings(embeddings_path: str) -> Dict: ...
def load_prompt(prompt_path: str) -> str: ...
def GraphFilter(embedding_model, meta_action_chains: List[str], graph: List[List[str]], number_of_triples: int) -> List[List[str]]: ...
def filter_answer(reasoning_result: str) -> List[List[str]]: ...
def load_config(config_path: str) -> Dict: ...
```

## Example Configuration

See `config.json` for a complete example configuration file.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Sentence Transformers
- FastText
- SPARQLWrapper
- pandas
- tqdm

## License

This is a sub-module of the TAG project. See the main project for license information.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{tag2024,
  title={TAG: Thought-Action Graph for Reasoning over Knowledge Graphs},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
