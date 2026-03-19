# TAG: Thought-Action Graph for Reasoning over Knowledge Graphs

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

TAG (Thought-Action Graph) is a novel framework for Knowledge Base Question Answering (KBQA) that enables reasoning over knowledge graphs through a dual-layer graph structure representing both abstract ontological reasoning paths and concrete entity-level action chains.

## Overview

TAG introduces a dual-layer graph structure that bridges the gap between high-level ontological reasoning and low-level entity-level execution:

- **Thought Layer**: Represents abstract ontological reasoning paths using ontology types and relations
- **Action Layer**: Represents concrete execution chains with specific entities and actions

This dual-layer approach enables more interpretable and efficient reasoning over large-scale knowledge graphs like Freebase.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Thought-Action Graph                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Thought Layer (Ontology-level)                             │
│  ┌────────────┐     ┌─────────────┐     ┌────────────┐     │
│  │   Person   │────▶│  Location   │────▶│   Country   │     │
│  └────────────┘     └─────────────┘     └────────────┘     │
│        │                   │                   │            │
│        ▼                   ▼                   ▼            │
│  Action Layer (Entity-level)                                │
│  ┌────────────┐     ┌─────────────┐     ┌────────────┐     │
│  │  France    │────▶│  Adjoin_S   │────▶│  Germany    │     │
│  └────────────┘     └─────────────┘     └────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
cd TAG_Open_Source_v2
pip install -r requirements.txt
```

### Usage

1. **Construct the Thought-Action Graph** (one-time setup):
   ```bash
   cd construct_TAG
   python generate_MAC.py          # Generate Meta-Action Chains
   python construct_TAG.py         # Build TAG from MACs
   python encoding.py              # Encode questions and answer types
   python tag_statistics.py        # View TAG statistics
   ```

2. **Perform Reasoning**:
   ```bash
   cd reasoning
   python reasoning.py             # Run reasoning on your dataset
   ```

3. **Faster Reasoning** (using saved MACs):
   ```bash
   cd reasoning_by_macs
   python reasoning_by_tag_llama3_1.py    # Using LLaMA3.1
   # or
   python reasoning_by_tag_gpt4o_mini.py  # Using GPT-4o-mini
   ```

## Project Structure

```
TAG_Open_Source_v2/
├── construct_TAG/          # TAG Construction Module
│   ├── generate_MAC.py     # Generate Meta-Action Chains from datasets
│   ├── construct_TAG.py    # Construct TAG from MACs
│   ├── encoding.py         # Encode questions and answer types
│   ├── tag_statistics.py   # Compute TAG statistics
│   ├── retrieval.py        # Semantic retrieval functions
│   └── ThoughtActionGraph.py  # TAG data structure
│
├── reasoning/              # Reasoning Module
│   ├── reasoning.py        # Main reasoning script
│   ├── retrieval.py        # Dual-layer retrieval methods
│   ├── thought_action_graph.py  # TAG class for reasoning
│   └── utils.py            # Utility functions
│
├── reasoning_by_macs/      # Fast Reasoning Module
│   ├── reasoning_by_tag_llama3_1.py    # LLaMA3.1-based reasoning
│   ├── reasoning_by_tag_gpt4o_mini.py  # GPT-4o-mini-based reasoning
│   └── utils.py            # Utility functions
│
├── data/                   # Training and evaluation data
│   ├── WebQSP/             # WebQSP dataset
│   ├── CWQ/                # ComplexWebQuestions dataset
│   └── GrailQA/            # GrailQA dataset
│
├── prompt/                 # Prompt templates
│   ├── predict_typeName_prompt.txt
│   ├── generate_meta_action_chain_prompt.txt
│   └── reasoning_prompt.txt
│
├── eval/                   # Evaluation scripts
│   ├── eval_WebQSP.py
│   ├── eval_CWQ.py
│   └── eval_GrailQA.py
│
├── requirements.txt        # Main dependencies
├── config.json.example     # Configuration template
└── README.md              # This file
```

## Main Workflow

### 1. TAG Construction ([construct_TAG/](construct_TAG/README.md))

This module builds the Thought-Action Graph from raw datasets:

1. **Generate Meta-Action Chains (MAC)**: Use GPT to generate reasoning paths from questions
   ```bash
   python generate_MAC.py
   ```
   This generates MACs for WebQSP, CWQ, and GrailQA datasets.

2. **Construct TAG**: Convert MACs to the dual-layer TAG structure
   ```bash
   python construct_TAG.py
   ```
   This creates a TAG file (`.pkl`) containing both thought and action layers.

3. **Encode Questions**: Embed questions and answer types for semantic retrieval
   ```bash
   python encoding.py
   ```
   Uses Qwen3-Embedding model to encode questions.

4. **Statistics**: View TAG structure statistics
   ```bash
   python tag_statistics.py --tag_path path/to/tag.pkl
   ```

### 2. Reasoning ([reasoning/](reasoning/README.md))

This module performs reasoning over the constructed TAG:

```bash
python reasoning.py
```

The reasoning process:
- Loads the TAG and encoded embeddings
- Retrieves relevant paths using dual-layer retrieval
- Uses LLaMA3.1 to generate reasoning chains
- Executes reasoning on the knowledge graph
- Saves results to output directory

### 3. Fast Reasoning ([reasoning_by_macs/](reasoning_by_macs/README.md))

For faster inference, use the saved MACs with a graph database:

```bash
python reasoning_by_tag_llama3_1.py
# or
python reasoning_by_tag_gpt4o_mini.py
```

This approach:
- Uses pre-computed MACs stored in JSON
- Can integrate with graph databases (e.g., Neo4j) for faster retrieval
- Supports both LLaMA3.1 and GPT-4o-mini models

### 4. Training ([data/](data/))

Training data for navigator and executor components:

- **Navigator**: Training data for MAC generation
- **Executor**: Training data for reasoning execution

Users can fine-tune models on these datasets for improved performance.

## Installation

### Requirements

- Python 3.8+
- PyTorch >= 1.13.0
- Transformers >= 4.30.0
- Sentence Transformers >= 2.0.0
- FastText
- SPARQLWrapper
- OpenAI API client

### Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Models

Download required models:

1. **LLaMA3.1-8B-Instruct** (for reasoning):
   - Download from Hugging Face
   - Path: `meta-llama/Meta-Llama-3.1-8B-Instruct`

2. **Qwen3-Embedding-4B** (for encoding):
   - Download from Hugging Face
   - Path: `Qwen/Qwen3-Embedding-4B`

3. **FastText** (for graph filtering):
   - Download: `fasttext download model dbpedia`
   - Path: `dbpedia.bin`

## Configuration

Create a `config.json` file based on `config.json.example`:

```json
{
  "sparql_endpoint": "http://your-sparql-endpoint.com/sparql/",
  "paths": {
    "llama3_1_model_path": "/path/to/Llama-3.1-8B-Instruct",
    "tag_path": "/path/to/TAGv2(WebQSP+CWQ,Freebase).pkl",
    "qwen3_embedding_model_path": "/path/to/Qwen3-Embedding-4B/",
    "query_embeddings_path": "/path/to/WebQSP_CWQ_Question_Embeddings.pkl",
    "answer_type_name_embeddings_path": "/path/to/AnswerTypeName_WebQSP_CWQ_Embeddingsv2.pkl",
    "fasttext_embeddings_path": "/path/to/FastText/dbpedia.bin"
  },
  "parameters": {
    "tag_explore_breadth": 5,
    "tag_explore_depth": 3,
    "max_attempts": 5,
    "delay": 30
  }
}
```

## Datasets

TAG supports the following KBQA datasets:

- **WebQSP**: Web-scale Question Answering dataset
- **CWQ**: Complex WebQuestions dataset with multi-hop reasoning
- **GrailQA**: Large-scale dataset for semantic parsing over Freebase

Dataset statistics are included in the pre-constructed TAG files.

## Evaluation

Run evaluation scripts:

```bash
cd eval
python eval_WebQSP.py   # Evaluate on WebQSP
python eval_CWQ.py      # Evaluate on CWQ
python eval_GrailQA.py  # Evaluate on GrailQA
```

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Freebase knowledge graph
- LLaMA3.1 team for open-weight models
- OpenAI for GPT models
- Hugging Face for transformer library

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
