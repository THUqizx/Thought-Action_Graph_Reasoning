# TAG: Thought-Action Graph for Reasoning over Knowledge Graphs

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

TAG (Thought-Action Graph) is a novel framework for Knowledge Base Question Answering (KBQA) that enables reasoning over knowledge graphs through a dual-layer graph structure representing both abstract ontological reasoning paths and concrete entity-level action chains.

## Overview

TAG introduces a dual-layer graph structure that bridges the gap between high-level ontological reasoning and low-level entity-level execution:

- **Thought Layer**: Represents abstract ontological reasoning paths using ontology types and relations
- **Action Layer**: Represents concrete execution chains with specific entities and actions

This dual-layer approach enables more interpretable and efficient reasoning over large-scale knowledge graphs like Freebase.

## Architecture
<img width="6059" height="3745" alt="instruction" src="https://github.com/user-attachments/assets/4a50d86f-17ee-4288-b87e-9799fdf0aeae" />

## Quick Start

### Installation

```bash
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
   <img width="6050" height="2447" alt="TAG_Construction" src="https://github.com/user-attachments/assets/04854406-b8d7-4dbe-81a1-29428ab189a6" />

2. **Perform Reasoning**:
   ```bash
   cd reasoning
   python reasoning.py             # Run reasoning on your dataset
   ```
<img width="6695" height="2708" alt="TAG_Reasoning" src="https://github.com/user-attachments/assets/d4d84956-f5a1-4414-bede-99db4e526cfa" />


3. **Faster Reasoning** (using saved MACs):
   ```bash
   cd reasoning_by_macs
   python reasoning_by_tag_llama3_1.py    # Using LLaMA3.1
   # or
   python reasoning_by_tag_gpt4o_mini.py  # Using GPT-4o-mini
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

## Configuration

Create a `config.json` file based on `config.json.example`:

```json
{
  "sparql_endpoint": "http://your-sparql-endpoint.com/sparql/",
  "paths": {
    "llama3_1_model_path": "/path/to/Llama-3.1-8B-Instruct",
    "tag_path": "/path/to/TAG.pkl",
    "qwen3_embedding_model_path": "/path/to/Qwen3-Embedding-4B/",
    "query_embeddings_path": "/path/to/Question_Embeddings.pkl",
    "answer_type_name_embeddings_path": "/path/to/AnswerTypeName_Embeddings.pkl",
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

## Data

All processed data, including Thought-Action Graphs (TAG), embeddings, and training sets, are available on [Hugging Face Datasets](https://huggingface.co/datasets/ZhixiaoQi/Thought-Action_Graph_Reasoning).

The repository is structured as follows:

### 1. `construct_TAG/`
Contains raw datasets and their extracted reasoning paths:
* `{WebQSP, CWQ, GrailQA}.train.json`: Original training sets for the three KBQA benchmarks.
* `MAC_WebQSP_CWQ_GrailQA.json`: **Meta-Action-Chains (MAC)** extracted from the combined datasets, serving as the foundation for TAG construction.

### 2. `embeddings/`
Pre-computed semantic vectors for efficient retrieval:
* `Question_Embedding.pkl`: Encoded representations of questions across all datasets.
* `AnswerTypeName_Embedding.pkl`: Encoded representations of answer types.
* *Note: These were generated using the **Qwen3-Embedding** model.*

### 3. `TAG/`
* `TAG.pkl`: The finalized **Thought-Action Graph** built from WebQSP, CWQ, and GrailQA. This file contains the integrated dual-layer structure (Thought Layer and Action Layer).

### 4. `reasoning_by_macs/`
* Contains pre-saved MAC inference results. Users can use these files to perform **fast reasoning** without re-generating reasoning paths from scratch.

### 5. `train/`
Specific datasets used for fine-tuning the **LLaMA-3.1** models within the TAG-R framework:
* `generate_macs.json`: Training data for the **Navigator** (Task: generating reasoning paths).
* `execute_in_graph.json`: Training data for the **Executor** (Task: executing on the Knowledge Graph).

## Evaluation

Run evaluation scripts:

```bash
cd eval
python eval_WebQSP.py   # Evaluate on WebQSP
python eval_CWQ.py      # Evaluate on CWQ
python eval_GrailQA.py  # Evaluate on GrailQA
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
