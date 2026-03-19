"""
Thought-Action Graph Reasoning Module

This module provides functionality for reasoning over the Thought-Action Graph
for Knowledge Base Question Answering (KBQA) tasks.

Main Components:
- ThoughtActionGraph: Graph structure for reasoning
- SemanticDualLayerRetrieval: Dual-layer retrieval methods
- utils: Utility functions for loading models and data
- main_reasoning: Main entry point for reasoning

Usage:
    from reasoning import main_reasoning
    main_reasoning.main()
"""

from .thought_action_graph import ThoughtActionGraph
from .semantic_dual_layer_retrieval import (
    RetrievalFromStart,
    pruning,
    ThoughtNode2ActionNode,
    RetrievalFromStart2End,
    FilterActionChain,
    ActionChain2ThoughtChain,
    MergeMetaActionChain,
    RetrievalSimilarQueries
)
from .utils import (
    load_llama3_1,
    llama3_1_generate,
    load_dataset,
    load_prompt,
    load_embeddings,
    typeName,
    typeName_all,
    GraphFilter,
    filter_answer,
    load_config
)

__version__ = "1.0.0"
__all__ = [
    'ThoughtActionGraph',
    'RetrievalFromStart',
    'pruning',
    'ThoughtNode2ActionNode',
    'RetrievalFromStart2End',
    'FilterActionChain',
    'ActionChain2ThoughtChain',
    'MergeMetaActionChain',
    'RetrievalSimilarQueries',
    'load_llama3_1',
    'llama3_1_generate',
    'load_dataset',
    'load_prompt',
    'load_embeddings',
    'typeName',
    'typeName_all',
    'GraphFilter',
    'filter_answer',
    'load_config',
    'main_reasoning'
]
