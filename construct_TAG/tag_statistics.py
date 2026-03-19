"""
Statistics for Thought-Action Graph.

This module provides functionality to compute and display statistics
about the Thought-Action Graph structure.
"""

from ThoughtActionGraph import ThoughtActionGraph


def compute_statistics(tag_path: str) -> dict:
    """
    Compute statistics for a Thought-Action Graph.
    
    Args:
        tag_path: Path to the TAG pickle file.
        
    Returns:
        Dictionary containing entity and relation statistics.
    """
    print(f"Loading TAG from: {tag_path}")
    tag = ThoughtActionGraph.load_from_file(tag_path)
    
    entity_counts = tag.count_entity_types()
    relation_counts = tag.count_relation_types()
    
    statistics = {
        "total_entities": len(tag.entities),
        "total_relations": sum(len(rels) for rels in tag.relations.values()),
        "total_triples": sum(
            sum(len(tails) for tails in relations.values()) 
            for relations in tag.relations.values()
        ),
        "entity_types": entity_counts,
        "relation_types": relation_counts
    }
    
    return statistics


def print_statistics(statistics: dict) -> None:
    """
    Print statistics in a formatted way.
    
    Args:
        statistics: Dictionary of statistics to print.
    """
    print("\n" + "=" * 60)
    print("Thought-Action Graph Statistics")
    print("=" * 60)
    
    print(f"\nTotal Entities: {statistics['total_entities']}")
    print(f"Total Relations: {statistics['total_relations']}")
    print(f"Total Triples: {statistics['total_triples']}")
    
    print("\nEntity Type Distribution:")
    print("-" * 40)
    for entity_type, count in statistics['entity_types'].items():
        print(f"  {entity_type}: {count}")
    
    print("\nRelation Type Distribution:")
    print("-" * 40)
    for relation_type, count in statistics['relation_types'].items():
        print(f"  {relation_type}: {count}")
    
    print("\n" + "=" * 60)


def main():
    """
    Main function to compute and display TAG statistics.
    
    Usage:
        1. Set paths in config.json
        2. Run: python tag_statistics.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute TAG statistics")
    parser.add_argument("--tag_path", type=str, help="Path to TAG pickle file")
    
    args = parser.parse_args()
    
    if args.tag_path:
        statistics = compute_statistics(args.tag_path)
        print_statistics(statistics)
    else:
        print("Please provide tag_path or run with config.json")


if __name__ == "__main__":
    main()
