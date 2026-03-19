import pickle
from typing import Dict, List, Tuple, Any, Optional, Set


class ThoughtActionGraph:
    """
    A graph structure representing the Thought-Action Graph for reasoning tasks.
    
    This class provides methods to:
    - Manage entities and their relationships
    - Retrieve paths through the graph
    - Filter and prune subgraphs based on semantic similarity
    """
    
    def __init__(self):
        """
        Initialize an empty ThoughtActionGraph.
        
        Attributes:
            entities: Dictionary mapping entity IDs to entity data (name and attributes)
            relations: Dictionary mapping head entity IDs to their relationships
            name_to_ids: Dictionary mapping entity names to their IDs (handles duplicate names)
            next_id: Counter for generating unique entity IDs
        """
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, Dict[str, List[str]]] = {}
        self.name_to_ids: Dict[str, List[str]] = {}
        self.next_id = 1
        
    def _generate_id(self) -> str:
        """
        Generate a unique entity ID.
        
        Returns:
            A unique entity ID string in format 'entity_{n}'
        """
        entity_id = f"entity_{self.next_id}"
        self.next_id += 1
        return entity_id
    
    def add_entity(self, name: str, attributes: Dict[str, Any]) -> str:
        """
        Add an entity to the graph.
        
        Args:
            name: Entity name
            attributes: Dictionary of entity attributes
            
        Returns:
            The ID of the newly added entity
        """
        entity_id = self._generate_id()
        self.entities[entity_id] = {
            "name": name,
            "attributes": attributes.copy()
        }
        
        if name not in self.name_to_ids:
            self.name_to_ids[name] = []
        self.name_to_ids[name].append(entity_id)
        
        return entity_id
    
    def add_relation(self, head_id: str, relation: str, tail_id: str) -> None:
        """
        Add a relationship between two entities.
        
        Args:
            head_id: ID of the head entity
            relation: Type of relationship
            tail_id: ID of the tail entity
            
        Raises:
            ValueError: If either entity does not exist in the graph
        """
        if head_id not in self.entities or tail_id not in self.entities:
            raise ValueError("Head entity or tail entity does not exist in the graph")
        
        if head_id not in self.relations:
            self.relations[head_id] = {}
        
        if relation not in self.relations[head_id]:
            self.relations[head_id][relation] = []
        
        if tail_id not in self.relations[head_id][relation]:
            self.relations[head_id][relation].append(tail_id)
    
    def add_triple(self, head_name: str, head_attrs: Dict[str, Any], 
                   relation: str, 
                   tail_name: str, tail_attrs: Dict[str, Any]) -> Tuple[str, str]:
        """
        Add a triple (head-relation-tail) to the graph, creating entities if needed.
        
        Args:
            head_name: Name of the head entity
            head_attrs: Attributes of the head entity
            relation: Relationship type
            tail_name: Name of the tail entity
            tail_attrs: Attributes of the tail entity
            
        Returns:
            Tuple of (head_entity_id, tail_entity_id)
        """
        head_id = self._find_entity_with_attributes(head_name, head_attrs)
        if not head_id:
            head_id = self.add_entity(head_name, head_attrs)
        
        tail_id = self._find_entity_with_attributes(tail_name, tail_attrs)
        if not tail_id:
            tail_id = self.add_entity(tail_name, tail_attrs)
        
        self.add_relation(head_id, relation, tail_id)
        return head_id, tail_id
    
    def _find_entity_with_attributes(self, name: str, attributes: Dict[str, Any]) -> Optional[str]:
        """
        Find an entity ID by name and exact attribute match.
        
        Args:
            name: Entity name
            attributes: Entity attributes to match exactly
            
        Returns:
            Entity ID if found, None otherwise
        """
        if name not in self.name_to_ids:
            return None
        
        for entity_id in self.name_to_ids[name]:
            entity = self.entities[entity_id]
            if entity["attributes"] == attributes:
                return entity_id
        
        return None
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity data by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data dictionary or None if not found
        """
        return self.entities.get(entity_id)
    
    def get_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all entities with the given name.
        
        Args:
            name: Entity name
            
        Returns:
            List of entity data dictionaries
        """
        entity_ids = self.name_to_ids.get(name, [])
        return [self.entities[entity_id] for entity_id in entity_ids]
    
    def get_relations(self, entity_id: str, relation: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get relations for an entity.
        
        Args:
            entity_id: The entity ID
            relation: Optional specific relation type
            
        Returns:
            Dictionary of relations, or specific relation if specified
        """
        if entity_id not in self.relations:
            return {}
        
        if relation:
            return {relation: self.relations[entity_id].get(relation, [])}
        return self.relations[entity_id].copy()
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the graph to a pickle file.
        
        Args:
            file_path: Path to save the file
        """
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "name_to_ids": self.name_to_ids,
            "next_id": self.next_id
        }
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    
    def count_entities_by_type(self, type_attr_name: str, target_type: Any) -> int:
        """
        Count entities with a specific type attribute.
        
        Args:
            type_attr_name: Name of the type attribute
            target_type: Target type value
            
        Returns:
            Number of entities matching the type
        """
        count = 0
        for entity in self.entities.values():
            if (type_attr_name in entity["attributes"] and 
                entity["attributes"][type_attr_name] == target_type):
                count += 1
        return count

    def count_relation_types(self) -> Dict[str, int]:
        """
        Count occurrences of each relation type.
        
        Returns:
            Dictionary mapping relation types to their counts
        """
        relation_counts = {}
        
        for head_relations in self.relations.values():
            for relation, tails in head_relations.items():
                if relation in relation_counts:
                    relation_counts[relation] += len(tails)
                else:
                    relation_counts[relation] = len(tails)
        
        return relation_counts

    def find_nodes_by_name_and_attributes(self, name: str, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find nodes by name and attributes.
        
        Args:
            name: Node name
            attrs: Attribute dictionary
            
        Returns:
            List of matching node data dictionaries
        """
        cache_key = (name, frozenset(attrs.items())) if attrs else name
        
        if hasattr(self, '_node_cache') and cache_key in self._node_cache:
            return self._node_cache[cache_key]
        
        matching_nodes = []
        
        if name in self.name_to_ids:
            for entity_id in self.name_to_ids[name]:
                entity = self.entities.get(entity_id)
                if entity:
                    match = True
                    for attr_name, attr_value in attrs.items():
                        if (attr_name not in entity["attributes"] or 
                            entity["attributes"][attr_name] != attr_value):
                            match = False
                            break
                    if match:
                        matching_nodes.append(entity)
        
        if not hasattr(self, '_node_cache'):
            self._node_cache = {}
        
        self._node_cache[cache_key] = matching_nodes
        return matching_nodes

    def find_triples_with_tail_node(self, name: str, attrs: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """
        Find all triples where the specified node is the tail.
        
        Args:
            name: Tail node name
            attrs: Tail node attributes
            
        Returns:
            List of triples (head_entity, relation, tail_entity)
        """
        tail_nodes = self.find_nodes_by_name_and_attributes(name, attrs)
        if not tail_nodes:
            return []
        
        tail_ids = []
        for node in tail_nodes:
            for entity_id, entity_data in self.entities.items():
                if entity_data == node:
                    tail_ids.append(entity_id)
                    break
        
        result_triples = []
        for head_id, relations in self.relations.items():
            for relation_type, tail_nodes_list in relations.items():
                for tail_id in tail_ids:
                    if tail_id in tail_nodes_list:
                        head_entity = self.get_entity_by_id(head_id)
                        tail_entity = self.get_entity_by_id(tail_id)
                        if head_entity and tail_entity:
                            result_triples.append((head_entity, relation_type, tail_entity))
        
        return result_triples

    def find_triples_with_head_node(self, name: str, attrs: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """
        Find all triples where the specified node is the head.
        
        Args:
            name: Head node name
            attrs: Head node attributes
            
        Returns:
            List of triples (head_entity, relation, tail_entity)
        """
        head_nodes = self.find_nodes_by_name_and_attributes(name, attrs)
        if not head_nodes:
            return []
        
        head_ids = []
        for node in head_nodes:
            for entity_id, entity_data in self.entities.items():
                if entity_data == node:
                    head_ids.append(entity_id)
                    break
        
        result_triples = []
        for head_id in head_ids:
            if head_id in self.relations:
                for relation_type, tail_ids in self.relations[head_id].items():
                    for tail_id in tail_ids:
                        head_entity = self.get_entity_by_id(head_id)
                        tail_entity = self.get_entity_by_id(tail_id)
                        if head_entity and tail_entity:
                            result_triples.append((head_entity, relation_type, tail_entity))
        
        return result_triples
    
    def find_head_nodes_by_tail_and_relation(self, tail_name: str, tail_attrs: Dict[str, Any], relation: str) -> List[Dict[str, Any]]:
        """
        Find all head nodes connected to a tail node via a specific relation.
        
        Args:
            tail_name: Tail node name
            tail_attrs: Tail node attributes
            relation: Relationship type
            
        Returns:
            List of head node data dictionaries
        """
        tail_id = self._find_entity_with_attributes(tail_name, tail_attrs)
        if tail_id is None:
            return []
        
        head_nodes = []
        for head_id, relations_dict in self.relations.items():
            if relation in relations_dict and tail_id in relations_dict[relation]:
                head_entity = self.get_entity_by_id(head_id)
                if head_entity:
                    head_nodes.append(head_entity)
        
        return head_nodes

    def get_related_relations(self, name: str, attrs: Dict[str, Any]) -> List[str]:
        """
        Get all relation types associated with a node.
        
        Args:
            name: Node name
            attrs: Node attributes
            
        Returns:
            List of unique relation names
        """
        target_nodes = self.find_nodes_by_name_and_attributes(name, attrs)
        if not target_nodes:
            return []
        
        target_ids = []
        for node in target_nodes:
            for entity_id, entity_data in self.entities.items():
                if entity_data == node:
                    target_ids.append(entity_id)
                    break
        
        related_relations = set()
        
        for node_id in target_ids:
            if node_id in self.relations:
                related_relations.update(self.relations[node_id].keys())
        
        for head_id, relations in self.relations.items():
            for relation_type, tail_ids in relations.items():
                for node_id in target_ids:
                    if node_id in tail_ids:
                        related_relations.add(relation_type)
                        break
        
        return list(related_relations)

    def _get_entity_id_by_data(self, entity_data: Dict[str, Any]) -> Optional[str]:
        """
        Get entity ID from entity data (name + attributes).
        
        Args:
            entity_data: Entity data dictionary
            
        Returns:
            Entity ID or None if not found
        """
        name = entity_data["name"]
        attrs = entity_data["attributes"]
        
        if name in self.name_to_ids:
            for entity_id in self.name_to_ids[name]:
                if self.entities[entity_id]["attributes"] == attrs:
                    return entity_id
        return None
    
    def _traverse_next_end(self, 
                          current_node: Dict[str, Any], 
                          current_path: List[Tuple[Dict[str, Any], str, Dict[str, Any]]], 
                          visited_ids: Set[str], 
                          all_paths: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]
                          ) -> None:
        """
        Recursively traverse 'next' relations until finding 'end' relations.
        
        Args:
            current_node: Current node being processed
            current_path: Current path being built
            visited_ids: Set of visited entity IDs (to prevent cycles)
            all_paths: List to collect all complete paths
        """
        current_triples = self.find_triples_with_head_node(
            name=current_node["name"],
            attrs=current_node["attributes"]
        )
        
        next_triples = [t for t in current_triples if t[1] == "next"]
        end_triples = [t for t in current_triples if t[1] == "end"]

        for end_triple in end_triples:
            complete_path = current_path.copy()
            complete_path.append(end_triple)
            all_paths.append(complete_path)

        for next_triple in next_triples:
            next_tail_node = next_triple[2]
            next_tail_id = self._get_entity_id_by_data(next_tail_node)

            if next_tail_id is None or next_tail_id in visited_ids:
                continue

            new_path = current_path.copy()
            new_path.append(next_triple)
            new_visited = visited_ids.copy()
            new_visited.add(next_tail_id)

            self._traverse_next_end(
                current_node=next_tail_node,
                current_path=new_path,
                visited_ids=new_visited,
                all_paths=all_paths
            )

    def _traverse_to_target_end(self,
                                current_node: Dict[str, Any],
                                current_path: List[Tuple[Dict[str, Any], str, Dict[str, Any]]],
                                visited_ids: Set[str],
                                end_node_ids: Set[str],
                                all_valid_paths: List[List[Tuple[Dict[str, Any], str, Dict[str, Any]]]]
                                ) -> None:
        """
        Recursively traverse 'next' relations to find paths ending at target nodes.
        
        Args:
            current_node: Current node being processed
            current_path: Current path being built
            visited_ids: Set of visited entity IDs
            end_node_ids: Set of target end node IDs
            all_valid_paths: List to collect valid complete paths
        """
        current_triples = self.find_triples_with_head_node(
            name=current_node["name"],
            attrs=current_node["attributes"]
        )
        
        next_triples = [t for t in current_triples if t[1] == "next"]
        end_triples = [t for t in current_triples if t[1] == "end"]

        for end_triple in end_triples:
            end_tail_node = end_triple[2]
            end_tail_node_id = self._get_entity_id_by_data(end_tail_node)
            if end_tail_node_id in end_node_ids:
                complete_path = current_path.copy()
                complete_path.append(end_triple)
                all_valid_paths.append(complete_path)

        for next_triple in next_triples:
            next_tail_node = next_triple[2]
            next_tail_id = self._get_entity_id_by_data(next_tail_node)

            if next_tail_id is None or next_tail_id in visited_ids:
                continue

            new_path = current_path.copy()
            new_path.append(next_triple)
            new_visited = visited_ids.copy()
            new_visited.add(next_tail_node_id)

            self._traverse_to_target_end(
                current_node=next_tail_node,
                current_path=new_path,
                visited_ids=new_visited,
                end_node_ids=end_node_ids,
                all_valid_paths=all_valid_paths
            )
            
    @classmethod
    def load_from_file(cls, file_path: str) -> "ThoughtActionGraph":
        """
        Load a ThoughtActionGraph from a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Loaded ThoughtActionGraph instance
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        tag = cls()
        tag.entities = data["entities"]
        tag.relations = data["relations"]
        tag.name_to_ids = data["name_to_ids"]
        tag.next_id = data["next_id"]
        
        return tag
