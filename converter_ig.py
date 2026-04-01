"""igraph version of converter.py - Load Neo4j graphs as igraph objects instead of NetworkX."""

from neo4j import GraphDatabase
import igraph as ig
import logging
from typing import Optional, Tuple, List, Dict
import argparse
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jToIgraph:
    def __init__(self, uri: str, username: str, password: str, batch_size: int = 10000):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.batch_size = batch_size
        
    def _get_node_count(self, session) -> int:
        query = "MATCH (n) RETURN count(n) as count"
        result = session.run(query)
        return result.single()["count"]
    
    def _get_edge_count(self, session) -> int:
        query = "MATCH ()-[r]-() RETURN count(r) as count"
        result = session.run(query)
        return result.single()["count"]

    def load_simplified_graph(self) -> ig.Graph:
        """
        Load graph as igraph object with simplified attributes.
        Only preserves essential attributes in a format suitable for processing.
        
        Returns:
            igraph.Graph object with nodes and edges loaded from Neo4j
        """
        try:
            edges = []
            node_mapping = {}
            current_node_idx = 0
            node_labels = {}
            node_ids = {}
            edge_types = {}
            
            with self.driver.session() as session:
                # Get total counts
                total_nodes = self._get_node_count(session)
                total_edges = self._get_edge_count(session)
                
                # Process nodes
                logger.info("Processing nodes...")
                skip = 0
                while skip < total_nodes:
                    query = """
                    MATCH (n)
                    RETURN id(n) as node_id, 
                           labels(n) as labels,
                           n.id as custom_id,
                           n.label as custom_label
                    SKIP $skip LIMIT $limit
                    """
                    result = session.run(query, skip=skip, limit=self.batch_size)
                    
                    for record in result:
                        node_id = record["node_id"]
                        if node_id not in node_mapping:
                            node_mapping[node_id] = current_node_idx
                            
                            # Use custom label if available, otherwise use first Neo4j label
                            custom_label = record["custom_label"]
                            neo4j_labels = record["labels"]
                            display_label = (custom_label or 
                                          (neo4j_labels[0] if neo4j_labels else "Node"))
                            
                            # Store node attributes
                            node_labels[current_node_idx] = str(display_label)
                            node_ids[current_node_idx] = str(record["custom_id"] or node_id)
                            current_node_idx += 1
                    
                    skip += self.batch_size
                
                # Process edges with minimal attributes
                logger.info("Processing edges...")
                skip = 0
                edge_idx = 0
                while skip < total_edges:
                    query = """
                    MATCH (n)-[r]-(m)
                    RETURN id(n) as source, id(m) as target, 
                            type(r) as edge_type
                    SKIP $skip LIMIT $limit
                    """
                    result = session.run(query, skip=skip, limit=self.batch_size)
    
                    for record in result:
                        src = node_mapping[record["source"]]
                        dst = node_mapping[record["target"]]
                        edge_types[edge_idx] = str(record["edge_type"])
                        edges.append((src, dst))
                        edge_idx += 1
    
                    skip += self.batch_size
                
                # Create igraph
                G = ig.Graph(edges)
                
                # Add node attributes
                for v_id in G.vs:
                    v_id['label'] = node_labels.get(v_id.index, "Node")
                    v_id['id'] = node_ids.get(v_id.index, str(v_id.index))
                
                # Add edge attributes
                for e_id, edge in enumerate(G.es):
                    edge['weight'] = 1.0
                    edge['type'] = edge_types.get(e_id, "unknown")
                
                logger.info(f"Loaded graph with {len(G.vs)} nodes and {len(G.es)} edges")
                return G
                
        except Exception as e:
            logger.error(f"Error loading graph from Neo4j: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Neo4j to igraph converter')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j connection URI')
    parser.add_argument('--username', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--output', type=str, default='graph_ig.pkl',
                        help='Output file for pickled igraph')
    args = parser.parse_args()
    
    converter = Neo4jToIgraph(args.uri, args.username, args.password)
    graph = converter.load_simplified_graph()
    
    # Save graph
    with open(args.output, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Graph saved to {args.output}")

if __name__ == '__main__':
    main()
