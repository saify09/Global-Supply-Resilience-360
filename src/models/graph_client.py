import networkx as nx
import pandas as pd
import networkx as nx
import pandas as pd
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "password"), use_mock=False):
        self.use_mock = use_mock
        self.driver = None
        self.nx_graph = None
        
        if not self.use_mock:
            if GraphDatabase is None:
                logger.warning("Neo4j module not found. Falling back to NetworkX mock.")
                self.use_mock = True
            else:
                try:
                    self.driver = GraphDatabase.driver(uri, auth=auth)
                    self.driver.verify_connectivity()
                    logger.info("Connected to Neo4j.")
                except Exception as e:
                    logger.warning(f"Neo4j connection failed: {e}. Falling back to NetworkX mock.")
                    self.use_mock = True
        
        if self.use_mock:
            self._load_mock_graph()

    def _load_mock_graph(self):
        logger.info("Loading NetworkX mock graph from CSVs...")
        data_dir = Path("data/raw")
        if not data_dir.exists():
            raise FileNotFoundError("Data directory not found. Run data_gen.py first.")
            
        self.nx_graph = nx.DiGraph()
        
        # Load Nodes
        for f, label, id_col in [
            ("companies.csv", "Company", "id"),
            ("suppliers.csv", "Supplier", "id"),
            ("products.csv", "Product", "id"),
            ("components.csv", "Component", "id"),
            ("ports.csv", "Port", "port_id"),
            ("countries.csv", "Country", "country_id")
        ]:
            if (data_dir / f).exists():
                df = pd.read_csv(data_dir / f)
                for _, row in df.iterrows():
                    self.nx_graph.add_node(row[id_col], label=label, **row.to_dict())
        
        # Load Edges
        if (data_dir / "relationships.csv").exists():
            df_rel = pd.read_csv(data_dir / "relationships.csv")
            for _, row in df_rel.iterrows():
                attrs = row.to_dict()
                u = attrs.pop('from_id')
                v = attrs.pop('to_id')
                # 'type' is already in attrs
                self.nx_graph.add_edge(u, v, **attrs)
                
        logger.info(f"Mock Graph Loaded: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges.")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_node(self, node_id):
        if self.use_mock:
            if self.nx_graph.has_node(node_id):
                return self.nx_graph.nodes[node_id]
            return None
        else:
            query = "MATCH (n {id: $id}) RETURN n"
            with self.driver.session() as session:
                result = session.run(query, id=node_id)
                record = result.single()
                return dict(record['n']) if record else None

    def get_upstream_suppliers(self, node_id, depth=3):
        """Finds who supplies TO this node (recursive)."""
        if self.use_mock:
            # NetworkX predecessors/traversal
            # Note: SOURCES_FROM is usually Supplier -> Company (Flow of Goods) 
            # Wait, usually Supplier supplies Company. So Edge is Supplier -> Company?
            # In my data_gen: Company -> SOURCES_FROM -> Supplier.
            # So the edge direction is Buyer -> Supplier.
            # So "upstream" means walking the SOURCES_FROM edges.
            
            subgraph = nx.bfs_tree(self.nx_graph, node_id, depth_limit=depth, reverse=False)
            # This gives successors in 'SOURCES_FROM' direction (Nodes I buy from)
            nodes = []
            for n in subgraph.nodes():
                if n != node_id:
                     nodes.append(self.nx_graph.nodes[n])
            return nodes
        else:
            query = f"""
            MATCH (n {{id: $id}})-[:SOURCES_FROM*1..{depth}]->(m)
            RETURN distinct m
            """
            with self.driver.session() as session:
                result = session.run(query, id=node_id)
                return [dict(record['m']) for record in result]

    def get_downstream_impact(self, node_id, depth=3):
        """Finds who this node supplies TO (Impact if this node fails)."""
        # Edge: Buyer -> Supplier
        # So if Supplier fails, we look at who points TO it.
        # i.e. Incoming Edges of SOURCES_FROM.
        if self.use_mock:
            # Reverse BFS (traverse incoming edges)
            # NetworkX: predecessors = sources of edges pointing to me
            # Since edges are Buyer -> Supplier, predecessors are Buyers.
            # Correct.
            
            # Simple BFS on reversed graph
            rev_graph = self.nx_graph.reverse()
            subgraph = nx.bfs_tree(rev_graph, node_id, depth_limit=depth)
            
            nodes = []
            for n in subgraph.nodes():
                if n != node_id:
                     nodes.append(self.nx_graph.nodes[n])
            return nodes
        else:
            query = f"""
            MATCH (m)-[:SOURCES_FROM*1..{depth}]->(n {{id: $id}})
            RETURN distinct m
            """
            with self.driver.session() as session:
                result = session.run(query, id=node_id)
                return [dict(record['m']) for record in result]

    def export_for_gnn(self):
        """
        Exports the graph topology and features for PyTorch Geometric.
        Returns: (x, edge_index, node_map)
        """
        # For simplicity, we implement this for the Mock/NetworkX version 
        # as accessing raw Neo4j data for GNN often involves dumping to memory anyway.
        
        if not self.use_mock:
            # Load graph into memory from Neo4j or switch to mock loader
            # For this demo, let's just force load mock
            self._load_mock_graph()
            
        # 1. Map all nodes to integer indices
        node_ids = list(self.nx_graph.nodes())
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        
        # 2. Edge Index
        src = []
        dst = []
        for u, v in self.nx_graph.edges():
            src.append(node_map[u])
            dst.append(node_map[v])
            
        # 3. Features (Dummy features for now + risk score if exists)
        # We'll create a simple feature vector: [Risk, Reliability, Tier]
        # In a real app we'd one-hot encode Industry, Country, etc.
        import torch
        
        features = []
        for nid in node_ids:
            node = self.nx_graph.nodes[nid]
            risk = float(node.get('risk_score', 0.5)) if 'risk_score' in node else 0.5
            tier = float(node.get('tier', 0)) / 3.0  # Normalize
            features.append([risk, tier])
            
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return x, edge_index, node_map
