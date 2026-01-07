import os
import time
from neo4j import GraphDatabase
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ingestion.data_schema import *

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

class GraphLoader:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        
    def close(self):
        self.driver.close()
        
    def check_connection(self):
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_constraints(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cnt:Country) REQUIRE cnt.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (prt:Port) REQUIRE prt.id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("Constraints created.")

    def load_nodes(self, file_path, label, id_col, props_map=None):
        # We will assume client runs this where 'data/raw' is accessible 
        # But Neo4j runs in container. 
        # For simplicity, we'll read CSV in Python and batch insert.
        # This is slower but works without detailed volume mapping debugging.
        
        print(f"Loading {label} from {file_path}...")
        df = pd.read_csv(file_path)
        
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{id: row.{id_col}}})
        SET n += row
        """
        
        # Batch processing
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size].to_dict('records')
                session.run(query, rows=batch)
        print(f"Loaded {len(df)} {label} nodes.")

    def load_relationships(self, file_path):
        print(f"Loading Relationships from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Group by type to optimize
        for rel_type in df['type'].unique():
            sub_df = df[df['type'] == rel_type]
            print(f"  Processing {rel_type} ({len(sub_df)})...")
            
            query = f"""
            UNWIND $rows AS row
            MATCH (source {{id: row.from_id}})
            MATCH (target {{id: row.to_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += row
            """
            
            batch_size = 1000
            with self.driver.session() as session:
                for i in range(0, len(sub_df), batch_size):
                    batch = sub_df.iloc[i:i+batch_size].to_dict('records')
                    session.run(query, rows=batch)

if __name__ == "__main__":
    loader = GraphLoader(URI, AUTH)
    if loader.check_connection():
        loader.clear_database()
        loader.create_constraints()
        
        data_dir = Path("data/raw")
        
        loader.load_nodes(data_dir / "countries.csv", "Country", "country_id")
        loader.load_nodes(data_dir / "ports.csv", "Port", "port_id")
        loader.load_nodes(data_dir / "companies.csv", "Company", "id")
        loader.load_nodes(data_dir / "suppliers.csv", "Supplier", "id")
        loader.load_nodes(data_dir / "products.csv", "Product", "id")
        loader.load_nodes(data_dir / "components.csv", "Component", "id")
        
        loader.load_relationships(data_dir / "relationships.csv")
        
        loader.close()
    else:
        print("Could not connect to Neo4j. Skipping DB load.")
