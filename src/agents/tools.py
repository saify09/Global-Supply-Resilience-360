from langchain_core.tools import tool
from typing import List, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.graph_client import GraphClient

# Initialize global graph client
# In a real app, this would be dependency injected or managed by a singleton
graph_client = GraphClient(use_mock=True) # Default to mock for safety

@tool
def get_entity_info(entity_id: str) -> Dict:
    """
    Get detailed information about a specific entity (Company, Supplier, Port, etc.) by its ID.
    Returns the node properties.
    """
    node = graph_client.get_node(entity_id)
    if node:
        return node
    return {"error": "Entity not found"}

@tool
def find_downstream_impact(entity_id: str) -> List[Dict]:
    """
    Identify which companies or products are downstream of a given entity.
    Useful for assessing who is affected if a supplier or port fails.
    """
    impacted = graph_client.get_downstream_impact(entity_id, depth=3)
    # Simplify output for context window
    return [{"id": n.get('id', 'N/A'), "name": n.get('name', 'Unknown'), "type": n.get('label', 'Node')} for n in impacted[:50]]

@tool
def find_upstream_dependencies(entity_id: str) -> List[Dict]:
    """
    Identify identifying upstream suppliers for a company or product.
    Useful for finding the root cause of a shortage.
    """
    deps = graph_client.get_upstream_suppliers(entity_id, depth=3)
    return [{"id": n.get('id', 'N/A'), "name": n.get('name', 'Unknown'), "tier": n.get('tier', '?')} for n in deps[:50]]

@tool
def calculate_risk_exposure(entity_id: str) -> Dict:
    """
    Calculates a risk score (0-100) for an entity based on its dependencies.
    """
    # Logic: Average risk of upstream deps
    deps = graph_client.get_upstream_suppliers(entity_id, depth=2)
    if not deps:
        return {"risk_score": 0, "reason": "No dependencies found"}
    
    total_risk = 0
    count = 0
    for d in deps:
        # Mock risk calculation
        r = float(d.get('risk_score', 0) if d.get('risk_score') else 50)
        total_risk += r
        count += 1
        
    avg_risk = total_risk / count if count > 0 else 0
    return {"entity_id": entity_id, "upstream_risk_avg": round(avg_risk, 2), "dependency_count": count}
