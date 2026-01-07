import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY) from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.graph_client import GraphClient
from src.agents.agent import app as agent_app
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Global Supply Resilience 360", layout="wide")

@st.cache_resource
def get_graph_client():
    return GraphClient(use_mock=True)

client = get_graph_client()

st.title("🌐 Global Supply Resilience 360")
st.markdown("### Agentic AI for Supply Chain Risk Management")

# Sidebar
st.sidebar.header("Control Panel")
mode = st.sidebar.selectbox("Mode", ["Dashboard", "Graph Explorer", "Agent Chat", "Simulation"])

if mode == "Dashboard":
    st.subheader("High-Level KPIs")
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock Metrics loaded from graph
    n_companies = len([n for n, d in client.nx_graph.nodes(data=True) if d.get('label') == 'Company'])
    n_suppliers = len([n for n, d in client.nx_graph.nodes(data=True) if d.get('label') == 'Supplier'])
    n_alerts = 12 # Mock
    avg_risk = 45.2 # Mock
    
    col1.metric("Companies Monitored", n_companies)
    col2.metric("Active Suppliers", n_suppliers)
    col3.metric("Critical Alerts", n_alerts, delta="-2")
    col4.metric("Global Risk Index", avg_risk, delta="1.2%")
    
    st.divider()
    
    # Map Visualization (Ports)
    st.subheader("Global Port Status")
    ports = []
    for n, d in client.nx_graph.nodes(data=True):
        if d.get('label') == 'Port':
            # Generate fake coords for map demo
            import random
            ports.append({
                "name": d.get('name'), 
                "lat": random.uniform(-50, 60), 
                "lon": random.uniform(-120, 140),
                "status": d.get('status', 'Active')
            })
    df_ports = pd.DataFrame(ports)
    if not df_ports.empty:
        st.map(df_ports, size=20)
    else:
        st.info("No port data found.")

elif mode == "Graph Explorer":
    st.subheader("Supply Chain Network")
    
    # Search for an entity
    search_term = st.text_input("Search Entity ID (e.g., COM_0001, SUP_T1_00001)")
    
    if search_term:
        node = client.get_node(search_term)
        if node:
            st.json(node)
            
            # Show immediate neighbors
            neighbors = client.get_upstream_suppliers(search_term, depth=1)
            st.write(f"Upstream Suppliers: {len(neighbors)}")
            
            # Simple visualization of subgraph
            if neighbors:
                subset_nodes = [search_term] + [n['id'] for n in neighbors]
                subgraph = client.nx_graph.subgraph(subset_nodes)
                
                # Plotly Network
                edge_x = []
                edge_y = []
                # Simple random layout
                pos = nx.spring_layout(subgraph)
                
                for edge in subgraph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')

                node_x = []
                node_y = []
                node_text = []
                for node_id in subgraph.nodes():
                    x, y = pos[node_id]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node_id)

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(size=10, color='blue')
                )
                
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='Dependency Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Entity not found.")

elif mode == "Agent Chat":
    st.subheader("💬 AI Supply Chain Analyst")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about risks (e.g., 'Analyze risk for COM_0001')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Invoke Agent
                inputs = {"messages": [HumanMessage(content=prompt)]}
                # Handling streaming/response
                # For simplicity in this demo, strict invoke
                result = agent_app.invoke(inputs)
                ai_msg = result['messages'][-1].content
                full_response = ai_msg
            except Exception as e:
                # Automatic Fallback if API fails (e.g. Quota exceeded)
                error_msg = str(e)
                if "insufficient_quota" in error_msg or "RateLimitError" in error_msg:
                    fallback_msg = "⚠️ **OpenAI Quota Exceeded / API Error**.\n\n*Switching to Simulation Mode response:*\n\nBased on the graph topology, this entity has significant upstream dependencies. Risk is elevated."
                    message_placeholder.markdown(fallback_msg)
                    full_response = fallback_msg
                else:
                    full_response = f"Agent Error: {str(e)}\n\n(Check OpenAI API Key in .env)"
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

elif mode == "Simulation":
    st.subheader("What-If Scenario Simulation")
    
    scenario = st.selectbox("Select Scenario", ["Port Closure", "Supplier Bankruptcy", "Geopolitical Embargo"])
    target = st.text_input("Target Entity ID (e.g., COM_0001, SUP_T1_00001)")
    
    if st.button("Run Simulation"):
        if not target:
            st.error("Please enter a Target Entity ID.")
        else:
            node = client.get_node(target)
            if not node:
                 st.error(f"Entity {target} not found in the graph.")
            else:
                st.write(f"Simulating **{scenario}** on **{target}**...")
                
                # Dynamic Graph Analysis
                # 1. Find Downstream Impact (Who buys from this entity?)
                downstream_impact = client.get_downstream_impact(target, depth=3)
                impact_count = len(downstream_impact)
                
                # 2. Estimate Revenue at Risk (Mock calculation based on impacted nodes)
                # Assume each downstream node represents $100k - $1M revenue
                import random
                est_loss = impact_count * random.uniform(0.1, 0.5) 
                
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.success(f"Simulation Complete for {target}")
                
                st.metric("Entities Impacted", impact_count)
                st.metric("Est. Revenue Risk", f"${est_loss:.2f}M")
                
                if downstream_impact:
                    st.write("### Impacted Entities (Downstream)")
                    df_impact = pd.DataFrame(downstream_impact)
                    
                    # Define desired columns
                    cols_to_show = ['id', 'label']
                    if 'risk_score' in df_impact.columns:
                        cols_to_show.append('risk_score')
                    
                    # Filter to available columns
                    available_cols = [c for c in cols_to_show if c in df_impact.columns]
                    
                    if available_cols:
                        st.dataframe(df_impact[available_cols].head(10))
                    else:
                        st.dataframe(df_impact.head())
                else:
                    st.info("No downstream entities found (End of Chain).")
