from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.tools import get_entity_info, find_downstream_impact, find_upstream_dependencies, calculate_risk_exposure

# Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define Tools
tools = [get_entity_info, find_downstream_impact, find_upstream_dependencies, calculate_risk_exposure]
# tool_executor removed as it is not used and deprecated


# Define Model (Using Mock/Print if no API Key, ideally would be GPT-4)
import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import FakeListChatModel

if "OPENAI_API_KEY" not in os.environ or \
   os.environ["OPENAI_API_KEY"].startswith("sk-dummy") or \
   os.environ["OPENAI_API_KEY"] == "sk-...":
    print("WARNING: No valid OPENAI_API_KEY found. Using Fake Chat Model.")
    # Fallback to a fake model that returns pre-canned responses for demo
    model = FakeListChatModel(responses=[
        "Based on the analysis, Company COM_0001 has a generic high risk exposure due to upstream dependencies.",
        "The simulation suggests a 45% probability of cascading failure if the port closes.",
        "Downstream impact includes 12 Tier-1 suppliers and $4M in potential revenue loss."
    ])
else:
    model = ChatOpenAI(temperature=0, streaming=True)
    model = model.bind_tools(tools)


def agent_node(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Define Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

if __name__ == "__main__":
    # Test Run
    print("Testing Agent...")
    try:
        inputs = {"messages": [HumanMessage(content="What are the risks for Company COM_0001?")]}
        for s in app.stream(inputs):
            print(list(s.values())[0])
            print("----")
    except Exception as e:
        print(f"Agent Execution Failed (Likely due to missing API Key): {e}")
