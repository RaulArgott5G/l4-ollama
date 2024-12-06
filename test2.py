from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def multiply(a: float, b: float) -> float:
    """Multiply a and b.

    Args:
        a: first float
        b: second float
    """
    return a * b

def add(a: float, b: float) -> float:
    """Add a and b.
    
    Args:
        a: first float number to add 
        b: second float number to add
    """
    return a + b

def divide(a: float, b: float) -> float:
    """Divide a and b.
    
    Args:
        a: first float number to divide
        b: second float number to divide
    """
    return a / b

llm = ChatOllama(model="llama3.1", temperature=0)

llm_with_tools = llm.bind_tools([add, multiply, divide])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, add, divide]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()


messages = [HumanMessage(content="Could you perfom the following operation?: [(53x3 + 2)/2] + 35")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()