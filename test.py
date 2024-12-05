from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

@tool
def add(a: int, b: int) -> int:
    """Sum two integer numbers

    Args:
       a (int): the first number to sum
       b (int): the second number to sum

    """
    return a + b

@tool
def mul(a: float, b: float) -> float:
    """Multiply two numbers

    Args:
        a (float): first number to multiply
        b (float): second number to multiply

    """
    return a * b

@tool
def div(a: float, b: float) -> float:
    """Divide two numbers

    Args:
        a (float): first number to divide
        b (float): second number to divide

    """
    return a / b

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools([add, mul, div])


user_query = input("Please enter your query: ")


messages = [HumanMessage(user_query)]

ai_msg = llm.invoke(messages)

print(ai_msg.tool_calls)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "mul": mul, "div": div}[tool_call["name"].lower()]
    result = selected_tool.invoke(tool_call)
    messages.append(result)

    print(f"\nResult of the operation: {tool_call['name']}: {result.content}")