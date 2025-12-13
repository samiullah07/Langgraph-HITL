from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

@tool
def get_stock_price(symbol: str):
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    stock_api   =os.getenv('stock_API_KEY') 
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api}"
    )
    r = requests.get(url)
    data = r.json()    
    return data

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    NOTE: This is a mock implementation:
    - No real brokerage API is called.
    - It simply returns a confirmation payload.
    """
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    user_message = state["messages"]
    response = llm_with_tools.invoke(user_message)
    return {"messages":[response]}


tool_node = ToolNode(tools)
memory = MemorySaver()



graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    thread_id = "demo-thread"

    system_msg = SystemMessage(
        content=(
            "You are a stock trading assistant. "
            "Use tools when needed. "
            "For prices, call get_stock_price. "
            "For buying stocks, call purchase_stock."
        )
    )
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        state = {
            "messages": [
                system_msg,
                HumanMessage(content=user_input)
            ]
        }
        response = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )


         # Get the latest message from the assistant
        messages = response["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")
