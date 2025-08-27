from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agent.tools import tools
import json


class State(TypedDict):
    messages: Annotated[list, add_messages]
    

llm = init_chat_model(model="azure_openai:gpt-4.1", azure_deployment="gpt-4.1")



def planner_node(state: State):
    
    str_tools = "\n".join([str(tool.model_dump()) for tool in tools])
    
    prompt = f"""You are a planning agent. Given the following conversation, determine the next steps to take.
    Your output will be used as a guide for another agent to respond appropriately. You are not to respond directly to the user.
    The available tools the agent can use are:
    {str_tools}
    DO NOT CALL THE TOOLS YOURSELF, JUST PLAN.
    """
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            MessagesPlaceholder("messages"),
        ]
    )
    
    formated_prompt = chat_prompt.format_prompt(messages=state["messages"])
    response = llm.invoke(formated_prompt)
    state["messages"] = AIMessage(content=f"<plannig>{response.content}<planning>")
    return state

def chatbot(state: State):
    prompt = """You are a helpful assistant. Continue the conversation based on the messages provided."""
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            MessagesPlaceholder("messages"),
        ]
    )
    formated_prompt = chat_prompt.format_prompt(messages=state["messages"])
    response = llm.bind_tools(tools).invoke(formated_prompt)
    state["messages"] = response
    return state


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("planner", planner_node)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "planner")

graph_builder.add_edge("chatbot", END)