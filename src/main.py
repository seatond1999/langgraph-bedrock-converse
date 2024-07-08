# %% --------------------------------------------------------------------------
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
import os
import boto3
from typing import Annotated, TypedDict
from langchain_core.tools import StructuredTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langgraph.graph import END, StateGraph
from langchain.agents import tool
import string
import operator

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv(".env")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

# %% --------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------


def get_llm() -> ChatBedrockConverse:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )

    model = ChatBedrockConverse(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=1000,
        temperature=0.1,
    )

    return model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    chat_history: ConversationBufferWindowMemory


class Agent:
    def __init__(
        self,
        model: ChatBedrockConverse,
        tools: list[StructuredTool],
        checkpointer,
    ):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_bedrock)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.llm = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.graph = graph.compile(checkpointer=checkpointer)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_bedrock(self, state: AgentState):

        messages = state["messages"]
        print(messages)
        message = self.llm.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}


# %% --------------------------------------------------------------------------
@tool
def get_forename(input: str) -> str:
    """Return the forename of the user's favourite tennis player. The input to this function is the string "forename"."""
    return "Andy"


@tool
def get_surname(input: str) -> str:
    """Return the surname of the user's favourite tennis player. The input to this function is the string "surname"."""
    return "Murray"


@tool
def get_name_alphabet_positions(full_name: str) -> list[int]:
    """Returns the alphabet positions of the user's favourite tennis player. The input to this function must be the first name, second name or full name of the tennis player which you should retrieve from a Previous Conversation, or using the get_forename and/or the get_surname function."""
    name_alphabet_positions = []
    alphabet_positions = {letter: position + 1 for position, letter in enumerate(string.ascii_lowercase)}
    alphabet_positions[" "] = 0
    for i in full_name:
        name_alphabet_positions.append(alphabet_positions[i.lower()])
    return name_alphabet_positions


# %% --------------------------------------------------------------------------


memory = SqliteSaver.from_conn_string(":memory:")
thread = {"configurable": {"thread_id": "1"}}
tools = [get_forename, get_surname, get_name_alphabet_positions]
prompt = "what is the full name of my favourite tennis player"
abot = Agent(get_llm(), tools, memory)
result = abot.graph.invoke({"messages": [HumanMessage(content=prompt)]}, thread)
result
