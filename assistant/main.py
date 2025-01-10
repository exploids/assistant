import os
import glob
from pathlib import Path
import sys
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.types import Command, interrupt
from langchain_community.tools import ShellTool
from pypdf import PdfReader
from pydantic import BaseModel


trusted = ["list_files", "search_files", "duckduckgo_results_json", "ask_human"]
# trusted += ["read_file", "read_pdf_as_text"]


class AskHuman(BaseModel):
  """Ask the user a question."""

  question: str


@tool
def read_file(path: str):
  """Read a UTF-8 encoded file and return its contents. The content might be cut off, if the file is too large."""
  with open(path, "r") as f:
    return f.read(8192)


@tool
def read_pdf_as_text(path: str):
  """Read a PDF file and return its contents as text. The content might be cut off, if the file is too large."""
  reader = PdfReader(path)
  text = ""
  for page in reader.pages:
    text += page.extract_text() + "\n"
  return text[:8192]


@tool
def list_files(path: str):
  """List all files in a directory. The format is 'file path | type | size'. The output might be cut off, if there are too many files."""
  return "\n".join(f"{entry.path} | {"file" if entry.is_file() else ("directory" if entry.is_dir() else "other")} | {entry.stat().st_size}" for entry in os.scandir(path))[:8192]


def insensitive_glob_pattern(pattern):
  def either(c):
    return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
  return ''.join(map(either, pattern))


@tool
def search_files(path: str, pattern: str, recursive: bool = True, case_sensitive: bool = False, prepend_and_append_wildcard: bool = True):
  """Find files matching a glob pattern in the given directory. Use * to match any character. Use ** to match zero or more directories. Use [Aa] to match different characters. The output might be cut off, if there are too many matches."""
  if not case_sensitive:
    pattern = insensitive_glob_pattern(pattern)
  if prepend_and_append_wildcard:
    pattern = f"*{pattern}*"
  return "\n".join(os.path.join(path, match) for match in glob.glob("**/" + pattern if recursive else pattern, root_dir=path, recursive=recursive))[:8192]


tools = [list_files, search_files, read_file, read_pdf_as_text, DuckDuckGoSearchResults(), ShellTool(), AskHuman]

tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
# model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
# model = ChatOllama(model="llama3.2").bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["ask_human", "human_review", "tools", END]:
  messages = state['messages']
  last_message = messages[-1]
  # If the LLM makes a tool call, then we route to the "tools" node
  if last_message.tool_calls and last_message.tool_calls[0]["name"] == "AskHuman":
    return "ask_human"
  elif last_message.tool_calls:
    for tool_call in last_message.tool_calls:
      if tool_call["name"] not in trusted:
        return "human_review"
    return "tools"
  # Otherwise, we stop (reply to the user)
  return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def human_review_node(state) -> Command[Literal["agent", "tools"]]:
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    human_review = interrupt(
        {
            "question": "Allow?",
            "tool_calls": tool_calls,
        }
    )

    review_action = human_review["action"]
    review_data2 = human_review.get("data")

    # if approved, call the tool
    if review_action == "continue":
        return Command(goto="tools")

    # update the AI message AND call tools
    elif review_action == "update":
        updated_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    # This the update provided by the human
                    "args": review_data,
                }
            for tool_call, review_data in zip(tool_calls, review_data2)],
            # This is important - this needs to be the same as the message you replacing!
            # Otherwise, it will show up as a separate message
            "id": last_message.id,
        }
        return Command(goto="tools", update={"messages": [updated_message]})

    # provide feedback to LLM
    elif review_action == "feedback":
        # NOTE: we're adding feedback message as a ToolMessage
        # to preserve the correct order in the message history
        # (AI messages with tool calls need to be followed by tool call messages)
        tool_messages = [{
            "role": "tool",
            # This is our natural language feedback
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        } for tool_call, review_data in zip(tool_calls, review_data2)]
        return Command(goto="agent", update={"messages": tool_messages})

def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    reply = interrupt(state["messages"][-1].tool_calls[0]["args"]["question"])
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": reply}]
    return {"messages": tool_message}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

workflow.add_edge("ask_human", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

def cli():
  prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("🧑 >>> ")
  initial_input = {"messages": [SystemMessage(content=f"You are a helpful assistant. When providing information, always cite your source.\n\nUser home directory: {Path.home()}"), HumanMessage(content=prompt)]}
  thread = {"configurable": {"thread_id": 42}}
  while True:
    for event in app.stream(initial_input, thread, stream_mode="updates"):
      if "agent" in event:
        calls = event["agent"]["messages"][-1].additional_kwargs.get("tool_calls")
        if calls is None:
          print("🤖 ASSISTANT:", event["agent"]["messages"][-1].content)
      #   else:
      #     for call in calls:
      #       print(f"🛠️  {call["function"]['name']} {call["function"]['arguments']}")
      # elif "tools" in event:
      #   print(f"↩️  {event["tools"]["messages"][-1].name} \"{event["tools"]["messages"][-1].content.replace('\n', ' ')[:60]}\"")
      # else:
      #   print(event)
    if "__interrupt__" in event:
      data = event.get("__interrupt__")[0].value
      if "tool_calls" in data:
        do_continue = True
        feedbacks = []
        print("⚠️  The assistant wants to run the following tools:")
        for tool_call in data["tool_calls"]:
          if not tool_call["name"] in trusted:
            print("   🛠️ ", tool_call["name"], tool_call["args"])
        answer = input(f"Allow all? (y/n) ")
        if answer.lower() == "y":
          initial_input = Command(resume={"action": "continue"})
        else:
          for tool_call in data["tool_calls"]:
            answer = "y" if tool_call["name"] in trusted else input(f"   🛠️  {tool_call['name']} {tool_call['args']} Allow this tool? (y/n) ")
            if answer.lower() == "y":
              feedbacks.append("This tool was approved, but others were not.")
            else:
              do_continue = False
              feedback = input("Feedback (optional): ")
              feedbacks.append(feedback or "The user denied the execution of this tool with the specified arguments.")
          initial_input = Command(resume={"action": "continue" if do_continue else "feedback", "data": feedbacks})
      else:
        print(f"❓ ASSISTANT: {data}")
        reply = input("🧑 >>> ")
        initial_input = Command(resume="The user refused to answer, likely because they want you to use other tools." if len(reply.strip()) == 0 else reply)
    else:
      reply = input("🧑 >>> (leave blank to exit) ")
      if len(reply.strip()) == 0:
        break
      initial_input = {"messages": [HumanMessage(content=reply)]}

if __name__ == "__main__":
  cli()
