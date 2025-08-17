import os
import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

# Load secrets from Streamlit's secrets management
LLM_KEY = st.secrets["OPENAI"]["API_KEY"]
LLM_MODEL = st.secrets["OPENAI"]["MODEL"]
TAVILY_API_KEY = st.secrets["TAVILY"]["API_KEY"]  

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    summary: str

class Agent:
    def __init__(self):

        tavily_search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, 
                                          include_images=False,
                                          include_image_descriptions=False,
                                          max_results=4)
        tools = [tavily_search_tool]
        self.tool_names = {t.name: t for t in tools } 
        llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_KEY, temperature=1)
        """
        By disabling parallel tool calls, I ensured that:
        Each tool call is associated with a unique output (tool_call_id).
        The correct tool outputs are sent back to the LLM for subsequent steps.
        """
        self.llm = llm.bind_tools(tools=tools, parallel_tool_calls=False)

        today = datetime.date.today().isoformat()
        self.sys_msg = (
            "You are a helpful, concise, and knowledgeable AI assistant. "
            "Use external tools when needed to provide accurate , up-to-date or current information. "
            "If you are unsure, ask clarifying questions. "
            "Always be polite and clear in your responses."
            f"Today is {today}"
        )
        self.n_messages = 12
        # Build graph
        graph = StateGraph(MessagesState)
        graph.add_node("assistant", self.assistant)
        graph.add_node("apply_tools", self.apply_tools)
        graph.add_node("summarize_conversation", self.summarize_conversation)
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges(
            "assistant",
            self.exists_tool,
            {True: "apply_tools", False: "summarize_conversation"} # if the function returns True, go to apply_tools, otherwise end the graph
            )
        graph.add_edge("apply_tools", "assistant") # This edge allows returning to the LLM after tool execution
        #graph.add_conditional_edges("assistant", self.should_continue)
        graph.add_edge("summarize_conversation", END)
        checkpointer = MemorySaver()
        self.graph = graph.compile(checkpointer=checkpointer)
        # Save the graph as an image
        #graph_image = self.graph.get_graph().draw_mermaid_png()
        #with open("graph.png", "wb") as f:
        #    f.write(graph_image)


    # Define the logic to call the model
    def assistant(self, state: MessagesState):
        """Call the LLM with the current messages in the state."""
        summary = state.get("summary", "")

        # Construir mensaje de sistema
        if summary:
            system_message = (
                f"{self.sys_msg}\n"
                f"Summary of conversation earlier: {summary}"
            )
        else:
            system_message = self.sys_msg

        messages = [SystemMessage(content=system_message)] + state["messages"] # valid_messages

        response = self.llm.invoke(messages)
        return {"messages": response}


    def summarize_conversation(self, state: MessagesState):
        messages = state["messages"]
        # First, we get any existing summary
        summary = state.get("summary", "")
        delete_messages = []
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > self.n_messages:
            print("Summarizing conversation...")
            """Summarize the conversation so far."""
            # Create our summarization prompt 
            if summary:
                # A summary already exists
                summary_message = (
                    f"This is summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages above:"
                )
            else:
                summary_message = "Create a summary of the conversation above:"

            # Add prompt to our history
            messages = state["messages"] + [HumanMessage(content=summary_message)]
            response = self.llm.invoke(messages)
            summary =  response.content
            # Delete all but the 2 most recent messages
            messages_to_delete = state["messages"][:-2]
            # If the last message is an AI message with tool calls, we delete the last three messages
            # This logic ensures that the assistant’s tool call and the tool’s response always stay together in the message history,
            #  which is required by the OpenAI function calling protocol.
            if messages_to_delete[-1].type == "ai" and len(messages_to_delete[-1].tool_calls) >0:
                messages_to_delete = state["messages"][:-3]
           
            delete_messages = [RemoveMessage(id=m.id) for m in messages_to_delete]

        return {"summary": summary, "messages": delete_messages}

    def should_continue(self, state: MessagesState) -> Literal["summarize_conversation", END]:
        """Return the next node to execute."""
        messages = state["messages"]
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > self.n_messages:
            return "summarize_conversation"
        
        # Otherwise we can just end
        return END


    def exists_tool(self, state: MessagesState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    

    def apply_tools(self, state: MessagesState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tool_names:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tool_names[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


    def invoke(self, message, thread_id=None):
        messages = [HumanMessage(content=message)]
        result = self.graph.invoke({"messages": messages},
                                    {"configurable": {"thread_id": thread_id}})
        answer = result['messages'][-1].content
        return answer, result
    
agent = Agent()

if __name__ == "__main__":
    abot = Agent()
    thread_id = "1"  # This can be used to maintain context across multiple interactions
    user_message = input("Human: ")

    while user_message.lower() != "exit":
        agent_response, agent_state = abot.invoke(user_message, thread_id=thread_id)
        print(f"AI: {agent_response}")
        user_message = input("Human: ")
        print(agent_state)