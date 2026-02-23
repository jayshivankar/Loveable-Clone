from langchain_openai import ChatOpenAI
from langgraph.graph import START,StateGraph,END
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from Agents.Prompts import *
from Agents.Structured_output import *


load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')


def planner_agent(states:dict) -> dict:
    user_input = states['user_input']
    try:
        if user_input:
            planner_prompts = planner_prompt(user_input)
            result = llm.with_structured_output(Plan).invoke(planner_prompts)

            return {
            'name' : result.name,
            'description' : result.description,
            'techstack': result.techstack,
            'features' : result.features,
            'files' : result.files
            }
    except Exception as e:
        return f"Error :  {e}"









graph = StateGraph(dict)
graph.add_node('planner_agent',planner_agent)
graph.add_edge(START,'planner_agent')
graph.add_edge('planner_agent',END)

workflow = graph.compile()
result = workflow.invoke({'user_input': 'Build me a good app using langgraph and langchain'})
print(result)