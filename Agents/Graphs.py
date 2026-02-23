from langchain_core.prompts.string import PromptTemplateFormat
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


def architect_agent(states:dict):

    prompts = architect_prompt()
    architect_prompts = prompts.format()

    human_prompt = f"""
    These below are all the attributes of the Plan
    name = {states['name']}
    description = {states['description']}
    techstack = {states['techstack']}
    features = {states['features']}
    files = {states['files']}
    """

    answer = llm.with_structured_output(TaskPlan).invoke(
        SystemMessage(content=architect_prompts),
        HumanMessage(content = human_prompt)
    )
    return {'implementation_steps':answer.implementation_steps}




graph = StateGraph(dict)
graph.add_node('planner_agent',planner_agent)
graph.add_node('architect_agent',architect_agent)

graph.add_edge(START,'planner_agent')
graph.add_edge('planner_agent','architect_agent')
graph.add_edge('architect_agent',END)

workflow = graph.compile()
result = workflow.invoke({'user_input': 'Build me a good webapp with datastore'})
print(result)