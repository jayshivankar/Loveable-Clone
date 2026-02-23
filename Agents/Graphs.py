from langchain_core.prompts.string import PromptTemplateFormat
from langchain_openai import ChatOpenAI
from langgraph.graph import START,StateGraph,END
from langchain_core.prompts import PromptTemplate
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
    sys_msg = SystemMessage(prompts.format())
    prompt_human_msg = PromptTemplate.from_template("""
         1. name of project = {name},
         2. description of project= {description},
         3. techstack used in project = {techstack},
         4. features used in project = {features},
         5. files used in project = {files}
    """ )
    human_msg = prompt_human_msg.format(name=states['name'],description=states['description'],techstack=states['techstack'],features=states['features'],files=states['files'])
    answer = llm.with_structured_output(TaskPlan).invoke([sys_msg,HumanMessage(human_msg)])
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