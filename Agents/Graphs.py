from langchain_core.prompts.string import PromptTemplateFormat
from langchain_openai import ChatOpenAI
from langgraph.graph import START,StateGraph,END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
from Agents.Prompts import *
from Agents.Structured_output import *
from Agents.tools import *
from langgraph.prebuilt import ToolNode, tools_condition


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
    prompt = architect_prompt()

    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt
                ,
            ),
            ("human",
             """
             1. name of project = {name},
             2. description of project= {description},
             3. techstack used in project = {techstack},
             4. features used in project = {features},
             5. files used in project = {files}
                 """
             ),
        ]
    )

    name = states['name'],
    description = states['description'],
    techstack = states['techstack'],
    features = states['features'],
    files = states['files']

    evaluation_llm = doc_eval_prompt | llm.with_structured_output(TaskPlan)

    answer = evaluation_llm.invoke({'name':name,'description':description,'techstack':techstack,'features':features,'files':files})
    return {'implementation_steps':answer.implementation_steps}


tools = [write_file, read_file, list_files, get_current_directory,run_cmd]

def coder_agent(states:dict):
    prompt = coder_system_prompt()
    implementation_steps = states['implementation_steps']
    learner_steps = states['learner_steps']

    coder_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt
                ,
            ),
            ("human",
             f"""
             This list contains filepath of file to be modified and tasks to be implemented in the particular file.
             It has in total {learner_steps} files to be modified .GO through each of them one by one and write the 
             code per the tasks for the particular file. Stop the execution whenever all the steps are done.
             
             Steps :
             {implementation_steps}
              
             """
             ),
        ]
    )

    llm_with_tools = llm.bind_tools(tools)

    coder_chain = coder_prompt | llm_with_tools

    output = coder_chain.invoke({'learner_steps':learner_steps,'implementation_steps':implementation_steps})
    if output:
        return {'all done'}



tool_node = ToolNode(tools)


graph = StateGraph(dict)
graph.add_node('planner_agent',planner_agent)
graph.add_node('architect_agent',architect_agent)
graph.add_node('coder_agent',coder_agent)
graph.add_node("tools", tool_node)

graph.add_edge(START,'planner_agent')
graph.add_edge('planner_agent','architect_agent')
graph.add_conditional_edges('architect_agent',tools_condition)
graph.add_edge('tools','architect_agent')
graph.add_edge('architect_agent',END)

workflow = graph.compile()
result = workflow.invoke({'user_input': 'build a basic web application'})
print(result)