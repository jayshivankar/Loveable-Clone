from langchain_core.prompts.string import PromptTemplateFormat
from langchain_openai import ChatOpenAI
from langgraph.graph import START,StateGraph,END
from langchain_core.prompts import ChatPromptTemplate
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

    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are the ARCHITECT agent. Given this project plan below by the user, break it down into explicit engineering tasks.

                RULES:
                - For each FILE in the plan, create one or more IMPLEMENTATION TASKS.
                - In each task description:
                    * Specify exactly what to implement.
                    * Name the variables, functions, classes, and components to be defined.
                    * Mention how this task depends on or will be used by previous tasks.
                    * Include integration details: imports, expected function signatures, data flow.
                - Order tasks so that dependencies are implemented first.
                - Each step must be SELF-CONTAINED but also carry FORWARD the relevant context from earlier tasks.

                """
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





graph = StateGraph(dict)
graph.add_node('planner_agent',planner_agent)
graph.add_node('architect_agent',architect_agent)

graph.add_edge(START,'planner_agent')
graph.add_edge('planner_agent','architect_agent')
graph.add_edge('architect_agent',END)

workflow = graph.compile()
result = workflow.invoke({'user_input': 'Build me a good multimodalRAG system'})
print(result)