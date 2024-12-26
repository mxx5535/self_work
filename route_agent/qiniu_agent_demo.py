import asyncio
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
# from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

def search_express_tool(query:str):
    '''
    查询快递tools，业务逻辑省略，模拟给出结果
    :param query:
    :return:
    '''

    return {
        "corp_name": "顺丰",
        "expose_status": 1,
        "source_city": "北京",
        "derminate_city": "上海",
        "current_city": "郑州"
    }

def search_course_tool(query:str):
    '''
    查询课程信息tool，业务代码查询省略，直接返回结果
    :param query:
    :return:
    '''
    if '讲真' in query:
        return {
            "lession_name": query,
            "date_begin_time": '2024-12-01',
            "date_end_time": '2024-12-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Tonny'
        }
    elif '千尺' in query:
        return {
            "lession_name": query,
            "date_begin_time": '2024-11-01',
            "date_end_time": '2024-11-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Tom'
        }
    else:
        return {
            "lession_name": query,
            "date_begin_time": '2024-10-01',
            "date_end_time": '2024-10-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Sun'
        }

def rag_tool(query:str):
    '''
    其他问题通过rag调用回答，具体实现逻辑省略
    :param query:
    :return:
    '''
    return "好的"

# 定义大模型
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="test-az-eus-gpt-4o",
    model="gpt-4o",
    api_version="2023-05-15",
    azure_endpoint="https://test-az-eus-ai-openai01.openai.azure.com/",
    # azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    api_key="02855675d52d4abfa48868c00c6f2773", # For key-based authentication.
)


# 注册agent
planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your task is to accurately select the right agent to answer the user's question.
    Your team members are:
        CourseInformationAgent: Query questions related to course start information
        ExpressInformationAgent: Query express related questions
        OtherQuestinoAgent：A backstop agent that solves problems unrelated to other agents
    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

# 注册第二个agent
course_information_agent = AssistantAgent(
    "CourseInformationAgent",
    description="An agent who knows when the course starts.",
    tools=[search_course_tool],
    model_client=model_client,
    system_message="""
    You are a course information agent.
    Your only tool is search_course_tool - use it to find information.
    When the user asks for information about the start time, etc., you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)


# 注册第三个agent
express_information_agent = AssistantAgent(
    "ExpressInformationAgent",
    description="An agent who knows Express process related information",
    tools=[search_express_tool],
    model_client=model_client,
    system_message="""
    You are a express information agent.
    Your only tool is search_express_tool - use it to find information.
    When the user asks for information about the Express progress, you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

# 注册第4个agent
other_rag_agent = AssistantAgent(
    "OtherQuestinoAgent",
    description="An agent who answers questions that have nothing to do with deliveries and courses",
    tools=[rag_tool],
    model_client=model_client,
    system_message="""
    You are a express information agent.
    Your only tool is search_express_tool - use it to find information.
    When users ask questions unrelated to delivery and class times, you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, course_information_agent, express_information_agent,other_rag_agent],
    model_client=model_client,
    termination_condition=termination,
)

# task = "你好"
# task = "快递现在到哪里了"
task = '千尺这门课几点开课啊'
# Use asyncio.run(...) if you are running this in a script.
asyncio.run(Console(team.run_stream(task=task)))


