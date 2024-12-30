# -*- coding: utf-8 -*-
# @Time : 2024/12/30 17:49
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_tool_rewrite_rerank.py
# @Project : self_work
import asyncio
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import SourceMatchTermination, TextMentionTermination, MaxMessageTermination
# from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from chats.azure_chat import AzureChatOpenAI
from embeddings.embeddings import BGEEmbedding
from vectorstores.vectorstores import MyFaissVectorstore
from prompt_templates.geralprompt import GeneratePrompt
from risk.rule_risk import KeyWordFilter
from chains.rag_chain import RAGChain

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

def rag_tool(query: str):
    '''
    其他问题通过rag调用回答，具体实现逻辑省略
    :param query:
    :return:
    '''
    print("query:", query)
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': query,
        'top_k': 10,
        'filter': None,
        'prompt_kwargs': {
            'question': query,  # query
        },
        'sensitive_words': ['骗钱']
    }

    # 执行
    chat = AzureChatOpenAI()
    faiss_vectorstore = MyFaissVectorstore()
    bge_embedding = BGEEmbedding()
    generate_prompt = GeneratePrompt()
    content_filter = KeyWordFilter()
    rag_chain = RAGChain(faiss_vectorstore, bge_embedding, chat, content_filter, generate_prompt)

    result = rag_chain.invoke(**kwargs)

    print("result:", result)

    return result + " TERMINATE"

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
    handoffs=['StartInformationAgent', 'ExpressInformationAgent', 'OtherQuestinoAgent'],
    system_message="""
    You are a planning agent.
    Your team members are:
        StartInformationAgent: 只负责回答课程开始时间相关信息，其他课程相关信息不负责回答
        ExpressInformationAgent: Query express related questions
        OtherQuestinoAgent：What StartInformationAgent and ExpressInformationAgent can't solve needs to be solved
    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>
    """,
)

# 注册第二个agent
start_time_information_agent = AssistantAgent(
    "StartInformationAgent",
    description="An agent who knows when the course starts.but Course start time information only, no other course information",
    tools=[search_course_tool],
    handoffs=['SummaryAgent'],
    model_client=model_client,
    system_message="""
    You are a course information agent.
    Your only tool is search_course_tool - use it to find information.
    When the user asks for information about the start time, etc., you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, You should give your results to the next agent.
    Your team members are:
        SummaryAgent: Summarize the information passed and answer user questions
    When assigning tasks, use this format:
    1. <agent> : <task>
    """,
)


# 注册第三个agent
express_information_agent = AssistantAgent(
    "ExpressInformationAgent",
    description="An agent who knows Express process related information",
    tools=[search_express_tool],
    handoffs=['SummaryAgent'],
    model_client=model_client,
    system_message="""
    You are a express information agent.
    Your only tool is search_express_tool - use it to find information.
    When the user asks for information about the Express progress, you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, You should give your results to the next agent.
    Your team members are:
        SummaryAgent: Summarize the information passed and answer user questions
    When assigning tasks, use this format:
    1. <agent> : <task>
    """,
)

# 注册第4个agent
other_rag_agent = AssistantAgent(
    "OtherQuestinoAgent",
    description="An agent who answers questions that have nothing to do with deliveries and courses",
    tools=[rag_tool],
    handoffs=['SummaryAgent'],
    model_client=model_client,
    system_message="""
    You are a express information agent.
    Your only tool is search_express_tool - use it to find information.
    When users ask questions unrelated to delivery and class times, you need to call the tool to get the information
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

# 注册第五个agent 执行agent
summary_agent = AssistantAgent(
    "SummaryAgent",
    description="An agent who Responsible for summarizing the previous agent content and answering user questions",
    model_client=model_client,
    system_message="""
    You are a summary information agent.
    You are responsible for summarizing the previous agent information and answering user questions
    You only execute it once.    
    When the summary is complete, 'TERMINATE'  must be output.
    用中文回答
    """,
)


text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=15)
# max_messages_termination = SourceMatchTermination(['StartInformationAgent', 'ExpressInformationAgent', 'OtherQuestinoAgent'])
termination = text_mention_termination | max_messages_termination

team = Swarm([planning_agent, start_time_information_agent, express_information_agent, other_rag_agent, summary_agent], termination_condition=termination)

task = "书免费吗"
# task = "快递现在到哪里了"
# task = '千尺这门课几点开课啊'
# task = '千尺几点开课'
# Use asyncio.run(...) if you are running this in a script.
asyncio.run(Console(team.run_stream(task=task)))


