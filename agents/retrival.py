# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:27
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : retrival.py
# @Project : ai_qiniu_chatbot
from .base import BaseAgent
from autogen_agentchat.agents import AssistantAgent
from agents.tools.common_tools import get_current_time
from agents.tools.rag_tools import rag_tool

class RAGAgent(BaseAgent):
    def __init__(self, model_client):
        super().__init__(model_client)
        self.tools = [get_current_time, rag_tool]

    def _get_agent(self):
        rag_agent = AssistantAgent(
            "OtherQuestinoAgent",
            description="An agent who answers questions that have nothing to do with deliveries and courses",
            tools=self.tools,
            handoffs=['SummaryAgent'],
            model_client=self.model_client,
            system_message="""
            You are a express information agent.
            Your only tool is search_express_tool - use it to find information.
            When users ask questions unrelated to delivery and class times, you need to call the tool to get the information
            You make only one search call at a time.
            Once you have the results, you never do calculations based on them.
            """,
        )
        return rag_agent
