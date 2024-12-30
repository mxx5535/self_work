# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:14
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : express.py
# @Project : ai_qiniu_chatbot

from .base import BaseAgent
from autogen_agentchat.agents import AssistantAgent
from agents.tools.express_tools import search_express_tool
from agents.tools.common_tools import get_current_time

class ExpressAgent(BaseAgent):
    def __init__(self, model_client):
        super().__init__(model_client)
        self.tools = [search_express_tool, get_current_time]

    def _get_agent(self):
        express_information_agent = AssistantAgent(
            "ExpressInformationAgent",
            description="An agent who knows Express process related information",
            tools=self.tools,
            handoffs=['SummaryAgent'],
            model_client=self.model_client,
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
        return express_information_agent
