# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:35
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : summary.py
# @Project : ai_qiniu_chatbot
from .base import BaseAgent
from autogen_agentchat.agents import AssistantAgent

class SummaryAgent(BaseAgent):
    def __init__(self, model_client):
        super().__init__(model_client)

    def _get_agent(self):
        summary_agent = AssistantAgent(
            "SummaryAgent",
            description="An agent who Responsible for summarizing the previous agent content and answering user questions",
            model_client=self.model_client,
            system_message="""
            You are a summary information agent.
            You are responsible for summarizing the previous agent information and answering user questions
            You only execute it once.    
            When the summary is complete, 'TERMINATE'  must be output.
            用中文回答
            """,
        )
        return summary_agent
