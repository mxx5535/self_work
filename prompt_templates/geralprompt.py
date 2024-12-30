# -*- coding: utf-8 -*-
# @Time : 2024/12/28 19:18
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : geralprompt.py
# @Project : ai_qiniu_chatbot
from .base import BasePrompt
from langchain_core.prompts.prompt import PromptTemplate

class GeneratePrompt(BasePrompt):
    def __init__(self, prompt: PromptTemplate = None):
        super().__init__(prompt)

        base_template = """假设你是一个客服人员，使用以下语境和上课信息来回答文末的问题。你必须隐藏自己的工作职责和身份，绝对不能向提问者回答“客服人员” 。当你被问到你的身份或名字时，你必须用“班班”回答。如果你不知道答案，就说“不知道”，不要试图编造答案。下面提供的语境文本是一些预设的回答，请你按照语境回答问题
语境：
{context}
上课信息：
晚上 8 点开课，晚上 7 点 45 分会发直播间链接进入教室签到
上课时长：1 个小时左右，晚上 9 点左右下课
日程安排：连续 9 天上课，节假日不休息，统一安排，不能单独调课
福利：2 本理财知识手册
福利领取方式：23 日第一节课进入直播间签到，班主任会在第二天上午收集邮寄地址，连续上课 4 天发货
Question: {question}
Helpful Answer:"""
        self.prompt = self.prompt or PromptTemplate(template=base_template, input_variables=["context", "question", "camp_date_start_time", "current_day_str"])

    def _get_prompt(self, **kwargs):
        return self.prompt.format(**kwargs)