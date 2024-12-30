# -*- coding: utf-8 -*-
# @Time : 2024/12/28 19:16
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot
from langchain_core.prompts.prompt import PromptTemplate

class BasePrompt:
    def __init__(self, prompt: PromptTemplate = None):
        self.prompt = prompt

    def get_prompt(self, **kwargs):
        return self._get_prompt(**kwargs)

    def _get_prompt(self, **kwargs):
        raise NotImplementedError

