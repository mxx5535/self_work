# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:20
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot

class BaseAgent():
    def __init__(self, model_client):
        self.model_client = model_client

    def get_agent(self):
        return self._get_agent()

    def _get_agent(self):
        raise NotImplementedError

