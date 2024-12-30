# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:35
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot
import json
import requests


class BaseChat():
    def __int__(self):
        pass

    def chat(self, prompt: str) -> str:
        return self._chat(prompt)

    # todo:prompt的类型
    def _chat(self, prompt:dict) -> json:
        raise NotImplementedError
