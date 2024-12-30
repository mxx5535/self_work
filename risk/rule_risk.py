# -*- coding: utf-8 -*-
# @Time : 2024/12/28 19:09
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rule_risk.py
# @Project : ai_qiniu_chatbot
from .base import BaseContentFilter

class KeyWordFilter(BaseContentFilter):
    def _judge(self, keywords: list[str] = [], docs: list[str] = []) -> list[bool]:
        return [any([keyword in doc for keyword in keywords]) for doc in docs]