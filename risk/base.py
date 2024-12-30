# -*- coding: utf-8 -*-
# @Time : 2024/12/28 19:07
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot
class BaseContentFilter:
    def judge(self, keywords: list[str] = [], docs: list[str] = []) -> list[bool]:
        return self._judge(keywords, docs)

    def _judge(self, keywords: list[str] = [], docs: list[str] = []) -> list[bool]:
        raise NotImplementedError

