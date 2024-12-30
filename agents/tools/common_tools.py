# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:48
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : common_tools.py
# @Project : ai_qiniu_chatbot
from typing import Annotated, TypedDict

class GetCurrentTime(TypedDict):
    current_time: str

GetCurrentTimeType = Annotated[GetCurrentTime, "A string representing current_time"]

def get_current_time() -> GetCurrentTimeType:
    import datetime
    # 获取当前时间
    current_time = datetime.datetime.now()
    return str(current_time)[:19]

