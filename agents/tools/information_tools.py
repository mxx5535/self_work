# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:50
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : information_tools.py
# @Project : ai_qiniu_chatbot
from typing import Annotated, TypedDict

class ClassInformation(TypedDict):
    lession_name: str
    date_begin_time: str
    date_end_time: str
    learn_start_time: str
    learn_end_time: str
    teacher_name: str

ClassInfomationType = Annotated[ClassInformation, "A dictionary representing a lession with lession_name, date_begin_time, date_end_time, learn_start_time, learn_end_time and teacher_name"]

def get_lession_information(lession_name: Annotated[str, "the name of lession"]) -> ClassInfomationType:
    """Get the information of the lession given by lession_name"""
    if lession_name == '讲真':
        return {
            "lession_name": lession_name,
            "date_begin_time": '2024-12-01',
            "date_end_time": '2024-12-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Tonny'
        }
    elif lession_name == '千尺':
        return {
            "lession_name": lession_name,
            "date_begin_time": '2024-11-01',
            "date_end_time": '2024-11-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Tom'
        }
    else:
        return {
            "lession_name": lession_name,
            "date_begin_time": '2024-10-01',
            "date_end_time": '2024-10-15',
            "learn_start_time": '19: 00',
            "learn_end_time": '21: 00',
            "teacher_name": 'Sun'
        }