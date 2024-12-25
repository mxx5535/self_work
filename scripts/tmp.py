# -*- coding: utf-8 -*-
# @Time : 2024/12/23 12:43
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : tmp.py
# @Project : education_chatbot
import os

from autogen import ConversableAgent

from typing import Annotated, TypedDict


class ClassInformation(TypedDict):
    lession_name: str
    date_begin_time: str
    date_end_time: str
    learn_start_time: str
    learn_end_time: str
    teacher_name: str


ClassInfomationType = Annotated[
    ClassInformation, "A dictionary representing a lession with lession_name, date_begin_time, date_end_time, learn_start_time, learn_end_time and teacher_name"]


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


class GetCurrentTime(TypedDict):
    current_time: str


GetCurrentTimeType = Annotated[GetCurrentTime, "A string representing current_time"]


def get_current_time() -> GetCurrentTimeType:
    import datetime

    # 获取当前时间
    current_time = datetime.datetime.now()

    return str(current_time)[:19]


class StandardAnswer(TypedDict):
    standard_answer: str


StandardAnswerType = Annotated[
    StandardAnswer, "A string data representing the extraced result from the llms final-round result. the question like judge yes or no in llms result"]


def standrad_answer(llms_result: Annotated[str, "the result of llms return"]) -> StandardAnswerType:
    if "是" in llms_result:
        return 'True' + 'TERMINATE'
    else:
        return 'False' + 'TERMINATE'


llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "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"}]}

# 创建助手代理
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
                   "you can help me answer some questions"
                   "Remember returning 'TERMINATE' when the task is done. Just like this one, The result is ... TERMINATE",
    llm_config=llm_config
)

# 创建用户代理
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# 注册计算器函数到助手代理
assistant.register_for_llm(name="get_current_time", description="a tool to get current time")(get_current_time)

# 注册计算器函数到助手代理
assistant.register_for_llm(name="get_lession_information",
                           description="a tool to get the information of the lession given by lession name")(
    get_lession_information)

assistant.register_for_llm(name="standrad_answer",
                           description="extract the standard answer from the llms final-round result")(standrad_answer)

# 注册计算器函数到用户代理
user_proxy.register_for_execution(name="get_current_time")(get_current_time)

# 注册计算器函数到用户代理
user_proxy.register_for_execution(name="get_lession_information")(get_lession_information)

user_proxy.register_for_execution(name="standrad_answer")(standrad_answer)

# chat_result = user_proxy.initiate_chat(assistant, message="请你判断今天晚上八点讲真这门课是否上课？如果今天上课，就返回“是”，如果今天不上课，请返回“不是”, 这个结果要调用standrad_answer函数去返回True或者False")










