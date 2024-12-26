# -*- coding: utf-8 -*-
# @Time : 2024/12/25 18:40
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_group.py
# @Project : self_work
import os

os.environ['OPENAI_API_VERSION'] = "2023-05-15"
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from rag_main import MyRetrieveUserProxyAgent
from rag_main import openai_ef, vector_db
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

class ExposeInformation(TypedDict):
    corp_name: str
    expose_status: int
    source_city: str
    derminate_city: str
    current_city: str

ExposeInformationType = Annotated[ExposeInformation,
                                  "A dictionary representing a exposeinformation with corp_name, expose_status, source_city, derminate_city, current_city"]

def get_express_information() -> ExposeInformationType:
    """ Get the information of the expose"""
    return {
        "corp_name": "顺丰",
        "expose_status": 1,
        "source_city": "北京",
        "derminate_city": "上海",
        "current_city": "郑州"
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

llm_config = {
    "config_list": [
        {
            "api_type": "azure",
            "model": "test-az-eus-gpt-4o",
            "api_key": "02855675d52d4abfa48868c00c6f2773",
            "base_url": "https://test-az-eus-ai-openai01.openai.azure.com/"
        }
    ]
}

# 创建课程助手代理
class_assistant = AssistantAgent(
    name="class_information_assistant",
    system_message="You are a helpful AI assistant. "
                   "you can help me answer some questions about class information"
                   "Remember returning 'TERMINATE' when the task is done. Just like this one, The result is ... TERMINATE",
    llm_config=llm_config
)

# 创建快递助手代理
express_assistant = AssistantAgent(
    name="express_information_assistant",
    system_message="You are a helpful AI assistant. "
                    "you can help me answer some questions about express delivery information"
                   "Remember returning 'TERMINATE' when the task is done. Just like this one, The result is ... TERMINATE",
    llm_config=llm_config
)

# 创建rag助手
ragproxyagent = MyRetrieveUserProxyAgent(
    name="ragproxyagent",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",  # "code", "qa" 和 "default"
        "model": llm_config['config_list'][0]["model"],
        "vector_db": vector_db,
        "collection_name": "qiniu_db_collection",
        "embedding_function": openai_ef,  # 确保此处使用正确的嵌入模型
        "get_or_create": False,  # 设置为False，如果不希望重复使用已有的集合
    },
    code_execution_config=False,  # 设置为False如果不希望执行代码
    human_input_mode="NEVER"
)

# 创建用户代理
user_proxy = UserProxyAgent(
    name="User",
    llm_config=llm_config,
    system_message = "A human user capable of interacting with AI agents. Remember returning 'TERMINATE' when the task is done. Just like this one, The result is ... TERMINATE",
    code_execution_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    # human_input_mode="ALWAYS"
)

# 注册计算器函数到助手代理
class_assistant.register_for_llm(name="get_current_time", description="a tool to get current time")(get_current_time)

# 注册计算器函数到助手代理
class_assistant.register_for_llm(name="get_lession_information", description="a tool to get the information of the lession given by lession name")(get_lession_information)

class_assistant.register_for_llm(name="standrad_answer", description="extract the standard answer from the llms final-round result")(standrad_answer)

express_assistant.register_for_llm(name = "get_express_information", description="a tool to get the information of the express delivery")(get_express_information)

# 注册计算器函数到用户代理
ragproxyagent.register_for_execution(name="get_current_time")(get_current_time)

# 注册计算器函数到用户代理
ragproxyagent.register_for_execution(name="get_lession_information")(get_lession_information)

ragproxyagent.register_for_execution(name="standrad_answer")(standrad_answer)

ragproxyagent.register_for_execution(name = "get_express_information")(get_express_information)

# group_chat = GroupChat(agents=[ragproxyagent, class_assistant, express_assistant], messages=[], max_round=120, speaker_selection_method="auto")
group_chat = GroupChat(agents=[class_assistant, express_assistant, ragproxyagent], messages=[], max_round=120, speaker_selection_method="auto")

group_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
)

# user_proxy.initiate_chat(group_manager, message="请你告诉我讲真这门课的上课时间和下课时间")

# ragproxyagent.initiate_chat(group_manager, message="请你告诉我快递信息")

# user_proxy.initiate_chat(group_manager, message="有回放吗？")
ragproxyagent.initiate_chat(group_manager, message=ragproxyagent.message_generator, problem="有回放吗？",  n_results = 3)
# user_proxy.initiate_chat(group_manager, message="请你告诉我讲真这门课的上课时间和下课时间")

# user_proxy.initiate_chat(group_manager, message="有回放吗？")
