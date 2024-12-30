# -*- coding: utf-8 -*-
# @Time : 2024/12/29 16:51
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : express.py
# @Project : ai_qiniu_chatbot
from typing import Annotated, TypedDict

class ExpressInformation(TypedDict):
    corp_name: str
    expose_status: int
    source_city: str
    derminate_city: str
    current_city: str

ExpressInformationType = Annotated[ExpressInformation, "A dictionary representing current express status with corp_name, expose_status, source_city, derminate_city and current_city "]

def search_express_tool() -> ExpressInformationType:
    """Get the information of current express information"""

    return {
        "corp_name": "顺丰",
        "expose_status": 1,
        "source_city": "北京",
        "derminate_city": "上海",
        "current_city": "郑州"
    }