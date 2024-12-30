# -*- coding: utf-8 -*-
# @Time : 2024/12/28 13:08
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : test_vectorstores.py
# @Project : ai_qiniu_chatbot
from loguru import logger
from nose.plugins.attrib import attr
from .vectorstores import MyFaissVectorstore

@attr(select=1552)
def test_matcher():
    logger.debug('run test_matcher')
    mather = MyFaissVectorstore()

