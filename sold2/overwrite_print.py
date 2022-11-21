#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: overwrite_print.py
@time: 2022/11/16 上午8:52
@desc:  https://www.cnblogs.com/saxum/p/15787880.html
"""

##################################################################################
# 以下内容放在所有代码之前,实现print自动打印到日志
import os, sys, time, io
import builtins as __builtin__


def print(*args, **kwargs):
    # __builtin__.print('New print function')
    return __builtin__.print(time.strftime("%Y-%m-%d %H:%M:%S -----  ", time.localtime()), *args, **kwargs)


class Logger(object):
    def __init__(self, filepath="./Default.log"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf8')
        self.prev_time = time.time()
        self.flush_sec = 5

    def write(self, message):
        if time.time() - self.prev_time > self.flush_sec:
            self.flush()

        self.terminal.write(message)
        self.log.write(message)
        self.prev_time = time.time()

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


def set_log(log_filepath='./train.log'):
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    sys.stdout = Logger(log_filepath)
##################################################################################
