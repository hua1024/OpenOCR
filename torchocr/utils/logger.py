import logging
import os
import torch.distributed as dist

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


log_info = dict(
    # 日志保留路径,默认保留在项目跟目录下的logs文件
    log_dir='',
    # 日志级别,默认INFO级别,ERROR,WARNING,WARN,INFO,DEBUG
    log_level='DEBUG',
    # 每天生成几个日志文件,默认每天生成1个
    log_interval=1,
    # #日志保留多少天,默认保留7天的日志
    log_backupCount=7,
)


class Logging():
    __instance=None
    # ERROR,WARNING,WARN,INFO,DEBUG
    __logger_level_dic={
        'ERROR':logging.ERROR,
        'WARNING':logging.WARNING,
        'WARN':logging.WARN,
        'INFO':logging.INFO,
        'DEBUG':logging.DEBUG,
    }

    def __init__(self):
        self.__logger = logging.getLogger()
        # 日志文件名
        self.__filename = 'train.log'
        __log_info = log_info
        self.__log_dir = __log_info['log_dir']
        self.__log_level = __log_info['log_level']
        log_backupCount = __log_info['log_backupCount']
        log_interval = __log_info['log_interval']

        if log_backupCount is None or not isinstance(log_backupCount,int):
            log_backupCount = 7
        elif log_backupCount<0:
            log_backupCount = 7

        if log_interval is None or not isinstance(log_interval,int):
            log_interval = 1
        elif log_interval<0:
            log_interval = 7

        if self.__log_level == None or self.__log_level =='':
            self.__level = logging.INFO
        else:
            if self.__log_level.upper() not in self.__logger_level_dic.keys():
                self.__level = logging.INFO
            else:
                self.__level = self.__logger_level_dic[self.__log_level.upper()]


        if self.__log_dir == None or self.__log_dir =='':
            project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            self.__log_dir = os.path.join(project_root_path,'logs')
        if not os.path.exists(self.__log_dir):
            os.makedirs(self.__log_dir)
        # 创建一个handler，用于写入日志文件 (每天生成1个，保留10天的日志)
        fh = logging.handlers.TimedRotatingFileHandler(os.path.join(self.__log_dir,self.__filename), 'D', log_interval,log_backupCount)
        fh.suffix = "%Y%m%d-%H%M.log"
        fh.setLevel(self.__level)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(self.__level)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.__logger.setLevel(self.__level)
        # 给logger添加handler
        self.__logger.addHandler(fh)
        self.__logger.addHandler(ch)

    @classmethod
    def getLogger(cls)-> logging.Logger:
        if not cls.__instance:
            cls.__instance = Logging()
        return cls.__instance.__logger

if __name__ == '__main__':
    logger = Logging.getLogger()
    for i in range(100):
        logger.debug('aa')
        logger.error('bb')
        logger.info('cc')
        # time.sleep(1)
