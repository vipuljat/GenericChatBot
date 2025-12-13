import logging
import time
import requests


import config as conf
from logging.config import dictConfig
from constants import logger_name,log_format,log_console_class


def format_date_time():
    gmt_time = list(time.gmtime())
    date_format = "-".join(str(e) for e in gmt_time[:3])
    time_format = ":".join(str(e) for e in gmt_time[3:6])
    return f"{date_format} {time_format}"


class EventTypeFilter(logging.Filter):
    def __init__(self, event_type):
        super().__init__()
        self.event_type = event_type

    def filter(self, record):
        return getattr(record, 'event_type', None) == self.event_type


class DbLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
     
    def emit(self, record):
        log_entry = self.format(record)
        response = requests.post(url=conf.LOG_INGESTION_URL,data=log_entry)
        print(f"log response {response}")
        
      
    def close(self):
        logging.Handler.close(self)


class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):

        extra = self.extra.copy()
        
        kwargs['extra'] = extra
        
        return msg, kwargs  

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        'access': {
            '()': 'uvicorn.logging.AccessFormatter',
            "fmt": "%(levelprefix)s " + f"{format_date_time()} |" + log_format,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True
        },
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s " + f"{format_date_time()} | " + log_format,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True
        },
        "console": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelname)s " + f"{format_date_time()} | " + log_format,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(levelname)s " + f"{format_date_time()} | " + log_format,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "specific_event_filter": {
            "()": EventTypeFilter,
            "event_type": "db_event"
        }
    },
    "handlers": {
        "console": {
            "class": log_console_class,
            "formatter": "json",
            "stream": "ext://sys.stdout",
            "level": "DEBUG"
        },
        "access": {
            "class": log_console_class,
            "formatter": "json",
            "stream": "ext://sys.stdout"
        },
        "default": {
            "formatter": "json",
            "class": log_console_class,
            "stream": "ext://sys.stderr",
        },
        "db_log_handler":{
            "()": DbLogHandler,
            "formatter": "json",
            "filters": ["specific_event_filter"],
            "level": "DEBUG"
        }
        
    },
    "loggers": {
        logger_name: {
            "handlers": ["console","db_log_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": True
        },
        'uvicorn.access': {
            'handlers': ['access'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.error': {
            'level': 'INFO',
            'propagate': False
        },
        'passlib': {
            'level': 'WARNING',
            'propagate': False
        }
    }
}

dictConfig(logging_config)

base_logger = logging.getLogger(logger_name)
    
log = ContextLoggerAdapter(base_logger,extra={})
