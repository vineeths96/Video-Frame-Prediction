import os
import json
import math
import logging


class Config:
    """
    Config Parser class.
    """

    def __init__(self, file_path_or_dict, logger_name="global"):
        super(Config, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.cfg_init = self.load_config(file_path_or_dict)
        self.check_meta()

    def load_config(self, file_path_or_dict):
        if type(file_path_or_dict) is str:
            assert os.path.exists(file_path_or_dict), '"{}" not exists'.format(file_path_or_dict)
            config = dict(json.load(open(file_path_or_dict)))
        elif type(file_path_or_dict) is dict:
            config = file_path_or_dict
        else:
            raise Exception("The input must be a string path or a dict")

        return config

    def check_meta(self):
        if "meta" not in self.cfg_init:
            self.logger.warning("The cfg does not include meta tag, will generate default meta tag")
            self.logger.warning("Used the default meta configs.")

            self.cfg_init["meta"] = (
                {
                    "board_path": "board",
                },
            )
        else:
            cfg_meta = self.cfg_init["meta"]

            if "board_path" not in cfg_meta:
                self.logger.warning("Not specified board_path, used default. (board)")
                self.cfg_init["meta"]["board_path"] = "board"

        self.__dict__.update(self.cfg_init)

    def log_dict(self):
        self.logger.debug("Used config: \n {}".format(self.cfg_init))


class Meter:
    """
    Value tracker class.
    """

    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(name=self.name, val=self.val, avg=self.avg)

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter:
    """
    Average value tracker class.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)

        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
            self.sum[k] += kwargs[k]
            self.count[k] += batch

    def __repr__(self):
        s = ""
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
            name=attr, val=float(self.val[attr]), avg=float(self.sum[attr]) / self.count[attr]
        )

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


logs = set()


def get_format():
    """
    Logging formatter.
    """

    format_str = "[%(asctime)s-%(filename)s#%(lineno)3d] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_str)

    return formatter


def get_format_custom():
    """
    Custom logging formatter.
    """

    format_str = "[%(asctime)s-%(message)s"
    formatter = logging.Formatter(format_str)

    return formatter


def init_log(name, level=logging.INFO, format_func=get_format):
    """
    Initiate logger.
    """

    if (name, level) in logs:
        return logging.getLogger(name)

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = format_func()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def add_file_handler(name, log_file, level=logging.DEBUG):
    """
    Add file handling to logger.
    """

    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file, "w+")
    fh.setFormatter(get_format())
    fh.setLevel(level)
    logger.addHandler(fh)


def print_speed(i, i_time, n, logger_name="global"):
    """
    Print speed of training.
    """

    logger = logging.getLogger(logger_name)
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info(
        "Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n"
        % (i, n, i / n * 100, average_time, remaining_day, remaining_hour, remaining_min)
    )
