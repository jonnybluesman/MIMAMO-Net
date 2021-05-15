import os
import random
import logging
import numpy as np
from pathlib import Path


def set_random_state(random_seed):
    # os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(random_seed)
    np.random.seed(random_seed)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_files_with_extension(dir_path, extensions):    
    assert all([e[0] == "." for e in extensions]), \
        "All the extension names should start with a dot."
    all_files = []  # a list of files matching the pattern
    for ext in extensions:
        all_files.extend(Path(dir_path).glob(f"*{ext}"))
    return all_files

def append_all(a_tuple, *lists):
    assert len(a_tuple) == len(lists)
    for list_entry, tuple_entry in zip(lists, a_tuple):
        list_entry.append(tuple_entry)


def append_to_df(x, y):
    x.loc[len(x)+1] = y


def is_file(parser, f_arg):
    if not os.path.exists(f_arg):
        return parser.error("File %s does not exist!" % f_arg)
    return f_arg


def is_dir(parser, f_arg):
    if not os.path.isdir(f_arg):
        return parser.error("Directory %s does not exist!" % f_arg)
    return f_arg


def get_class_name(object):
    """Returns the class name of a given object."""
    return object.__class__.__name__


def set_logger(log_name, log_console=True, log_dir=None):
    
    logger_master = logging.getLogger(log_name)
    logger_master.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger_master.addHandler(ch)
    
    if log_dir:
        fh = logging.FileHandler(
            os.path.join(log_dir,
            f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger_master.addHandler(fh)

    return logger_master