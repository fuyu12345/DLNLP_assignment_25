# utils/memmap_setup.py

import os
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

DATA_FOLDER = "./memmap_data"
NUM_PREFERENCE_PAIRS = 1000
SEQ_LEN = 1024  

# make sure the folder exists
Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

# define the paths for the memmap files
preference_seq_memmap_path = os.path.join(DATA_FOLDER, "preference_seq.memmap.npy")
prompt_len_memmap_path = os.path.join(DATA_FOLDER, "prompt_len.memmap.npy")
self_reward_memmap_path = os.path.join(DATA_FOLDER, "self_reward.memmap.npy")


# preference_seq_memmap: [NUM_PREFERENCE_PAIRS, 2, SEQ_LEN]
#   -> 表示每个 pair 有 2 条 [prompt+completion] 序列, 每条seq长SEQ_LEN
preference_seq_memmap_shape = (NUM_PREFERENCE_PAIRS, 2, SEQ_LEN)

# prompt_len_memmap: [NUM_PREFERENCE_PAIRS] -> 每个 pair 存一个 prompt_len
prompt_len_memmap_shape = (NUM_PREFERENCE_PAIRS,)

# self_reward_memmap_file: [NUM_PREFERENCE_PAIRS, 2] -> 每个 pair 两个分数
self_reward_memmap_shape = (NUM_PREFERENCE_PAIRS, 2)

# create the memmap files
preference_seq_memmap = open_memmap(
    preference_seq_memmap_path,
    dtype='int',
    mode='w+',
    shape=preference_seq_memmap_shape
)

prompt_len_memmap = open_memmap(
    prompt_len_memmap_path,
    dtype='int',
    mode='w+',
    shape=prompt_len_memmap_shape
)

self_reward_memmap_file = open_memmap(
    self_reward_memmap_path,
    dtype='float32',
    mode='w+',
    shape=self_reward_memmap_shape
)


