import json, math, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

PATH = "MOL_ATTN_SCORE.pickle"

with open(PATH, "rb") as f:
    data = pickle.load(f)

print (data.keys())