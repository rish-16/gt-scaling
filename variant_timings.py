import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

SMALL_TRANSFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-Transformer/agg/test/best.json"
SMALL_PERFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-Performer/agg/test/best.json"
SMALL_BIGBIRD_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-BigBird/agg/test/best.json"

MEDIUM_TRANSFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-Transformer/agg/test/best.json"
MEDIUM_PERFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-Performer/agg/test/best.json"
MEDIUM_BIGBIRD_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-BigBird/agg/test/best.json"

LARGE_TRANSFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-Transformer/agg/test/best.json"
LARGE_PERFORMER_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-Performer/agg/test/best.json"
LARGE_BIGBIRD_PATH_TEST = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-BigBird/agg/test/best.json"

SMALL_TRANSFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-Transformer/agg/train/best.json"
SMALL_PERFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-Performer/agg/train/best.json"
SMALL_BIGBIRD_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSsmall-RWSE-LapPE-BigBird/agg/train/best.json"

MEDIUM_TRANSFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-Transformer/agg/train/best.json"
MEDIUM_PERFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-Performer/agg/train/best.json"
MEDIUM_BIGBIRD_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSmedium-RWSE-LapPE-BigBird/agg/train/best.json"

LARGE_TRANSFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-Transformer/agg/train/best.json"
LARGE_PERFORMER_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-Performer/agg/train/best.json"
LARGE_BIGBIRD_PATH_TRAIN = "/Users/rish/Desktop/UROP/gt-scaling/results/variants/pcqm4m-subset-GPSlarge-RWSE-LapPE-BigBird/agg/train/best.json"

# transformer, performer, bigbird
SMALL_PATHS = [SMALL_TRANSFORMER_PATH_TEST, SMALL_PERFORMER_PATH_TEST, SMALL_BIGBIRD_PATH_TEST]
MEDIUM_PATHS = [MEDIUM_TRANSFORMER_PATH_TEST, MEDIUM_PERFORMER_PATH_TEST, MEDIUM_BIGBIRD_PATH_TEST]
LARGE_PATHS = [LARGE_TRANSFORMER_PATH_TEST, "", LARGE_BIGBIRD_PATH_TEST]
classes = ["S", "M", "L"]

labels = ['Transformer', 'Performer', 'BigBird']
small_times = []
medium_times = []
large_times = []

for path in SMALL_PATHS:
    if path != "":
        with open(path, "r") as f: 
            data = json.load(f)
            time = data['time_epoch'] # divide by test size for average test inference time?
            small_times.append(time)
    else:
        small_times.append(0)

for path in MEDIUM_PATHS:
    if path != "":
        with open(path, "r") as f: 
            data = json.load(f)
            time = data['time_epoch'] # divide by test size for average test inference time?
            medium_times.append(time)
    else:
        medium_times.append(0)

for path in LARGE_PATHS:
    if path != "":
        with open(path, "r") as f: 
            data = json.load(f)
            time = data['time_epoch'] # divide by test size for average test inference time?
            large_times.append(time)
    else:
        large_times.append(0)

df = pd.DataFrame([
    ['S','Transformer', small_times[0]],
    ['S','Performer', small_times[1]],
    ['S','BigBird', small_times[2]],
    
    ['M','Transformer',medium_times[0]],
    ['M','Performer', medium_times[1]],
    ['M','BigBird', medium_times[2]],

    ['L','Transformer', large_times[0]],
    ['L','Performer', large_times[1]],
    ['L','BigBird', large_times[2]]
], columns=['Scale','Model', 'val'])

df.pivot("Model", "Scale", "val").plot(kind='bar')

plt.title("PCQM4Mv2-Subset Test Inference Times")
plt.xticks(rotation = 0)
plt.show()