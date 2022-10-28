import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

# ---------------------------------------------------------------------------------------------

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

transformer_paths_test = [SMALL_TRANSFORMER_PATH_TEST, MEDIUM_TRANSFORMER_PATH_TEST, LARGE_TRANSFORMER_PATH_TEST]
performer_paths_test = [SMALL_PERFORMER_PATH_TEST, MEDIUM_PERFORMER_PATH_TEST, LARGE_PERFORMER_PATH_TEST]
bigbird_paths_test = [SMALL_BIGBIRD_PATH_TEST, MEDIUM_BIGBIRD_PATH_TEST, LARGE_BIGBIRD_PATH_TEST]

transformer_paths_train = [SMALL_TRANSFORMER_PATH_TRAIN, MEDIUM_TRANSFORMER_PATH_TRAIN, LARGE_TRANSFORMER_PATH_TRAIN]
performer_paths_train = [SMALL_PERFORMER_PATH_TRAIN, MEDIUM_PERFORMER_PATH_TRAIN, LARGE_PERFORMER_PATH_TRAIN]
bigbird_paths_train = [SMALL_BIGBIRD_PATH_TRAIN, MEDIUM_BIGBIRD_PATH_TRAIN, LARGE_BIGBIRD_PATH_TRAIN]

# ---------------------------------------------------------------------------------------------

transformer_mae = []
transformer_time = []

performer_mae = []
performer_time = []

bigbird_mae = []
bigbird_time = []

classes = ["S", "M", "L"]
for path in transformer_paths_test:
    with open(path, "r") as f:
        data = json.load(f)
        mae = data["mae"]
        time = data["time_epoch"]
        transformer_mae.append(mae)
        transformer_time.append(time)

for path in performer_paths_test:
    with open(path, "r") as f:
        data = json.load(f)
        mae = data["mae"]
        time = data["time_epoch"]
        performer_mae.append(mae)
        performer_time.append(time)

for path in bigbird_paths_test:
    with open(path, "r") as f:
        data = json.load(f)
        mae = data["mae"]
        time = data["time_epoch"]
        bigbird_mae.append(mae)
        bigbird_time.append(time)

# ---------------------------------------------------------------------------------------------        

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(transformer_time, transformer_mae, marker="o", markersize=10, color="red")
for i, xy in enumerate(zip(transformer_time, transformer_mae)):
    ax.annotate(f'TF-{classes[i]}', xy=xy, textcoords='data')
ax.plot(bigbird_time, bigbird_mae, marker="+", markersize=10, color="green")
for i, xy in enumerate(zip(bigbird_time, bigbird_mae)):
    ax.annotate(f'BB-{classes[i]}', xy=xy, textcoords='data')

ax.grid()
plt.xlabel("Inference Time (s)")
plt.ylabel("Test MAE")
plt.title("PCQM4Mv2-Subset | Transformer vs BigBird")
plt.show()

# ---------------------------------------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(transformer_time, transformer_mae, marker="o", markersize=10, color="red")
for i, xy in enumerate(zip(transformer_time, transformer_mae)):
    ax.annotate(f'TF-{classes[i]}', xy=xy, textcoords='data')
ax.plot(performer_time, performer_mae, marker="+", markersize=10, color="green")
for i, xy in enumerate(zip(performer_time, performer_mae)):
    ax.annotate(f'PF-{classes[i]}', xy=xy, textcoords='data')

ax.grid()
plt.xlabel("Inference Time (s)")
plt.ylabel("Test MAE")
plt.title("PCQM4Mv2-Subset | Transformer vs Performer")
plt.show()