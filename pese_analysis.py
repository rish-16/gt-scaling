import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

PESE_PATH_1_TEST = "results/pese/pcqm4m-full-GPSmedium-Transformer-1-PESE/agg/test/best.json"
PESE_PATH_2_TEST = "results/pese/pcqm4m-full-GPSmedium-Transformer-2-PESE/agg/test/best.json"
PESE_PATH_3_TEST = "results/pese/pcqm4m-full-GPSmedium-Transformer-3-PESE/agg/test/best.json"
PESE_PATH_4_TEST = "results/pese/pcqm4m-full-GPSmedium-Transformer-4-PESE/agg/test/best.json"
PESE_PATHS_TEST = [PESE_PATH_1_TEST, PESE_PATH_2_TEST, PESE_PATH_3_TEST, PESE_PATH_4_TEST]

PESE_PATH_1_TRAIN = "results/pese/pcqm4m-full-GPSmedium-Transformer-1-PESE/agg/train/best.json"
PESE_PATH_2_TRAIN = "results/pese/pcqm4m-full-GPSmedium-Transformer-2-PESE/agg/train/best.json"
PESE_PATH_3_TRAIN = "results/pese/pcqm4m-full-GPSmedium-Transformer-3-PESE/agg/train/best.json"
PESE_PATH_4_TRAIN = "results/pese/pcqm4m-full-GPSmedium-Transformer-4-PESE/agg/train/best.json"
PESE_PATHS_TRAIN = [PESE_PATH_1_TRAIN, PESE_PATH_2_TRAIN, PESE_PATH_3_TRAIN, PESE_PATH_4_TRAIN]

pese_counts = [1, 2, 3, 4]
test_mae = []
train_mae = []

for path in PESE_PATHS_TEST:
    with open(path, "r") as f:
        data = json.load(f)['mae']
        test_mae.append(data)

for path in PESE_PATHS_TRAIN:
    with open(path, "r") as f:
        data = json.load(f)['mae']
        train_mae.append(data)     

# plt.plot(pese_counts, test_mae, "x", color="red", markersize=10)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(121)

new_list = range(math.floor(min(pese_counts)), math.ceil(max(pese_counts))+1)
plt.xticks(new_list)

ax.plot(pese_counts, train_mae, marker="*", markersize=10, color="orange")
plt.grid(linestyle="dashed")
plt.xlabel("Number of PESE concatenated")
plt.ylabel("Train MAE")

ax = fig.add_subplot(122)

new_list = range(math.floor(min(pese_counts)), math.ceil(max(pese_counts))+1)
plt.xticks(new_list)

ax.plot(pese_counts, test_mae, marker="o", markersize=10, color="red")
plt.grid(linestyle="dashed")
plt.xlabel("Number of PESE concatenated")
plt.ylabel("Test MAE")

plt.show()
fig.savefig("figures/pese/pcqm4m_combined_compaison.pdf", dpi=400, bbox_inches='tight')