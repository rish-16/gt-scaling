import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

PESE_1_PATH = "results/pese/pcqm4m-full-GPSmedium-Transformer-1-PESE/agg/test/stats.json"
PESE_2_PATH = "results/pese/pcqm4m-full-GPSmedium-Transformer-2-PESE/agg/test/stats.json"
PESE_3_PATH = "results/pese/pcqm4m-full-GPSmedium-Transformer-3-PESE/agg/test/stats.json"
PESE_4_PATH = "results/pese/pcqm4m-full-GPSmedium-Transformer-4-PESE/agg/test/stats.json"

PATHS = [PESE_1_PATH, PESE_2_PATH, PESE_3_PATH, PESE_4_PATH]

pese_1_data = []
pese_2_data = []
pese_3_data = []
pese_4_data = []

data = []

for i, path in enumerate(PATHS):
    temp = []
    with open(path, "r") as f:
        recs = json.load(f)["stats"]
        for j in range(len(recs)):
            temp.append(recs[j]["mae"])
    data.append(temp)

epochs = list(range(100))

fig = plt.figure()
plt.plot(epochs[1:], data[0][1:], label="1 PESE", color="orange")
plt.plot(epochs[1:], data[1][1:], label="2 PESE", color="blue")
plt.plot(epochs[1:], data[2][1:], label="3 PESE", color="green")
plt.plot(epochs[1:], data[3][1:], label="4 PESE", color="red")
plt.xlabel("Epochs")
plt.ylabel("Test MAE")
plt.legend()
plt.grid(linestyle="dashed")
plt.show()

fig.savefig("figures/pese/pcqm4m_pese_test_convergence.pdf", dpi=400, bbox_inches="tight")