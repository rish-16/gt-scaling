import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper right"})

TF_LARGE_TEST_PATH = "results/zinc/zinc-subset-GPSlarge-RWSE-LapPE-Transformer/agg/test/stats.json"
TF_LARGE_TRAIN_PATH = "results/zinc/zinc-subset-GPSlarge-RWSE-LapPE-Transformer/agg/train/stats.json"

PF_LARGE_TEST_PATH = "results/zinc/zinc-subset-GPSlarge-RWSE-LapPE-Performer/agg/test/stats.json"
PF_LARGE_TRAIN_PATH = "results/zinc/zinc-subset-GPSlarge-RWSE-LapPE-Performer/agg/train/stats.json"

with open(TF_LARGE_TEST_PATH, "r") as f:
    data = json.load(f)["stats"]
    transformer_test_mae = [rec["mae"] for rec in data][1:]
    transformer_test_epochs = [rec["epoch"] for rec in data][1:]

with open(TF_LARGE_TRAIN_PATH, "r") as f:
    data = json.load(f)["stats"]
    transformer_train_mae = [rec["mae"] for rec in data][1:]
    transformer_train_epochs = [rec["epoch"] for rec in data][1:]

with open(PF_LARGE_TEST_PATH, "r") as f:
    data = json.load(f)["stats"]
    performer_test_mae = [rec["mae"] for rec in data][1:]
    performer_test_epochs = [rec["epoch"] for rec in data][1:]

with open(PF_LARGE_TRAIN_PATH, "r") as f:
    data = json.load(f)["stats"]
    performer_train_mae = [rec["mae"] for rec in data][1:]
    performer_train_epochs = [rec["epoch"] for rec in data][1:]    

plt.plot(transformer_train_epochs, transformer_train_mae, label="TF-Large Train", color="red")
plt.plot(transformer_test_epochs, transformer_test_mae, label="TF-Large Test", color="green")
plt.plot(performer_train_epochs, performer_train_mae, label="PF-Large Train", color="blue")
plt.plot(performer_test_epochs, performer_test_mae, label="PF-Large Test", color="orange")
plt.grid()
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Test MAE (ZINC)")
plt.show()