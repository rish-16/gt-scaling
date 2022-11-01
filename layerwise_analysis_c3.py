import json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

# COMBO3: GAT + Transformer + AtomEncoder
combo3_trial1 = [
    
]
combo3_runtime1 = 

combo3_trial2 = [
    
]
combo3_runtime2 = 

combo3_trial3 = [
   
]
combo3_runtime3 = 


combo3_trials = zip(combo3_trial1, combo3_trial2, combo3_trial3)
combo3_runtimes = [combo3_runtime1, combo3_runtime2, combo3_runtime3]

combo3_gmp_times = []
combo3_lmp_times = []

for r1, r2, r3 in combo3_trials:
    gmp1 = r1["global_mp"]
    gmp2 = r2["global_mp"]
    gmp3 = r3["global_mp"]

    lmp1 = r1["local_mp"]
    lmp2 = r2["local_mp"]
    lmp3 = r3["local_mp"]

    combo3_gmp_times.append((gmp1 + gmp2 + gmp3) / 3)
    combo3_lmp_times.append((lmp1 + lmp2 + lmp3) / 3)

combo3_layer_idx = list(range(1, 11))[1:]
combo3_gmp_times = combo3_gmp_times[1:]
combo3_lmp_times = combo3_lmp_times[1:]

df = pd.DataFrame({
    "Local MP": combo3_lmp_times,
    "Global MP": combo3_gmp_times
}, index=combo3_layer_idx)

plot = df.plot.bar()
plt.xlabel("Layer Index (after input layer)")
plt.ylabel("Batch Inference Runtime (B=128) / sec")
plt.xticks(rotation=0)
plt.show()

plot.get_figure().savefig("figures/layerwise/gat-tf-atomenc.pdf", format="pdf")