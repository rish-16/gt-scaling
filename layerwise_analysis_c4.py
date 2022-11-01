import json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

# COMBO4: PNA + Transformer + AtomEncoder
combo4_trial1 = [
    {'global_mp': 0.0010313987731933594, 'local_mp': 0.819810152053833},
    {'global_mp': 0.0005540847778320312, 'local_mp': 0.0008676052093505859},
    {'global_mp': 0.0005495548248291016, 'local_mp': 0.0008640289306640625},
    {'global_mp': 0.0006134510040283203, 'local_mp': 0.0008261203765869141},
    {'global_mp': 0.0005428791046142578, 'local_mp': 0.0008380413055419922},
    {'global_mp': 0.0005013942718505859, 'local_mp': 0.0007801055908203125},
    {'global_mp': 0.0004949569702148438, 'local_mp': 0.0007822513580322266},
    {'global_mp': 0.0004918575286865234, 'local_mp': 0.0007798671722412109},
    {'global_mp': 0.0005221366882324219, 'local_mp': 0.0008003711700439453},
    {'global_mp': 0.0005033016204833984, 'local_mp': 0.0008089542388916016}
]
combo4_runtime1 = 0.8476681709289551

combo4_trial2 = [
    {'global_mp': 0.0010769367218017578, 'local_mp': 0.8077077865600586},
    {'global_mp': 0.0005595684051513672, 'local_mp': 0.0008642673492431641},
    {'global_mp': 0.0005002021789550781, 'local_mp': 0.0008037090301513672},
    {'global_mp': 0.0005002021789550781, 'local_mp': 0.0008153915405273438},
    {'global_mp': 0.0004863739013671875, 'local_mp': 0.0007572174072265625},
    {'global_mp': 0.00048613548278808594, 'local_mp': 0.0007781982421875},
    {'global_mp': 0.0005843639373779297, 'local_mp': 0.0008533000946044922},
    {'global_mp': 0.0005207061767578125, 'local_mp': 0.0008184909820556641},
    {'global_mp': 0.0004818439483642578, 'local_mp': 0.0008072853088378906},
    {'global_mp': 0.0005075931549072266, 'local_mp': 0.0008003711700439453}
]
combo4_runtime2 = 0.835219144821167

combo4_trial3 = [
    {'global_mp': 0.001188516616821289, 'local_mp': 0.8206408023834229},
    {'global_mp': 0.0006859302520751953, 'local_mp': 0.0010707378387451172},
    {'global_mp': 0.0006897449493408203, 'local_mp': 0.001108407974243164},
    {'global_mp': 0.0006916522979736328, 'local_mp': 0.0009908676147460938},
    {'global_mp': 0.0006139278411865234, 'local_mp': 0.0009593963623046875},
    {'global_mp': 0.0006136894226074219, 'local_mp': 0.001001119613647461},
    {'global_mp': 0.0006961822509765625, 'local_mp': 0.0009946823120117188},
    {'global_mp': 0.0006563663482666016, 'local_mp': 0.0009462833404541016},
    {'global_mp': 0.0006766319274902344, 'local_mp': 0.0010476112365722656},
    {'global_mp': 0.0006372928619384766, 'local_mp': 0.0012331008911132812}
]
combo4_runtime3 = 0.853724479675293

combo4_trials = zip(combo4_trial1, combo4_trial2, combo4_trial3)
combo4_runtimes = [combo4_runtime1, combo4_runtime2, combo4_runtime3]

combo4_gmp_times = []
combo4_lmp_times = []

for r1, r2, r3 in combo4_trials:
    gmp1 = r1["global_mp"]
    gmp2 = r2["global_mp"]
    gmp3 = r3["global_mp"]

    lmp1 = r1["local_mp"]
    lmp2 = r2["local_mp"]
    lmp3 = r3["local_mp"]

    combo4_gmp_times.append((gmp1 + gmp2 + gmp3) / 3)
    combo4_lmp_times.append((lmp1 + lmp2 + lmp3) / 3)

combo4_layer_idx = list(range(1, 11))[1:]
combo4_gmp_times = combo4_gmp_times[1:]
combo4_lmp_times = combo4_lmp_times[1:]

df = pd.DataFrame({
    "Local MP": combo4_lmp_times,
    "Global MP": combo4_gmp_times
}, index=combo4_layer_idx)

plot = df.plot.bar()
plt.xlabel("Layer Index (after input layer)")
plt.ylabel("Batch Inference Runtime (B=128) / sec")
plt.xticks(rotation=0)
plt.show()

plot.get_figure().savefig("figures/layerwise/pna-tf-atomenc.pdf", format="pdf")