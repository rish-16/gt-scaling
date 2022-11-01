import json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

# COMBO1: GatedGCN + Transformer + AtomEncoder
combo1_trial1 = [
    {'global_mp': 0.001085042953491211, 'local_mp': 0.822650671005249},
    {'global_mp': 0.0005664825439453125, 'local_mp': 0.0008668899536132812},
    {'global_mp': 0.0005567073822021484, 'local_mp': 0.0008263587951660156},
    {'global_mp': 0.0005364418029785156, 'local_mp': 0.0007395744323730469},
    {'global_mp': 0.0005133152008056641, 'local_mp': 0.0007791519165039062},
    {'global_mp': 0.0005333423614501953, 'local_mp': 0.0007195472717285156},
    {'global_mp': 0.0005097389221191406, 'local_mp': 0.0007171630859375},
    {'global_mp': 0.0005359649658203125, 'local_mp': 0.0007658004760742188},
    {'global_mp': 0.0005078315734863281, 'local_mp': 0.0007078647613525391},
    {'global_mp': 0.0005326271057128906, 'local_mp': 0.0007498264312744141}
]
combo1_runtime1 = 0.84934401512146

combo1_trial2 = [
    {'global_mp': 0.0009799003601074219, 'local_mp': 0.8283932209014893},
    {'global_mp': 0.000553131103515625, 'local_mp': 0.0008423328399658203},
    {'global_mp': 0.0005702972412109375, 'local_mp': 0.00075531005859375},
    {'global_mp': 0.0005409717559814453, 'local_mp': 0.0009171962738037109},
    {'global_mp': 0.0005354881286621094, 'local_mp': 0.0007784366607666016},
    {'global_mp': 0.0005068778991699219, 'local_mp': 0.0007119178771972656},
    {'global_mp': 0.0005056858062744141, 'local_mp': 0.0007534027099609375},
    {'global_mp': 0.0005626678466796875, 'local_mp': 0.0007061958312988281},
    {'global_mp': 0.0005042552947998047, 'local_mp': 0.0007143020629882812},
    {'global_mp': 0.0005297660827636719, 'local_mp': 0.0007240772247314453}
]
combo1_runtime2 = 0.8552913665771484

combo1_trial3 = [
    {'global_mp': 0.0009739398956298828, 'local_mp': 0.8253533840179443},
    {'global_mp': 0.0005621910095214844, 'local_mp': 0.0008089542388916016},
    {'global_mp': 0.0005381107330322266, 'local_mp': 0.0007779598236083984},
    {'global_mp': 0.000553131103515625, 'local_mp': 0.0009369850158691406},
    {'global_mp': 0.0005609989166259766, 'local_mp': 0.0007874965667724609},
    {'global_mp': 0.0005207061767578125, 'local_mp': 0.0007734298706054688},
    {'global_mp': 0.000518798828125, 'local_mp': 0.0007264614105224609},
    {'global_mp': 0.0005023479461669922, 'local_mp': 0.0007231235504150391},
    {'global_mp': 0.0005168914794921875, 'local_mp': 0.0007271766662597656},
    {'global_mp': 0.0005156993865966797, 'local_mp': 0.0007219314575195312}
]
combo1_runtime3 = 0.8521368503570557


combo1_trials = zip(combo1_trial1, combo1_trial2, combo1_trial3)
combo1_runtimes = [combo1_runtime1, combo1_runtime2, combo1_runtime3]

combo1_gmp_times = []
combo1_lmp_times = []

for r1, r2, r3 in combo1_trials:
    gmp1 = r1["global_mp"]
    gmp2 = r2["global_mp"]
    gmp3 = r3["global_mp"]

    lmp1 = r1["local_mp"]
    lmp2 = r2["local_mp"]
    lmp3 = r3["local_mp"]

    combo1_gmp_times.append((gmp1 + gmp2 + gmp3) / 3)
    combo1_lmp_times.append((lmp1 + lmp2 + lmp3) / 3)

combo1_layer_idx = list(range(1, 11))[1:]
combo1_gmp_times = combo1_gmp_times[1:]
combo1_lmp_times = combo1_lmp_times[1:]

df = pd.DataFrame({
    "Local MP": combo1_lmp_times,
    "Global MP": combo1_gmp_times
}, index=combo1_layer_idx)

plot = df.plot.bar()
plt.xlabel("Layer Index (after input layer)")
plt.ylabel("Batch Inference Runtime (B=128) / sec")
plt.xticks(rotation=0)
plt.show()

plot.get_figure().savefig("figures/layerwise/gatedgcn-tf-atomenc.pdf", format="pdf")