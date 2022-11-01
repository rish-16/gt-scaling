import json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

# COMBO2: GINE + Transformer + AtomEncoder
combo2_trial1 = [
    {'global_mp': 0.0011868476867675781, 'local_mp': 0.8325037956237793},
    {'global_mp': 0.0005335807800292969, 'local_mp': 0.00033354759216308594},
    {'global_mp': 0.0005331039428710938, 'local_mp': 0.0003046989440917969},
    {'global_mp': 0.0004994869232177734, 'local_mp': 0.0002980232238769531},
    {'global_mp': 0.0005261898040771484, 'local_mp': 0.000308990478515625},
    {'global_mp': 0.0004813671112060547, 'local_mp': 0.0002918243408203125},
    {'global_mp': 0.00048065185546875, 'local_mp': 0.0002732276916503906},
    {'global_mp': 0.00048279762268066406, 'local_mp': 0.0002703666687011719},
    {'global_mp': 0.0005249977111816406, 'local_mp': 0.0002865791320800781},
    {'global_mp': 0.0005164146423339844, 'local_mp': 0.0002884864807128906}
]
combo2_runtime1 = 0.8561856746673584

combo2_trial2 = [
    {'global_mp': 0.0011446475982666016, 'local_mp': 0.8315293788909912},
    {'global_mp': 0.0005631446838378906, 'local_mp': 0.0003495216369628906},
    {'global_mp': 0.0004956722259521484, 'local_mp': 0.0002994537353515625},
    {'global_mp': 0.0005009174346923828, 'local_mp': 0.0003025531768798828},
    {'global_mp': 0.0004906654357910156, 'local_mp': 0.000278472900390625},
    {'global_mp': 0.0004868507385253906, 'local_mp': 0.0003027915954589844},
    {'global_mp': 0.00048828125, 'local_mp': 0.00027632713317871094},
    {'global_mp': 0.0005979537963867188, 'local_mp': 0.0003120899200439453},
    {'global_mp': 0.000522613525390625, 'local_mp': 0.0002899169921875},
    {'global_mp': 0.0006017684936523438, 'local_mp': 0.0002722740173339844}
]
combo2_runtime2 = 0.8557133674621582

combo2_trial3 = [
    {'global_mp': 0.0011265277862548828, 'local_mp': 0.8572933673858643},
    {'global_mp': 0.0006518363952636719, 'local_mp': 0.0003726482391357422},
    {'global_mp': 0.0006730556488037109, 'local_mp': 0.00037598609924316406},
    {'global_mp': 0.0006196498870849609, 'local_mp': 0.00032520294189453125},
    {'global_mp': 0.0006091594696044922, 'local_mp': 0.0003466606140136719},
    {'global_mp': 0.0006089210510253906, 'local_mp': 0.000316619873046875},
    {'global_mp': 0.0006184577941894531, 'local_mp': 0.00035834312438964844},
    {'global_mp': 0.0006232261657714844, 'local_mp': 0.0003330707550048828},
    {'global_mp': 0.0005409717559814453, 'local_mp': 0.0003027915954589844},
    {'global_mp': 0.0006103515625, 'local_mp': 0.0003008842468261719}
]
combo2_runtime3 = 0.8834118843078613


combo2_trials = zip(combo2_trial1, combo2_trial2, combo2_trial3)
combo2_runtimes = [combo2_runtime1, combo2_runtime2, combo2_runtime3]

combo2_gmp_times = []
combo2_lmp_times = []

for r1, r2, r3 in combo2_trials:
    gmp1 = r1["global_mp"]
    gmp2 = r2["global_mp"]
    gmp3 = r3["global_mp"]

    lmp1 = r1["local_mp"]
    lmp2 = r2["local_mp"]
    lmp3 = r3["local_mp"]

    combo2_gmp_times.append((gmp1 + gmp2 + gmp3) / 3)
    combo2_lmp_times.append((lmp1 + lmp2 + lmp3) / 3)

combo2_layer_idx = list(range(1, 11))[1:]
combo2_gmp_times = combo2_gmp_times[1:]
combo2_lmp_times = combo2_lmp_times[1:]

df = pd.DataFrame({
    "Local MP": combo2_lmp_times,
    "Global MP": combo2_gmp_times
}, index=combo2_layer_idx)

plot = df.plot.bar()
plt.xlabel("Layer Index (after input layer)")
plt.ylabel("Batch Inference Runtime (B=128) / sec")
plt.xticks(rotation=0)
plt.show()

plot.get_figure().savefig("figures/layerwise/gine-tf-atomenc.pdf", format="pdf")