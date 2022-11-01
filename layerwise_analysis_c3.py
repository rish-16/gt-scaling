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
    {'global_mp': 0.0010802745819091797, 'local_mp': 0.8905887603759766},
    {'global_mp': 0.0008027553558349609, 'local_mp': 0.3846144676208496},
    {'global_mp': 0.0005660057067871094, 'local_mp': 0.0011196136474609375},
    {'global_mp': 0.0005657672882080078, 'local_mp': 0.0014874935150146484},
    {'global_mp': 0.0005393028259277344, 'local_mp': 0.0009753704071044922},
    {'global_mp': 0.0005204677581787109, 'local_mp': 0.000919342041015625},
    {'global_mp': 0.0005314350128173828, 'local_mp': 0.0009057521820068359},
    {'global_mp': 0.000530242919921875, 'local_mp': 0.0009734630584716797},
    {'global_mp': 0.0005261898040771484, 'local_mp': 0.0009717941284179688},
    {'global_mp': 0.0005567073822021484, 'local_mp': 0.0008955001831054688}
]
combo3_runtime1 = 1.3046894073486328

combo3_trial2 = [
    {'global_mp': 0.0010905265808105469, 'local_mp': 0.8878934383392334},
    {'global_mp': 0.0008118152618408203, 'local_mp': 0.281660795211792},
    {'global_mp': 0.0006284713745117188, 'local_mp': 0.0012240409851074219},
    {'global_mp': 0.0005474090576171875, 'local_mp': 0.0014090538024902344},
    {'global_mp': 0.0005271434783935547, 'local_mp': 0.0009441375732421875},
    {'global_mp': 0.0005435943603515625, 'local_mp': 0.0009953975677490234},
    {'global_mp': 0.0005452632904052734, 'local_mp': 0.0009794235229492188},
    {'global_mp': 0.0005142688751220703, 'local_mp': 0.0009014606475830078},
    {'global_mp': 0.0005905628204345703, 'local_mp': 0.0011162757873535156},
    {'global_mp': 0.0005660057067871094, 'local_mp': 0.0009648799896240234}
]
combo3_runtime2 = 1.1997270584106445

combo3_trial3 = [
    {'global_mp': 0.001096487045288086, 'local_mp': 0.8860492706298828},
    {'global_mp': 0.0008096694946289062, 'local_mp': 0.28184080123901367},
    {'global_mp': 0.0005536079406738281, 'local_mp': 0.0010943412780761719},
    {'global_mp': 0.0005483627319335938, 'local_mp': 0.001398324966430664},
    {'global_mp': 0.0005321502685546875, 'local_mp': 0.0010173320770263672},
    {'global_mp': 0.0005824565887451172, 'local_mp': 0.0009679794311523438},
    {'global_mp': 0.0005402565002441406, 'local_mp': 0.0010762214660644531},
    {'global_mp': 0.0005095005035400391, 'local_mp': 0.0009717941284179688},
    {'global_mp': 0.0005283355712890625, 'local_mp': 0.0009014606475830078},
    {'global_mp': 0.00051116943359375, 'local_mp': 0.0009617805480957031}
]
combo3_runtime3 = 1.197601318359375


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

combo3_layer_idx = list(range(1, 11))[2:]
combo3_gmp_times = combo3_gmp_times[2:]
combo3_lmp_times = combo3_lmp_times[2:]

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