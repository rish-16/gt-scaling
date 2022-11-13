import json, math
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

PATH = "layerwise_avg_timings.json"

with open(PATH, "rb") as f:
    data = json.load(f)

graphsizes = []
gmp = []
lmp = []

for nnodes, timings in data.items():
    nnodes = int(nnodes)
    graphsizes.append(nnodes)
    gmp.append(timings["avg_global"])
    lmp.append(timings["avg_local"])

df = pd.DataFrame({
    "Global": gmp,
    "Local": lmp
}, index=graphsizes)

plot = df.plot.bar(stacked=True)
plt.xlabel("Number of atoms")
plt.ylabel("Batch Inference Runtime (B=128) / sec")
plt.show()

"""
"18": {
    "avg_global": 0.0005865335464477539,
    "avg_local": 0.08375601768493653,
    "total": 0.8561151027679443
  },
"""

plot.get_figure().savefig("figures/layerwise/pcqm4m_gvl_bar.pdf", format="pdf")