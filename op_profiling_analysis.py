import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

PATH = "op_bucket_timing.json"

with open(PATH, "r") as f:
    op_dict = json.load(f)

graph_sizes = []
dot = []
softmax = []
av = []

for nnodes, timings in op_dict.items():
    graph_sizes.append(nnodes)
    dot.append(timings["qk"])    
    softmax.append(timings["softmax"])
    av.append(timings["av"])

grp_data = list(zip(graph_sizes, dot, softmax, av))
grp_data.sort(key=lambda rec : rec[0]) # sort by graph size

graph_sizes = [rec[0] for rec in grp_data]
dot = [rec[1] * 128 for rec in grp_data]
softmax = [rec[2] * 128 for rec in grp_data]
av = [rec[3] * 128 for rec in grp_data]

df = pd.DataFrame({
    "Q.T @ K": dot,
    "Softmax": softmax,
    "A @ V": av,
}, index=graph_sizes)
plot = df.plot.bar(stacked=True)
plt.xlabel("Number of atoms")
plt.ylabel("$O(n^2)$ Attention Operation Runtime / sec")
plt.show()

plot.get_figure().savefig("figures/scaling/pcqm4m_op_runtime.pdf", format="pdf")

# "18": {
#     "qk": 0.00010085105895996094,
#     "softmax": 7.152557373046875e-05,
#     "av": 4.57763671875e-05
# },