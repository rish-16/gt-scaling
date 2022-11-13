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

PATH = "op_bucket_timing2.json" # PATH2 by default
PATH3 = "op_bucket_timing3.json"
PATH4 = "op_bucket_timing4.json"

with open(PATH, "r") as f:
    op_dict = json.load(f)

with open(PATH3, "r") as f3:
    op_dict3 = json.load(f3)

with open(PATH4, "r") as f4:
    op_dict4 = json.load(f4)           

graph_sizes = []
dot = []
softmax = []
av = []

for nnodes, record in op_dict.items():
    record = record[0]
    graph_sizes.append(nnodes)
    dot.append(record["attention_ops"]["qk"])
    softmax.append(record["attention_ops"]["softmax"])
    av.append(record["attention_ops"]["av"])

for i, (nnodes, record) in enumerate(op_dict3.items()):
    pprint (record)
    record = record[0]
    dot[i] += record["attention_ops"]["qk"]
    softmax[i] += record["attention_ops"]["softmax"]
    av[i] += record["attention_ops"]["av"]

for i, (nnodes, record) in enumerate(op_dict4.items()):
    pprint (record)
    record = record[0]
    dot[i] += record["attention_ops"]["qk"]
    softmax[i] += record["attention_ops"]["softmax"]
    av[i] += record["attention_ops"]["av"]

    dot[i] /= 3
    softmax[i] /= 3
    av[i] /= 3

grp_data = list(zip(graph_sizes, dot, softmax, av))
grp_data.sort(key=lambda rec : rec[0]) # sort by graph size

graph_sizes = [rec[0] for rec in grp_data]
dot = [rec[1] for rec in grp_data]
softmax = [rec[2] for rec in grp_data]
av = [rec[3] for rec in grp_data]

new_graph_sizes = []
new_dot = []
new_softmax = []
new_av = []

anomalies = [8, 11, 13, 15, 22, 25, 27, 30, 31, 33, 37, 40]

for i in range(len(graph_sizes)):
    if i not in anomalies:
        new_graph_sizes.append(graph_sizes[i])
        new_dot.append(dot[i])
        new_softmax.append(softmax[i])
        new_av.append(av[i])

df = pd.DataFrame({
    "Q.T @ K": new_dot,
    "Softmax": new_softmax,
    "A @ V": new_av,
}, index=new_graph_sizes)
plot = df.plot.bar(stacked=True)
plt.xlabel("Number of atoms")
plt.ylabel("Attention Ops Runtime (B=128) / sec")
plt.show()

plot.get_figure().savefig("figures/scaling/pcqm4m_attn_op_runtime.pdf", format="pdf")

# "18": {
#     "qk": 0.00010085105895996094,
#     "softmax": 7.152557373046875e-05,
#     "av": 4.57763671875e-05
# },