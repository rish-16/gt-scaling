import json, math, pickle, torch, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import torch_geometric as tg

font = {'family' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

PATH = "MOL_ATTN_SCORE.pickle"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open(PATH, "rb") as f:
    data = CPU_Unpickler(f).load()

# sample = data[0][0]
# scores = data[0][1].detach().numpy()
# print (sample)
# atomic_numbers = [sample.x[i][0] for i in range(sample.x.size(0))]
# print (atomic_numbers)

print (data.keys())

fig, axs = plt.subplots(3, 2)
fig.suptitle('Molecule Attention Scores')
for i in range(3):
    sample = data[i][0]
    # atomic_numbers = [sample.x[j][0] for j in range(sample.x.size(0))]
    
    edge_index = sample.edge_index
    adj_mat = tg.utils.to_dense_adj(edge_index).squeeze(0).numpy()
    print (adj_mat.shape)

    scores = data[i][1].detach().numpy()
    
    axs[i, 0].imshow(adj_mat, cmap="gray")
    axs[i, 1].imshow(scores, cmap="Reds")
    
    new_list = range(math.floor(min(range(0, sample.x.size(0)))), math.ceil(max(range(0, sample.x.size(0))))+1)
    axs[i, 0].set_xticks(new_list)
    axs[i, 0].set_yticks(new_list)
    axs[i, 1].set_xticks(new_list)
    axs[i, 1].set_yticks(new_list)
plt.show()