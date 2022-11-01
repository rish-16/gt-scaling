import json, math, pickle, torch, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import torch_geometric as tg

font = {'family' : 'normal',
        'size'   : 15}

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

fig, axs = plt.subplots(2, 3)
# fig.suptitle('Molecule Attention Scores')
for i in range(3):
    sample = data[i][0]
    # atomic_numbers = [sample.x[j][0] for j in range(sample.x.size(0))]
    
    edge_index = sample.edge_index
    adj_mat = tg.utils.to_dense_adj(edge_index).squeeze(0).numpy()
    print (adj_mat.shape)

    scores = data[i][1].detach().numpy()
    
    axs[0, i].imshow(adj_mat, cmap="gray")
    axs[1, i].imshow(scores, cmap="YlGn")
    
    # new_list = range(math.floor(min(range(0, sample.x.size(0)))), math.ceil(max(range(0, sample.x.size(0))))+1)
    # axs[0, i].set_xticks(new_list)
    # axs[0, i].set_yticks(new_list)
    # axs[1, i].set_xticks(new_list)
    # axs[1, i].set_yticks(new_list)
    
    axs[0, i].set_title(f"Adjacency Matrix {i+1}")
    axs[1, i].set_title(f"Attention Matrix {i+1}")
plt.show()

fig.savefig("figures/attn/mol_attn_scores.pdf", dpi=400, bbox_inches="tight")