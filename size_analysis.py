import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

transformer_inferences = {
    1: [76, 0.026436567306518555],
    2: [129, 0.05170011520385742],
    3: [129, 0.06290912628173828],
    4: [129, 0.06813454627990723],
    5: [129, 0.06823229789733887],
    6: [129, 0.03816342353820801],
    7: [129, 0.04131507873535156],
    8: [129, 0.04596376419067383],
    9: [129, 0.045273542404174805],
    10: [129, 0.050623178482055664],
    11: [129, 0.046570777893066406],
    12: [129, 0.04304003715515137],
    13: [129, 0.04316997528076172],
    14: [129, 0.04382753372192383],
    15: [129, 0.9154067039489746],
    16: [129, 0.04761552810668945],
    17: [129, 0.0511627197265625],
    18: [129, 0.0580599308013916],
    19: [19, 0.055736541748046875],
    20: [129, 0.05860757827758789]
}

transformer_x = list(map(lambda x : x[1] / x[0], list(transformer_inferences.values())))
transformer_y = list(transformer_inferences.keys())

plt.plot(transformer_x, transformer_y, color="red", label="Transformer")
plt.xlabel("Number of nodes")
plt.ylabel("Inference time per instance (B = 128)")
plt.legend()
plt.grid()
plt.show()