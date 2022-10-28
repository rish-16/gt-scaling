import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

transformer_inferences = {
    1: 0.02037954330444336,
    2: 0.02208089828491211,
    3: 0.02248525619506836,
    4: 0.022007226943969727,
    5: 0.022867679595947266,
    6: 0.022356748580932617,
    7: 0.02208423614501953,
    8: 0.022420167922973633,
    9: 0.025183439254760742,
    10: 0.023545503616333008,
    11: 0.022477149963378906,
    12: 0.0220947265625,
    13: 0.022228002548217773,
    14: 0.022696733474731445,
    15: 0.022772789001464844,
    16: 0.022713899612426758,
    17: 0.8700220584869385,
    18: 0.02317047119140625,
    19: 0.023090600967407227,
    20: 0.02292323112487793
}

performer_inferences = {
    1: 0.027149200439453125,
    2: 0.030155420303344727,
    3: 0.03045511245727539,
    4: 0.030425548553466797,
    5: 0.030838489532470703,
    6: 0.030591726303100586,
    7: 0.03197526931762695,
    8: 0.035016775131225586,
    9: 0.03419780731201172,
    10: 0.036809682846069336,
    11: 0.036685943603515625,
    12: 0.03675556182861328,
    13: 0.03953862190246582,
    14: 0.03987741470336914,
    15: 0.042101144790649414,
    16: 0.042803049087524414,
    17: 0.8837883472442627,
    18: 0.03946542739868164,
    19: 0.039417266845703125,
    20: 0.04020333290100098
}

bigbird_inferences = {}

transformer_x = list(transformer_inferences.keys())
transformer_y = list(transformer_inferences.values())

performer_x = list(performer_inferences.keys())
performer_y = list(performer_inferences.values())

bigbird_x = list(bigbird_inferences.keys())
bigbird_y = list(bigbird_inferences.values())

plt.plot(transformer_x, transformer_y, color="green", label="Transformer")
plt.plot(performer_x, performer_y, color="green", label="Performer")
# plt.plot(bigbird_x, bigbird_y, color="green", label="BigBird")
plt.xlabel("Number of nodes (N)")
plt.ylabel("Inference time per instance (B = 128)")
plt.legend()
plt.grid()
plt.show()