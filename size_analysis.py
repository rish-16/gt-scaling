import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

transformer_inferences1 = {
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
    # 17: 0.8700220584869385,
    18: 0.02317047119140625,
    19: 0.023090600967407227,
    20: 0.02292323112487793
}

transformer_inferences2 = {
    1: 0.020636796951293945,
    2: 0.022446632385253906,
    3: 0.022495269775390625,
    4: 0.022426843643188477,
    5: 0.02290964126586914,
    6: 0.022312402725219727,
    7: 0.02267742156982422,
    8: 0.022769927978515625,
    9: 0.022472858428955078,
    10: 0.023607969284057617,
    11: 0.022319793701171875,
    12: 0.022353649139404297,
    13: 0.022261381149291992,
    14: 0.022441387176513672,
    15: 0.022579193115234375,
    16: 0.0225067138671875,
    #  17: 0.8686370849609375,
    18: 0.025827884674072266,
    19: 0.02344059944152832,
    20: 0.022981643676757812
}

transformer_inferences3 = {
    1: 0.020218849182128906,
    2: 0.021275758743286133,
    3: 0.02179098129272461,
    4: 0.021663188934326172,
    5: 0.02181720733642578,
    6: 0.021494626998901367,
    7: 0.021434783935546875,
    8: 0.02436065673828125,
    9: 0.02167510986328125,
    10: 0.022881746292114258,
    11: 0.021719932556152344,
    12: 0.021876811981201172,
    13: 0.02196216583251953,
    14: 0.022072792053222656,
    15: 0.021803855895996094,
    16: 0.022026777267456055,
    # 17: 0.8702347278594971,
    18: 0.022715330123901367,
    19: 0.02263355255126953,
    20: 0.021924972534179688
}

performer_inferences1 = {
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
    # 17: 0.8837883472442627,
    18: 0.03946542739868164,
    19: 0.039417266845703125,
    20: 0.04020333290100098
}

performer_inferences2 = {
    1: 0.027695417404174805,
    2: 0.029903173446655273,
    3: 0.030136823654174805,
    4: 0.030477523803710938,
    5: 0.030761241912841797,
    6: 0.031225204467773438,
    7: 0.03214097023010254,
    8: 0.03281283378601074,
    9: 0.034211158752441406,
    10: 0.03629136085510254,
    11: 0.03588557243347168,
    12: 0.03674125671386719,
    13: 0.038849592208862305,
    14: 0.03849148750305176,
    15: 0.04442763328552246,
    16: 0.04264998435974121,
    # 17: 0.8910841941833496,
    18: 0.03968667984008789,
    19: 0.0397951602935791,
    20: 0.03992176055908203
}

performer_inferences3 = {
    1: 0.02771615982055664,
    2: 0.030280351638793945,
    3: 0.03052210807800293,
    4: 0.030326128005981445,
    5: 0.03078603744506836,
    6: 0.0308229923248291,
    7: 0.03204226493835449,
    8: 0.03264498710632324,
    9: 0.03445005416870117,
    10: 0.03708195686340332,
    11: 0.03650164604187012,
    12: 0.0370023250579834,
    13: 0.03927111625671387,
    14: 0.04045534133911133,
    15: 0.04503226280212402,
    16: 0.04298043251037598,
    # 17: 0.8883919715881348,
    18: 0.03986692428588867,
    19: 0.040231943130493164,
    20: 0.0398557186126709
}

bigbird_inferences = {}

transformer_x = list(transformer_inferences1.keys())
transformer_y1 = list(transformer_inferences1.values())
transformer_y2 = list(transformer_inferences2.values())
transformer_y3 = list(transformer_inferences3.values())

transformer_y = []
for y1, y2, y3 in zip(transformer_y1, transformer_y2, transformer_y3):
    transformer_y.append([y1, y2, y3])
transformer_y = np.asarray(transformer_y)
transformer_y_avg = transformer_y.mean(axis=1)
transformer_y_std = transformer_y.std(axis=1)

# print (transformer_y_avg)
# print (sum([0.02037954330444336, 0.020636796951293945, 0.020218849182128906]) / 3)
# print (transformer_y_avg[0])

performer_x = list(performer_inferences1.keys())
performer_y1 = list(performer_inferences1.values())
performer_y2 = list(performer_inferences2.values())
performer_y3 = list(performer_inferences3.values())

performer_y = []
for y1, y2, y3 in zip(performer_y1, performer_y2, performer_y3):
    performer_y.append([y1, y2, y3])
performer_y = np.asarray(performer_y)
performer_y_avg = performer_y.mean(axis=1)
performer_y_std = performer_y.std(axis=1)

# bigbird_x = list(bigbird_inferences1.keys())
# bigbird_y1 = list(bigbird_inferences1.values())
# bigbird_y2 = list(bigbird_inferences2.values())
# bigbird_y3 = list(bigbird_inferences3.values())

xint = range(min(transformer_x), math.ceil(max(transformer_x))+1)
matplotlib.pyplot.xticks(xint)

plt.plot(transformer_x, transformer_y_avg, color="green", label="Transformer", markersize=10)
plt.fill_between(
    transformer_x,
    np.asarray(transformer_y_avg) - np.asarray(transformer_y_std), 
    np.asarray(transformer_y_avg) + np.asarray(transformer_y_std),
    alpha=0.4
)

plt.plot(performer_x, performer_y_avg, color="red", label="Performer", markersize=10)
plt.fill_between(
    performer_x,
    np.asarray(performer_y_avg) - np.asarray(performer_y_std),
    np.asarray(performer_y_avg) + np.asarray(performer_y_std),
    alpha=0.4
)

# plt.plot(bigbird_x, bigbird_y, color="cyan", label="BigBird")

plt.xlabel("Number of nodes (N)")
plt.ylabel("Inference time per instance (B = 64)")
plt.legend()
plt.grid()
plt.show()