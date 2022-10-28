import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
matplotlib.rc('legend', **{"loc": "upper left"})

# ALL TIMINGS ONLY HAVE Atom+LapPE encoders
transformer_inferences1 = {
    4: 0.021355628967285156,
    5: 0.021907806396484375,
    6: 0.022106409072875977,
    7: 0.021216154098510742,
    8: 0.021564722061157227,
    9: 0.02106952667236328,
    10: 0.02292656898498535,
    11: 0.021541595458984375,
    12: 0.021496057510375977,
    13: 0.021747589111328125,
    # 14: 0.8801631927490234,
    15: 0.021670818328857422,
    16: 0.02138662338256836,
    17: 0.02316141128540039,
    18: 0.026433229446411133,
    20: 0.022013187408447266
}

transformer_inferences2 = {
    4: 0.026386022567749023,
    5: 0.02672100067138672,
    6: 0.026973485946655273,
    7: 0.026386737823486328,
    8: 0.0261077880859375,
    9: 0.02667236328125,
    10: 0.027825355529785156,
    11: 0.026197433471679688,
    12: 0.026760101318359375,
    13: 0.02687835693359375,
    # 14: 0.8998994827270508,
    15: 0.02651834487915039,
    16: 0.030497312545776367,
    17: 0.027522802352905273,
    18: 0.027589082717895508,
    20: 0.02693796157836914
}

transformer_inferences3 = {
    4: 0.021655797958374023,
    5: 0.022098779678344727,
    6: 0.022292613983154297,
    7: 0.025502681732177734,
    8: 0.021599292755126953,
    9: 0.021866798400878906,
    10: 0.02338695526123047,
    11: 0.022010326385498047,
    12: 0.021478891372680664,
    13: 0.021742582321166992,
    # 14: 0.8866324424743652,
    15: 0.021630048751831055,
    16: 0.021963119506835938,
    17: 0.022993803024291992,
    18: 0.022691726684570312,
    20: 0.022133588790893555
}

performer_inferences1 = {
    4: 0.029345989227294922,
    5: 0.029874324798583984,
    6: 0.030661344528198242,
    7: 0.03141021728515625,
    8: 0.031859397888183594,
    9: 0.0374445915222168,
    10: 0.03543210029602051,
    11: 0.0356137752532959,
    12: 0.03626608848571777,
    13: 0.03752613067626953,
    # 14: 0.9029102325439453,
    15: 0.042160987854003906,
    16: 0.04233837127685547,
    17: 0.04006624221801758,
    18: 0.03880620002746582,
    20: 0.039351463317871094
}

performer_inferences2 = {
    4: 0.02945089340209961,
    5: 0.030077695846557617,
    6: 0.0306246280670166,
    7: 0.031328678131103516,
    8: 0.03183913230895996,
    9: 0.033472299575805664,
    10: 0.036631107330322266,
    11: 0.035762786865234375,
    12: 0.037563323974609375,
    13: 0.04258990287780762,
    # 14: 0.897571325302124,
    15: 0.04223299026489258,
    16: 0.04208230972290039,
    17: 0.03984427452087402,
    18: 0.03917527198791504,
    20: 0.03944277763366699
}

performer_inferences3 = {
    4: 0.029726266860961914,
    5: 0.03049445152282715,
    6: 0.030924558639526367,
    7: 0.0322566032409668,
    8: 0.0323634147644043,
    9: 0.03413248062133789,
    10: 0.03695988655090332,
    11: 0.03614449501037598,
    12: 0.03694009780883789,
    13: 0.03929710388183594,
    # 14: 0.8851344585418701,
    15: 0.04183053970336914,
    16: 0.04509377479553223,
    17: 0.04058504104614258,
    18: 0.039259910583496094,
    20: 0.039804935455322266
}

bigbird_inferences1 = {
    10: 0.07501959800720215,
    11: 0.07792925834655762,
    12: 0.07290983200073242,
    13: 0.08179259300231934,
    # 14: 0.9139096736907959,
    15: 0.08077669143676758,
    16: 0.0831298828125,
    17: 0.08383488655090332,
    18: 0.08241510391235352,
    20: 0.08607006072998047
}

bigbird_inferences2 = {
    10: 0.06212186813354492,
    11: 0.05645012855529785,
    12: 0.05511283874511719,
    13: 0.061762332916259766,
    # 14: 0.9181299209594727,
    15: 0.06124711036682129,
    16: 0.06340599060058594,
    17: 0.06479549407958984,
    18: 0.06270122528076172,
    20: 0.0657341480255127
}

bigbird_inferences3 = {
    10: 0.05696845054626465,
    11: 0.05574750900268555,
    12: 0.054698944091796875,
    13: 0.06125926971435547,
    # 14: 0.9163124561309814,
    15: 0.06121683120727539,
    16: 0.06658720970153809,
    17: 0.06390833854675293,
    18: 0.06207466125488281,
    20: 0.06490683555603027
}

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

bigbird_x = list(bigbird_inferences1.keys())
bigbird_y1 = list(bigbird_inferences1.values())
bigbird_y2 = list(bigbird_inferences2.values())
bigbird_y3 = list(bigbird_inferences3.values()) 

bigbird_y = []
for y1, y2, y3 in zip(bigbird_y1, bigbird_y2, bigbird_y3):
    bigbird_y.append([y1, y2, y3])
bigbird_y = np.asarray(bigbird_y)
bigbird_y_avg = bigbird_y.mean(axis=1)
bigbird_y_std = bigbird_y.std(axis=1)

xint = range(min(transformer_x), math.ceil(max(transformer_x))+1)
matplotlib.pyplot.xticks(xint)

plt.plot(transformer_x, transformer_y_avg, color="green", label="Transformer", markersize=10)
plt.fill_between(
    transformer_x,
    np.asarray(transformer_y_avg) - np.asarray(transformer_y_std), 
    np.asarray(transformer_y_avg) + np.asarray(transformer_y_std),
    alpha=0.3,
    color="green"
)

plt.plot(performer_x, performer_y_avg, color="red", label="Performer", markersize=10)
plt.fill_between(
    performer_x,
    np.asarray(performer_y_avg) - np.asarray(performer_y_std),
    np.asarray(performer_y_avg) + np.asarray(performer_y_std),
    alpha=0.3,
    color="red"
)

plt.plot(bigbird_x, bigbird_y_avg, color="blue", label="BigBird")
plt.fill_between(
    bigbird_x,
    np.asarray(bigbird_y_avg) - np.asarray(bigbird_y_std),
    np.asarray(bigbird_y_avg) + np.asarray(bigbird_y_std),
    alpha=0.3,
    color="cyan"
)

plt.xlabel("Number of nodes (N)")
plt.ylabel("Inference time per instance (B = 64)")
plt.legend()
plt.grid()
plt.show()