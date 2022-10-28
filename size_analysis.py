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
    2: 0.01990032196044922,
    3: 0.021232128143310547,
    4: 0.020940780639648438,
    5: 0.021483898162841797,
    6: 0.02102804183959961,
    7: 0.020621538162231445,
    8: 0.02233576774597168,
    9: 0.020986318588256836,
    10: 0.025046110153198242,
    11: 0.021068334579467773,
    12: 0.021134138107299805,
    13: 0.021242618560791016,
    # 14: 0.8922488689422607,
    15: 0.021107196807861328,
    16: 0.021152734756469727,
    17: 0.022296667098999023,
    18: 0.02214670181274414,
    20: 0.021591901779174805
}

transformer_inferences2 = {
    2: 0.020361900329589844,
    3: 0.021975994110107422,
    4: 0.021512508392333984,
    5: 0.021991729736328125,
    6: 0.02089524269104004,
    7: 0.02195143699645996,
    8: 0.023056745529174805,
    9: 0.021541118621826172,
    10: 0.021472454071044922,
    11: 0.021738767623901367,
    12: 0.02152395248413086,
    13: 0.02129054069519043,
    # 14: 0.8951284885406494,
    15: 0.021725893020629883,
    16: 0.021535634994506836,
    17: 0.022629499435424805,
    18: 0.022241592407226562,
    20: 0.02187347412109375
}

transformer_inferences3 = {
    2: 0.02039027214050293,
    3: 0.02148580551147461,
    4: 0.021226882934570312,
    5: 0.021997451782226562,
    6: 0.021564722061157227,
    7: 0.02158212661743164,
    8: 0.022815227508544922,
    9: 0.02147674560546875,
    10: 0.025614261627197266,
    11: 0.021026134490966797,
    12: 0.021547317504882812,
    13: 0.021865129470825195,
    # 14: 0.8848366737365723,
    15: 0.021604061126708984,
    16: 0.021808147430419922,
    17: 0.0225067138671875,
    18: 0.02237248420715332,
    20: 0.02210855484008789
}

performer_inferences1 = {
    2: 0.028009653091430664,
    3: 0.029083967208862305,
    4: 0.028905630111694336,
    5: 0.02969193458557129,
    6: 0.02955174446105957,
    7: 0.03105306625366211,
    8: 0.03377199172973633,
    9: 0.033089399337768555,
    10: 0.03399324417114258,
    11: 0.03506922721862793,
    12: 0.03687572479248047,
    13: 0.03708958625793457,
    # 14: 0.9037270545959473,
    15: 0.04090547561645508,
    16: 0.04536890983581543,
    17: 0.03968667984008789,
    18: 0.04046034812927246,
    20: 0.03919363021850586
}

performer_inferences2 = {
    2: 0.02798318862915039,
    3: 0.02927541732788086,
    4: 0.028989076614379883,
    5: 0.029680490493774414,
    6: 0.029759883880615234,
    7: 0.031136274337768555,
    8: 0.033904075622558594,
    9: 0.03327298164367676,
    10: 0.033904075622558594,
    11: 0.03542613983154297,
    12: 0.037261962890625,
    13: 0.037964582443237305,
    # 14: 0.9028422832489014,
    15: 0.043829917907714844,
    16: 0.04152798652648926,
    17: 0.03998136520385742,
    18: 0.04032588005065918,
    20: 0.03904366493225098
}

performer_inferences3 = {
    2: 0.027588367462158203,
    3: 0.02919149398803711,
    4: 0.02913951873779297,
    5: 0.02989816665649414,
    6: 0.029741525650024414,
    7: 0.03501105308532715,
    8: 0.03351616859436035,
    9: 0.03313088417053223,
    10: 0.033594369888305664,
    11: 0.03501081466674805,
    12: 0.03538632392883301,
    13: 0.036580801010131836,
    # 14: 0.9071159362792969,
    15: 0.04115414619445801,
    16: 0.04143977165222168,
    17: 0.03975653648376465,
    18: 0.04023098945617676,
    20: 0.039048194885253906
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
    10: 0.05961275100708008,
    11: 0.058423519134521484,
    12: 0.05726432800292969,
    13: 0.0689239501953125,
    14: 0.06479573249816895,
    15: 0.06435894966125488,
    16: 0.06703448295593262,
    17: 0.06704902648925781,
    18: 0.9388813972473145,
    19: 0.06779026985168457,
    20: 0.0677497386932373,
    21: 0.06758594512939453,
    22: 0.06838726997375488,
    23: 0.06885814666748047,
    24: 0.07159638404846191,
    25: 0.07726263999938965,
    26: 0.07731270790100098,
    27: 0.07663154602050781,
    28: 0.08025598526000977,
    29: 0.08089685440063477,
    30: 0.08426046371459961,
    31: 0.08324766159057617,
    32: 0.08376073837280273,
    33: 0.08277153968811035,
    34: 0.07505083084106445,
    35: 0.07589983940124512,
    36: 0.07523608207702637,
    37: 0.07657980918884277,
    38: 0.07659268379211426,
    39: 0.07617759704589844,
    40: 0.07806801795959473,
    41: 0.08188295364379883,
    42: 0.07755661010742188,
    43: 0.0803828239440918,
    44: 0.07973027229309082,
    45: 0.07917141914367676,
    46: 0.08440661430358887,
    48: 0.08081984519958496,
    49: 0.08222770690917969,
    50: 0.08336305618286133,
    51: 0.08169245719909668
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