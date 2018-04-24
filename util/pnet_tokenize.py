import re
import numpy as np


def tokenize_net(file):
    f = open(file)
    det = f.read()

    ret = []

    det = det.split("|")

    for l in det:
        ret.append(tokenize_layer(l))

    return ret


def tokenize_layer(l):
    ret = {}
    ins = re.split(r":", re.sub(r"[\n\s]", "", l))
    layerdet = re.split(r"[()]", ins[0])
    ret["ltype"] = layerdet[0]
    if len(layerdet) > 1 and layerdet[1]:
        ret["lspec"] = layerdet[1]

    ret["act"] = ins[1]
    ret["params"] = np.array([i for i in map(float, re.split(r",", ins[2]))])
    ret["dists"] = re.split(r",", ins[3])

    return ret
