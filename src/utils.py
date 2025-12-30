import numpy as np

def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum()
