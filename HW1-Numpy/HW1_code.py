import numpy as np

def signed_dist(x, th, th0):
    dist = (np.dot(th.T,x)+th0)/length(th)
    return dist

def positive(x, th, th0):
    dist = (np.dot(th.T,x)+th0)/length(th)
    return np.sign(dist)

def score(data, labels, th, th0):
    p=positive2(data,th,th0)
    bools= p==labels
    return np.sum(bools)

def best_separator(data, labels, ths, th0s):
    s=score(data,labels,ths,th0s)
    index=np.argmax(s)
    return (ths[:,index:index+1],th0s[:,index:index+1])