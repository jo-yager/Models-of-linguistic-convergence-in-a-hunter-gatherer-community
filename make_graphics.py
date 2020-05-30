import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save
from collections import defaultdict

filenames = ['basic_vocabulary_data','caused_motion_data','reciprocal_data','topological_relations_data'][::-1]

priors = ['dir','LN']

logliks = defaultdict(list)
for fn in filenames:
    for prior in priors:
        f = open('logliks_{}_{}'.format(prior,fn),'r')
        text = f.read().strip()
        f.close()
        text = [float(s) for s in text.split()]
        logliks[(fn,prior)] = text


for fn in filenames:
    plt.hist(logliks[(fn,'dir')],color='#1f77b4',alpha=.4)
    plt.axvline(np.mean(logliks[(fn,'dir')]),color='#1f77b4')
    plt.hist(logliks[(fn,'LN')],color='#ff7f0e',alpha=.4)
    plt.axvline(np.mean(logliks[(fn,'LN')]),color='#ff7f0e')
    tikz_save('loglik_{}'.format(fn))
    plt.clf()



import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_data
from sklearn.metrics import v_measure_score,homogeneity_score,completeness_score
from scipy.stats import entropy
from collections import Counter,defaultdict
from tikzplotlib import save as tikz_save


def get_matching_coefs(data,breaks,J):
    match = defaultdict(int)
    for i in range(J):
        for j in range(i+1,J):
            counter = 0
            for b in breaks:
                if sum(data[i,b[0]:b[1]]) != 0 and sum(data[j,b[0]:b[1]]) != 0:
                    counter += 1
                    if np.all(data[i,b[0]:b[1]] == data[j,b[0]:b[1]]):
                        match[(i,j)] += 1
            match[(i,j)] /= counter
    return(list(match.values()))


normalize = False
for fn in filenames:
    N,J,D,R,Y,Sigmas,featvar_id,ethnic_id = generate_data(fn+'.csv',normalize)
    breaks = list(zip([sum(R[0:j]) for j in range(D)],[sum(R[0:j]) for j in range(1,D+1)]))
    match = np.mean(get_matching_coefs(featvar_id,breaks,J))
    f = open('sim_match_{}_{}'.format(fn,'dir'))
    dir = f.read().strip()
    f.close()
    dir = [float(s) for s in dir.split()]
    f = open('sim_match_{}_{}'.format(fn,'LN'))
    LN = f.read().strip()
    f.close()
    LN = [float(s) for s in LN.split()]
    plt.hist(dir,color='#1f77b4',alpha=.4)
    plt.axvline(np.mean(dir),color='#1f77b4')
    plt.hist(LN,color='#ff7f0e',alpha=.4)
    plt.axvline(np.mean(LN),color='#ff7f0e')
    plt.axvline(match,color='black')
    tikz_save('PPC_{}'.format(fn))
    plt.clf()
    