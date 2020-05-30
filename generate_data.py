import numpy as np
from collections import defaultdict
from Levenshtein import distance


def generate_data(filename,normalize=True):
    """collect data and variables from specific data set"""
    f = open('datasets/'+filename,'r')
    text = f.read()#.strip()
    f.close()
    text = [l.split('\t') for l in text.split('\n')]
    """generate long data format"""
    long_data = [[text[0][j],text[i][0],text[i][j]] for i in range(1,len(text)) for j in range(1,len(text[i]))]
    """get rid of non-Rual speakers and missing entries"""
    bv_speakers = [l.strip().split('\t') for l in open('datasets/basic_vocabulary_data.csv','r')][0]
    if normalize == True:
        long_data = [l for l in long_data if l[0] in bv_speakers and l[2] != '?']
    else:
        long_data = [l for l in long_data if l[2] != '?']
    """"""
    outcomes = defaultdict(list)
    for l in long_data:
        outcomes[l[1]].append(l[2])
    for key in outcomes.keys():
        outcomes[key] = sorted(set(outcomes[key]))
    outcomes = {key:outcomes[key] for key in outcomes.keys() if len(outcomes[key]) > 1}
    long_data = [l for l in long_data if l[1] in outcomes.keys()]
    speaker_types = sorted(set([l[0] for l in long_data]))
    feature_types = list(outcomes.keys())
    featvar_types = [(k,v) for k in outcomes.keys() for v in outcomes[k]]
    """covariances"""
    Sigmas = [np.array([[distance(u,v)/max([len(u),len(v)]) for u in outcomes[k]] for v in outcomes[k]]) for k in outcomes.keys()]
    """number of data points"""
    N = len(long_data)     #N data points
    J = len(speaker_types) #N spakers
    D = len(feature_types) #N feature types
    R = [len(outcomes[k]) for k in outcomes.keys()]
    Y = sum(R)
    """variable IDs"""
    featvar_id = np.zeros([J,Y])
    for i,l in enumerate(long_data):
        featvar_id[speaker_types.index(l[0]),featvar_types.index((l[1],l[2]))] = 1.
    """speaker designation"""
    ethnic_id = [('jedek' if l.startswith('Jedek') else 'jahai') for l in speaker_types]
    return N,J,D,R,Y,Sigmas,featvar_id,ethnic_id



def generate_data_holdout(filename,k,normalize=True):
    k = int(k)
    """collect data and variables from specific data set"""
    f = open('datasets/'+filename,'r')
    text = f.read()#.strip()
    f.close()
    text = [l.split('\t') for l in text.split('\n')]
    """generate long data format"""
    long_data = [[text[0][j],text[i][0],text[i][j]] for i in range(1,len(text)) for j in range(1,len(text[i]))]
    """get rid of non-Rual speakers and missing entries"""
    bv_speakers = [l.strip().split('\t') for l in open('datasets/basic_vocabulary_data.csv','r')][0]
    if normalize == True:
        long_data = [l for l in long_data if l[0] in bv_speakers and l[2] != '?']
    else:
        long_data = [l for l in long_data if l[2] != '?']
    """"""
    outcomes = defaultdict(list)
    for l in long_data:
        outcomes[l[1]].append(l[2])
    for key in outcomes.keys():
        outcomes[key] = sorted(set(outcomes[key]))
    outcomes = {key:outcomes[key] for key in outcomes.keys() if len(outcomes[key]) > 1}
    long_data = [l for l in long_data if l[1] in outcomes.keys()]
    speaker_types = sorted(set([l[0] for l in long_data]))
    feature_types = list(outcomes.keys())
    featvar_types = [(k,v) for k in outcomes.keys() for v in outcomes[k]]
    """covariances"""
    Sigmas = [np.array([[distance(u,v)/max([len(u),len(v)]) for u in outcomes[k]] for v in outcomes[k]]) for k in outcomes.keys()]
    """number of data points"""
    N = len(long_data)     #N data points
    J = len(speaker_types) #N spakers
    D = len(feature_types) #N feature types
    R = [len(outcomes[k]) for k in outcomes.keys()]
    Y = sum(R)
    """variable IDs"""
    featvar_id = np.zeros([J,Y])
    for i,l in enumerate(long_data):
        featvar_id[speaker_types.index(l[0]),featvar_types.index((l[1],l[2]))] = 1.
    nonzero = list(zip(featvar_id.nonzero()[0],featvar_id.nonzero()[1]))
    K = 4
    np.random.seed(1234)
    inds = np.arange(N)
    np.random.shuffle(inds)
    batches = {k:[nonzero[i] for i in inds[int(np.ceil(N/K)*k):int(np.ceil(N/K)*(k+1))]] for k in range(K)}
    hold_in = np.zeros([J,Y])
    hold_out = np.zeros([J,Y])
    for key in batches.keys():
        for ind in batches[key]:
            if key == k:
                hold_out[ind] += 1
            else:
                hold_in[ind] += 1
    """speaker designation"""
    ethnic_id = [('jedek' if l.startswith('Jedek') else 'jahai') for l in speaker_types]
    return N,J,D,R,Y,Sigmas,hold_in,hold_out,ethnic_id