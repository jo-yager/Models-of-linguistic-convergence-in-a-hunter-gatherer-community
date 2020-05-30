import numpy as np
from collections import defaultdict
from Levenshtein import distance
from scipy.stats import entropy

def generate_data(filename,normalize=True):
    """collect data and variables from specific data set"""
    f = open('datasets/'+filename,'r')
    text = f.read()#.strip()
    f.close()
    text = [l.split('\t') for l in text.split('\n')]
    """generate long data format"""
    long_data = [[text[0][j],text[i][0],text[i][j]] for i in range(1,len(text)) for j in range(1,len(text[i]))]
    """get rid of non-Rual speakers and missing entries"""
    bv_speakers = [l.strip().split('\t') for l in open('datasets/basic_vocabulary_data.csv','r')][0][:-2]
    if normalize == True:
        long_data = [l for l in long_data if l[0] in bv_speakers and l[2] != '?']
    else:
        long_data = [l for l in long_data if 'Banun' not in l[0] and 'Manok' not in l[0] and l[2] != '?']
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
    return N,J,D,R,Y,Sigmas,featvar_id,ethnic_id,feature_types


normalize=False
N,J,D,R,Y,Sigmas,featvar_id,ethnic_id,feature_types = generate_data(filename,normalize)

T = 5
thetas = []
phis = []
for c in range(4):
    f = open('posterior_{}_{}_{}_{}.pkl'.format('dir',filename.split('.')[0],c,normalize),'rb')
    posterior = pkl.load(f)
    f.close()
    thetas.append(posterior['theta'])
    phis.append(np.stack([np.concatenate([posterior['phi_{}_{}'.format(t,d)] for d in range(D)],axis=1) for t in range(T)],axis=1))
    f = open('posterior_{}_{}_{}_{}.pkl'.format('LN',filename.split('.')[0],c,normalize),'rb')
    posterior = pkl.load(f)
    f.close()
    thetas.append(posterior['theta'])
    phis.append(np.stack([np.concatenate([np.squeeze(posterior['phi_{}_{}'.format(t,d)],1) for d in range(D)],axis=1) for t in range(T)],axis=1))


theta = np.concatenate(thetas,0)
phi = np.concatenate(phis,0)
breaks = list(zip([sum(R[0:j]) for j in range(D)],[sum(R[0:j]) for j in range(1,D+1)]))
#sum(np.tile(np.expand_dims(theta[0],1),[1,R[0]])*phi[0,:,0:2])
#P_ky = {feature_types[i]:np.tile(np.expand_dims(theta[i],1),[1,R[i]])*phi[i,:,breaks[i][0]:breaks[i][1]] i in range(phi.shape[0])}
p_k_y = {feature_types[d]:np.stack([np.tile(np.expand_dims(theta[i],1),[1,R[d]])*phi[i,:,breaks[d][0]:breaks[d][1]] for i in range(phi.shape[0])]) for d in range(D)}
h_k_y = {feature_types[d]:[(entropy(p_k_y[feature_types[d]][i].flatten()) - entropy(np.sum(p_k_y[feature_types[d]][i],1))) for i in range(phi.shape[0])] for d in range(D)}
h_y_k = {feature_types[d]:[(entropy(p_k_y[feature_types[d]][i].flatten()) - entropy(np.sum(p_k_y[feature_types[d]][i],0))) for i in range(phi.shape[0])] for d in range(D)}
h_k_y_ = {k:np.mean(h_k_y[k]) for k in h_y_k.keys()}
h_y_k_ = {k:np.mean(h_y_k[k]) for k in h_k_y.keys()}



f = open('group_predictability.csv','w')
for k in h_k_y_.keys():
  #print(sorted(outcomes.keys())[d],(x[d],y[d]))
  print(';'.join([k,str(h_k_y_[k]),str(h_y_k_[k])]),file=f)


f.close()