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



def get_matching_coefs_sim(data,empirical_data,breaks,J):
    match = defaultdict(int)
    for i in range(J):
        for j in range(i+1,J):
            counter = 0
            for b in breaks:
                if sum(empirical_data[i,b[0]:b[1]]) != 0 and sum(empirical_data[j,b[0]:b[1]]) != 0:
                    counter += 1
                    if np.all(data[i,b[0]:b[1]] == data[j,b[0]:b[1]]):
                        match[(i,j)] += 1
            match[(i,j)] /= counter
    return(list(match.values()))

    

def eval_cluster(filename,J,D,R,T,featvar_id,ethnic_id,prior,normalize):
    thetas = []
    phis = []
    for c in range(4):
        f = open('posterior_{}_{}_{}_{}.pkl'.format(prior,filename.split('.')[0],c,normalize),'rb')
        posterior = pkl.load(f)
        f.close()
        thetas.append(posterior['theta'])
        if prior == 'dir':
            phis.append(np.stack([np.concatenate([posterior['phi_{}_{}'.format(t,d)] for d in range(D)],axis=1) for t in range(T)],axis=1))
        if prior == 'LN':
            phis.append(np.stack([np.concatenate([np.squeeze(posterior['phi_{}_{}'.format(t,d)],1) for d in range(D)],axis=1) for t in range(T)],axis=1))
    theta = np.concatenate(thetas,0)
    phi = np.concatenate(phis,0)
    p_z = np.exp(np.expand_dims(np.log(theta),1) + np.einsum('njy,nty->njt',
                 np.repeat(np.expand_dims(featvar_id,0),phi.shape[0],0),np.log(phi)))
    p_z /= np.sum(p_z,-1,keepdims=True)
    Z = np.stack([[list(np.random.multinomial(1,p_z[c,j])).index(1) for j in range(J)] for c in range(p_z.shape[0])])
    print(Counter([len(set(z)) for z in Z]))
    v_measures = [v_measure_score(z,ethnic_id) for z in Z]
    f = open('v_measure_{}_{}_{}.tex'.format(filename.split('.')[0],prior,normalize),'w')
    print(np.mean(v_measures),file=f)
    f.close()
    """matching PPC"""
    breaks = list(zip([sum(R[0:j]) for j in range(D)],[sum(R[0:j]) for j in range(1,D+1)]))
    match = np.mean(get_matching_coefs(featvar_id,breaks,J))
    sim_match = []
    for i in range(phi.shape[0]):
        phi_i = phi[i][Z[i]]
        sim_data_i = np.stack([np.concatenate([np.random.multinomial(1,phi_i[j,b[0]:b[1]]) for b in breaks]) for j in range(J)])
        sim_match.append(np.mean(get_matching_coefs_sim(sim_data_i,featvar_id,breaks,J)))
    #plt.hist(sim_match,alpha=.6)
    #plt.axvline(np.mean(sim_match),color='red')
    #plt.axvline(match,color='black')
    f = open('sim_match_{}_{}'.format(filename.split('.')[0],prior),'w')
    print(' '.join([str(s) for s in sim_match]),file=f)
    f.close()
    #tikz_save('hist_{}_{}_{}'.format(filename.split('.')[0],prior,normalize))
    #plt.savefig('hist_{}_{}_{}.pdf'.format(filename.split('.')[0],prior,normalize))



T = 5
def main():
    normalize = True
    if len(sys.argv) < 3:
        print('usage: python eval_posterior.py prior DATA_SET_NAME')
    else:
        if len(sys.argv) > 3 and sys.argv[3] == 'FALSE':
            normalize = False
        print(normalize)
        filename = sys.argv[1]
        prior = sys.argv[2]
        print('processing_{}'.format(filename))
        N,J,D,R,Y,Sigmas,featvar_id,ethnic_id = generate_data(filename,normalize)
        eval_cluster(filename,J,D,R,T,featvar_id,ethnic_id,prior,normalize)


if __name__=='__main__':
    main()