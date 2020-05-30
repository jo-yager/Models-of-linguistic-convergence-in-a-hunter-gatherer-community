import sys
import numpy as np
from collections import defaultdict
import pickle as pkl

import pymc3 as pm
import theano
import theano.tensor as tt
#theano.config.gcc.cxxflags = "-fbracket-depth=10000"
from generate_data import generate_data,generate_data_holdout


def loglik(theta,phi,featvar_id):
    lliks = np.log(np.sum(np.exp(np.log(theta) + np.dot(featvar_id,np.log(phi.T))),axis=1))
    return(np.sum(lliks))


def main():
    T = 5
    normalize = False
    filename = sys.argv[1]
    prior = sys.argv[2]
    logliks = []
    for batch in range(4):
        N,J,D,R,Y,Sigmas,hold_in,hold_out,ethnic_id=generate_data_holdout(filename,batch,normalize)
        f = open('posterior_{}_{}_0_{}_holdout_{}.pkl'.format(prior,filename.split('.')[0],normalize,batch),'rb')
        posterior = pkl.load(f)
        f.close()
        theta = posterior['theta']
        if prior == 'dir':
            phi = np.stack([np.concatenate([posterior['phi_{}_{}'.format(t,d)] for d in range(D)],axis=1) for t in range(T)],axis=1)
        if prior == 'LN':
            phi = np.stack([np.concatenate([np.squeeze(posterior['phi_{}_{}'.format(t,d)],1) for d in range(D)],axis=1) for t in range(T)],axis=1)
        logliks += [loglik(theta[i],phi[i],hold_out)/len(hold_out.nonzero()[0]) for i in range(phi.shape[0])]
    f = open('logliks_{}_{}'.format(prior,filename.split('.')[0]),'w')
    print(' '.join([str(s) for s in logliks]),file=f)
    f.close()



if __name__=='__main__':
    main()