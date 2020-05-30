import sys
import numpy as np
from collections import defaultdict
import pickle as pkl

import pymc3 as pm
import theano
import theano.tensor as tt
#theano.config.gcc.cxxflags = "-fbracket-depth=10000"
from generate_data import generate_data,generate_data_holdout


def GEM(beta):
    """griffiths-engen-mccloskey distribution"""
    pi = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return (beta * pi)


def stickbreak_prior(name,a,shape):
    """truncated stick-breaking construction"""
    gamma = pm.Gamma('gamma_{}'.format(name),1.,1.)
    delta = pm.Gamma('delta_{}'.format(name),1.,a)
    beta_prime = tt.stack([pm.Beta('beta_prime_{}_{}'.format(name,k),1.,gamma) for k in range(shape)])
    beta = GEM(beta_prime)
    return(beta*delta)


def loglik(theta,phi):
    def llik(featvar_id):
        lliks = pm.math.logsumexp(tt.log(theta) + tt.dot(featvar_id,tt.log(phi.T)),axis=1)
        return(tt.sum(lliks))
    return(llik)


def fit_model_LN(N,J,D,R,T,Sigmas,featvar_id,filename,c,normalize,batch=False):
    model = pm.Model()
    with model:
        """hyperparameters"""
        theta_prior = stickbreak_prior('theta',1.,T)
        alpha = .1
        """priors"""
        theta = pm.Dirichlet('theta',theta_prior,shape=T)
        psi = [[pm.MvNormal('psi_{}_{}'.format(t,d),mu=tt.zeros(R[d]),cov=tt.exp(-Sigmas[d]),shape=R[d]) for d in range(D)] for t in range(T)]
        phi = tt.stack([tt.concatenate([pm.Deterministic('phi_{}_{}'.format(t,d),
        tt.nnet.softmax(psi[t][d]))[0]
        for d in range(D)]) 
        for t in range(T)])
        """likelihood"""
        target = pm.DensityDist('target',loglik(theta=theta,phi=phi),observed=dict(featvar_id=featvar_id))
        """fit model"""
        inference = pm.ADVI()
        inference.fit(100000, obj_optimizer=pm.adam(learning_rate=.01,beta1=.8),callbacks=[pm.callbacks.CheckParametersConvergence()])
        trace = inference.approx.sample()
        posterior = {k:trace[k] for k in trace.varnames if not k.endswith('__')}
        posterior['ELBO'] = inference.hist
        if batch == False:
            f = open('posterior_LN_{}_{}_{}.pkl'.format(filename.split('.')[0],c,normalize),'wb')
        else:
            f = open('posterior_LN_{}_{}_{}_holdout_{}.pkl'.format(filename.split('.')[0],c,normalize,batch),'wb')
        pkl.dump(posterior,f)
        f.close()



def fit_model_dir(N,J,D,R,T,featvar_id,filename,c,normalize,batch=False):
    print(normalize)
    print(batch)
    model = pm.Model()
    with model:
        """hyperparameters"""
        theta_prior = stickbreak_prior('theta',1.,T)
        alpha = .1
        """priors"""
        theta = pm.Dirichlet('theta',theta_prior,shape=T)
        phi = tt.stack([tt.concatenate([pm.Dirichlet('phi_{}_{}'.format(t,d),tt.ones(R[d])*alpha,shape=R[d]) for d in range(D)]) for t in range(T)])
        """likelihood"""
        target = pm.DensityDist('target',loglik(theta=theta,phi=phi),observed=dict(featvar_id=featvar_id))
        """fit model"""
        inference = pm.ADVI()
        inference.fit(100000, obj_optimizer=pm.adam(learning_rate=.01,beta1=.8),callbacks=[pm.callbacks.CheckParametersConvergence()])
        trace = inference.approx.sample()
        posterior = {k:trace[k] for k in trace.varnames if not k.endswith('__')}
        posterior['ELBO'] = inference.hist
        if batch == False:
            f = open('posterior_dir_{}_{}_{}.pkl'.format(filename.split('.')[0],c,normalize),'wb')
        else:
            f = open('posterior_dir_{}_{}_{}_holdout_{}.pkl'.format(filename.split('.')[0],c,normalize,batch),'wb')
        pkl.dump(posterior,f)
        f.close()


T = 5
def main():
    batch = ''
    normalize = True
    if len(sys.argv) < 3:
        print('usage: python DPMM_pm.py DATA_SET_NAME prior={dir,LN} chain normalize{TRUE,FALSE} [hold_out batch]')
    else:
        if len(sys.argv) > 3 and sys.argv[4] == 'FALSE':
            normalize = False
        filename = sys.argv[1]
        print('processing_{}'.format(filename))
        if len(sys.argv) == 7 and sys.argv[5] == 'hold_out':
            batch = sys.argv[6]
            N,J,D,R,Y,Sigmas,hold_in,hold_out,ethnic_id = generate_data_holdout(filename,batch,normalize)
        N,J,D,R,Y,Sigmas,featvar_id,ethnic_id = generate_data(filename,normalize)
        print(D,Y)
        chain = sys.argv[3]
        if sys.argv[2] == 'dir':
            if batch != '':
                fit_model_dir(N,J,D,R,T,hold_in,filename,chain,normalize,batch)
            else:
                fit_model_dir(N,J,D,R,T,featvar_id,filename,chain,normalize)
        if sys.argv[2] == 'LN':
            if batch != '':
                fit_model_LN(N,J,D,R,T,Sigmas,hold_in,filename,chain,normalize,batch)
            else:
                fit_model_LN(N,J,D,R,T,Sigmas,featvar_id,filename,chain,normalize)


if __name__=='__main__':
    main()
