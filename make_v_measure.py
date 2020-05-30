import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

filenames = ['basic_vocabulary_data','caused_motion_data','reciprocal_data','topological_relations_data'][::-1]


text = []


for prior in ['dir','LN']:
    for cond in ['False','True']:
        text_ = []
        for fn in filenames:    
            f = open('v_measure_{}_{}_{}.tex'.format(fn,prior,cond),'r')
            text_.append(float(f.read().strip()))
        text.append(text_)


marks = ['TR','RE','CM','BV']

plt.plot(text[0],c='#1f77b4')
plt.plot(text[1],c='#1f77b4',alpha=.4)
plt.plot(text[2],c='#ff7f0e')
plt.plot(text[3],c='#ff7f0e',alpha=.4)
plt.xticks([0,1,2,3],marks)

tikz_save('V-measures')