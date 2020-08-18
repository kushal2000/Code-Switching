import os 
import sys

hidden_sizes = [22, 50, 100, 200]

alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
beta = [1.0, 2.0, 5.0, 10.0]
# learning_rates = [1e-3, 1e-4, 2e-5]
files = ['Datasets/HUMOUR.pkl']

for fil in files:
    f = open(fil.replace('pkl', 'csv'), 'w')
    f.write(f'Hidden Size,Alpha,Beta,Macro,Micro,Acc\n')
    f.close()
    for hs in hidden_sizes:
        for a in alpha:
            for b in beta:
                os.system(f'python fusion_net_train.py -f {fil} --feature_dim 22 -lr {2e-4} -hs {hs} --learning_rate 1e-4 --num_labels 2 --alpha {a} --beta {b} -bs 32')
            