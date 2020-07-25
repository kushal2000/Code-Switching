import os 

hidden_sizes = [0,50,100,200]
learning_rates = [1e-3, 1e-4, 2e-5]
files = ['Datasets/humour/humour_dataset.pkl', 'Datasets/humour/humour_dataset_22.pkl', 'Datasets/humour/humour_dataset_9.pkl']

for fil in files:
    for lr in learning_rates:
        for hs in hidden_sizes:
            os.system(f'python bert_cross_val.py --lr {lr} --hs {hs} --filename {fil}')