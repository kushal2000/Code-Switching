import os 

hidden_sizes = [0,50,100,200]
learning_rates = [0.001, 0.0001]
files = ['Humour_Datasets/humour_dataset_9.pkl', 'Humour_Datasets/humour_dataset_23.pkl', 'Humour_Datasets/humour_dataset.pkl',
        'Sarcasm_Datasets/sarcasm_dataset_9.pkl', 'Sarcasm_Datasets/sarcasm_dataset_23.pkl', 'Sarcasm_Datasets/sarcasm_dataset.pkl'
]

for fil in files:
    for lr in learning_rates:
        for hs in hidden_sizes:
            os.system(f'python bert_cross_val.py {lr} {hs} {fil}')