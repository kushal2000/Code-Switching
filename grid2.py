import os 

hidden_sizes = [0,50,100,200]
learning_rates = [0.001, 0.0001]
files = ['Hate_Datasets/hate_dataset_9.pkl', 'Hate_Datasets/hate_dataset_23.pkl', 'Hate_Datasets/hate_dataset.pkl',
        'Sentiment_Datasets/data_frame_22.pkl', 'Sentiment_Datasets/data_frame_9.pkl', 'Sentiment_Datasets/data_frame.pkl'
]

for fil in files:
    for lr in learning_rates:
        for hs in hidden_sizes:
            os.system(f'python training.py {lr} {hs} {fil}')