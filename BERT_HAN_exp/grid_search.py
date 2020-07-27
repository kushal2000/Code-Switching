import os 

# hidden_sizes = [0,50,100,200]
# learning_rates = [1e-3, 1e-4, 2e-5]

hidden_sizes = [200]
learning_rates = [1e-4]

sarcasm_files = ['../Datasets/Sarcasm/sarcasm_dataset.pkl', '../Datasets/Sarcasm/sarcasm_dataset_9.pkl', '../Datasets/Sarcasm/sarcasm_dataset_22.pkl']
sentiment_files = ['../Datasets/Sentiment/data_frame.pkl', '../Datasets/Sentiment/data_frame_9.pkl', '../Datasets/Sentiment/data_frame_22.pkl']
hate_files = ['../Datasets/Hate/hate_dataset.pkl', '../Datasets/Hate/hate_dataset_9.pkl', '../Datasets/Hate/hate_dataset_22.pkl']
humour_files = ['../Datasets/Humour/humour_dataset.pkl', '../Datasets/Humour/humour_dataset_9.pkl', '../Datasets/Humour/humour_dataset_22.pkl']

files = sarcasm_files[-1:]

for fil in files:
    for lr in learning_rates:
        for hs in hidden_sizes:
            os.system(f'python bert_cross_val.py --lr {lr} --hs {hs} --filename {fil}')