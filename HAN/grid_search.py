import os 

hidden_sizes=[i for i in range(20,100,10)]
late_incoporation_type=["ReduceOne","MakeEqual"]
combined_layer_size=[25,30,35,40,45]

sarcasm_files = ['../Datasets/Sarcasm/sarcasm_dataset.pkl', '../Datasets/Sarcasm/sarcasm_dataset_9.pkl', '../Datasets/Sarcasm/sarcasm_dataset_22.pkl']
sentiment_files = ['../Datasets/Sentiment/data_frame.pkl', '../Datasets/Sentiment/data_frame_9.pkl', '../Datasets/Sentiment/data_frame_22.pkl']
hate_files = ['../Datasets/Hate/hate_dataset.pkl', '../Datasets/Hate/hate_dataset_9.pkl', '../Datasets/Hate/hate_dataset_22.pkl']
humour_files = ['../Datasets/Humour/humour_dataset.pkl', '../Datasets/Humour/humour_dataset_9.pkl', '../Datasets/Humour/humour_dataset_22.pkl']

files = sarcasm_files
embedding_file = '../Datasets/embedding_50d.pkl'
num_outputs = 2

for df in files[1:]:
    for li in late_incoporation_type:
        for cs in combined_layer_size:
            for hs in hidden_sizes:
                os.system(f'python HAN_cross_val.py -df {df} -ef {embedding_file} -hs {hs} -cs {cs} -li {li} -us True -no {num_outputs}')
                exit()

for df in files[:1]:
    for li in late_incoporation_type:
        for cs in combined_layer_size:
            for hs in hidden_sizes:
                os.system(f'python HAN_cross_val.py -df {df} -ef {embedding_file} -hs {hs} -cs {cs} -li {li} -us False -no {num_outputs}')
