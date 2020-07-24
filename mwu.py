from scipy.stats import mannwhitneyu as mwu
import numpy as np
import pandas as pd

folder = 'Results'
filenames = ['/humour_results.csv','/sarcasm_results.csv',
    '/hate_results.csv','/sentiment_results.csv']
for filename in filenames:
    print(filename)
    filename = folder + filename
    data = pd.read_csv(filename)
    models = ['BERT', 'BERT 9', 'BERT 22']

    f = open(filename.replace('results', 'mwu'),'w')
    f.write('Model1,Model2,u_score,p_val\n')
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            val1 = data[models[i]].values
            val2 = data[models[j]].values
            u_score, p_val = mwu(val1,val2)
            f.write(f'{models[i]},{models[j]},{u_score},{p_val}\n')
    f.close()