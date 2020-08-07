from scipy.stats import mannwhitneyu as mwu
import numpy as np
import pandas as pd
import pickle
import scipy.stats
import sklearn.feature_selection

folder = 'Datasets'
subfolders = ['Hate', 'Humour', 'Sarcasm', 'Sentiment']
filenames = ['data_frame_22.pkl']

for subfolder in subfolders:
    for filename in filenames:
        path = './'+folder + '/' + subfolder +'/' + filename
        with open(path, 'rb') as f:
            data = pickle.load(f)
        signals = np.array(data['signal'])

        blank = np.zeros((len(signals),len(signals[0])))
        for i in range(len(signals)):
            blank[i] = np.asarray(signals[i])
        signals = blank

        X = pd.DataFrame(signals)
        y = data['sentiment'].values

        feature_list = ['lang_entropy', 'p_switch', 'burst', 'span_entropy', 'memory','v','f1','f2','mean_hi','stddev_hi','mean_en','stddev_en', 'span_mean', 'span_std', 'hi_span_mean', 'hi_span_std', 'en_span_mean', 'en_span_std', 'l2_mean', 'l2_std', 'cos_mean', 'cos_std']
        feature_dic = {v:k for v,k in enumerate(feature_list)}
        X.rename(columns = feature_dic, inplace = True) 

        correlation_filename = path.replace('Datasets','Correlation').replace('.pkl','.csv')
        f = open(correlation_filename,'w')
        f.write('Feature,Chi-Square,P-Val\n')
        for feature in feature_list:
            x = np.array(X[feature].values)
            t = x.min()
            if t < 0:
                x = x-t
            stats = sklearn.feature_selection.chi2(x.reshape(-1,1),y.reshape(-1,1))
            chi, p = stats[0][0],stats[1][0]
            print(type(chi))
            f.write(feature+','+str(chi)+','+str(p)+'\n')