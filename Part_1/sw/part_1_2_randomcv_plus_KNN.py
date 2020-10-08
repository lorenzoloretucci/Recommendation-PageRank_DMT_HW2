import os 
from surprise import *
from surprise.model_selection import *
import pandas as pd
import time
from scipy.stats import randint, uniform
import numpy as np
import random as rd


file_path = os.path.expanduser(r'C:\Users\loren\Documents\DS_2Sem\DMT\DMT_2020__HW_2\Part_1\dataset\ratings.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

start_time = time.time()


# RandomizedSearchCV plus KKNBaseline algo 

kf = KFold(n_splits=5, random_state= 1)


 #Grid parameter                           
param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                              'reg': [1, 2]},
              'k': [i for i in range(1, 50)], 
              'min_k': [a for a in range(1, 21)], 
              'n_factors':[w for w in range(1, 51)],
              'n_epochs':[e for e in range(1, 51)],  
              'lr_all': uniform(0.002, 0.005),
              'reg_all':uniform(0.5, 0.9),
              'sim_options': {'name': ['pearson_baseline', 'cosine'], 
                              'min_support': [1, 5],
                              'user_based': [False, True]}  
              }

gs = RandomizedSearchCV(KNNBaseline, param_distributions= param_grid, n_iter=10, measures= ['rmse'], cv = kf, n_jobs= 6, joblib_verbose = 1000)
gs.fit(data)

end_time = time.time()


print ('TIME: ', end_time- start_time)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

pd.set_option('display.max_columns', None)
results_df = pd.DataFrame.from_dict(gs.cv_results)

print(results_df)


