import os 
from surprise import *
from surprise.model_selection import *
import pandas as pd
import time

file_path = os.path.expanduser(r'C:\Users\loren\Documents\DS_2Sem\DMT\DMT_2020__HW_2\Part_1\dataset\ratings.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")


######### GRIDSEARCHCV plus SVD #########

kf = KFold(n_splits=5, random_state= 1)
start_time = time.time()

#grid parameter
grid = {'n_factors':[100,150, 170],'n_epochs':[50, 100, 120], 
                'lr_all':[0.005, 0.002],'reg_all':[0.02, 0.1]}

gs = GridSearchCV(SVD, param_grid= grid, measures= ['rmse'], cv = kf, n_jobs= 6, joblib_verbose = 1000)
gs.fit(data)

end_time = time.time()


print ('TIME: ', end_time- start_time)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

pd.set_option('display.max_columns', None)
results_df = pd.DataFrame.from_dict(gs.cv_results)

print(results_df)