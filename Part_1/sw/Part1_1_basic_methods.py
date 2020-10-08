import os 
from surprise import *
from surprise.model_selection import *

file_path = os.path.expanduser(r'C:\Users\loren\Documents\DS_2Sem\DMT\DMT_2020__HW_2\Part_1\dataset\ratings.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

# BASIC METHODS 
kf = KFold(n_splits=5, random_state=0)

########### Normal Predictor Algo #############
current_algo = NormalPredictor() 

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6 )


########### BaselineOnly Algo #############
baseline_predictor_options = {   
  'method': "sgd",  # Optimization method to use.
  'learning_rate': 0.005,  # Learning rate parameter for the SGD optimization method.
  'n_epochs': 50,  # The number of iteration for the SGD optimization method.
  'reg': 0.02,  # The regularization parameter of the cost function that is optimized: a.k.a. LAMBDA.
}
current_algo_2 = BaselineOnly(bsl_options=baseline_predictor_options, verbose=True)

cross_validate(current_algo_2, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6)