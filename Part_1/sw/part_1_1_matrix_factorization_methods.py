import os 
from surprise import *
from surprise.model_selection import *

file_path = os.path.expanduser(r'C:\Users\loren\Documents\DS_2Sem\DMT\DMT_2020__HW_2\Part_1\dataset\ratings.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

# Matrix Factorization-based Methods

kf = KFold(n_splits=5, random_state=0)

################ SVD #################

current_algo  = SVD()
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6 )


############### SVDpp ################
current_algo_2  = SVDpp()
cross_validate(current_algo_2, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6 )

############### NMF ###################
current_algo_3  = NMF()
cross_validate(current_algo_3, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6 )
