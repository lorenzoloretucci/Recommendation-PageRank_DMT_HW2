import os 
from surprise import *
from surprise.model_selection import *

file_path = os.path.expanduser(r'C:\Users\loren\Documents\DS_2Sem\DMT\DMT_2020__HW_2\Part_1\dataset\ratings.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

###### Neighborhood Methods ############

MAXIMUM_number_of_neighbors_to_consider = 40  # The MAXIMUM number of neighbors to take into account for aggregation.
min_number_of_neighbors_to_consider = 1  # The minimum number of neighbors to take into account for aggregation.

# It use from all the KNN methods
similarity_options = {
   'name': "cosine", 
   'user_based': False,  
   'min_support': 3, 
}

kf = KFold(n_splits=5, random_state=0)

######### KNNBasic ################
current_algo = KNNBasic(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                       sim_options=similarity_options, verbose=True)


cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6)


########### KNNWithMeans ##############

current_algo_2 = KNNWithMeans(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                       sim_options=similarity_options, verbose=True)

cross_validate(current_algo_2, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6)


############### KNNWithZScore ##############
current_algo_3 =  KNNWithZScore(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                       sim_options=similarity_options, verbose=True)




cross_validate(current_algo_3, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6)


############# KNNBaseline ##################

baseline_predictor_options = {   
  'method': "sgd",  # Optimization method to use.
  'learning_rate': 0.005,  # Learning rate parameter for the SGD optimization method.
  'n_epochs': 50,  # The number of iteration for the SGD optimization method.
  'reg': 0.02,  # The regularization parameter of the cost function that is optimized: a.k.a. LAMBDA.
}

current_algo_4 =  KNNBaseline(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                       sim_options=similarity_options,bsl_options =baseline_predictor_options, verbose=True)


cross_validate(current_algo_4, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs= 6)



