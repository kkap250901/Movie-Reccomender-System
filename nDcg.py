import pandas as pd
import numpy as np
import random
import torch
from Dataloader import MovieLensDataset
import itertools
from Model_architecture import BST 
import math

#----------------------------------------------------Calculating nDCG for the personalised model---------------------------------------------#

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BST().to(device)
model.load_state_dict(torch.load('./new_bst'))
ratings = pd.read_csv('./data/ratings.csv')
ratings_data = pd.read_pickle('./data/For_reccomendations/Rating_sequences_list.pkl')
ratings_data_transformed = pd.read_csv('./data/For_reccomendations/Sequences_seperated_for_user.csv')
random_users_chosen = random.sample(range(1, 6040), 10)



#Getting Ground truth ratings
def ground_truth_ratings():
    """
    This is to get the dictionary to get the ground truth ratingss from the test dataste
    """
    ratings_dic = {}
    for user in random_users_chosen:
        filtered_ratings = ratings[ratings.user_id == user]
        potential_movies = random.sample(list(filtered_ratings.movie_id.unique()), 10)
        movies_idss = []
        ratingggg = []
        for movie_id in potential_movies:
            filtered_ratings_movies = filtered_ratings[filtered_ratings.movie_id == movie_id]
            rating = filtered_ratings_movies.rating.iloc[0]
            movies_idss.append(movie_id)
            ratingggg.append(rating)
        ratings_dic.update({str(user) : [movies_idss, ratingggg]})
    return ratings_dic


def making_reccomendations_not_seen(dictionary_to_predict):
    overall_dfs = {}
    for key in dictionary_to_predict.keys():
        user_number = key
        # Just copy the list of movies rated
        ratings_data1 = ratings_data.copy()
        ratings_data1.user_id = ratings_data1.user_id.astype(str)

        # Copy raty
        ratingsg = ratings.copy()
        ratingsg.movie_id = ratingsg.movie_id.astype(str)

        # Last sequence for the user given
        ratings_data_transformed1 = ratings_data_transformed.copy()
        ratings_data_transformed1 = ratings_data_transformed1.drop_duplicates(subset=['user_id'], keep='last')
        ratings_data_transformed1.user_id = ratings_data_transformed1.user_id.astype(str)

        # Getting a list not watched movies
        l1 = dictionary_to_predict[key][0]
        l2 = eval(ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].sequence_movie_ids)

        new_l2 = []

        for i in range(len(l1)):
            newnggg = l1[i]
            newwww = list(l2[:7]) + [newnggg]
            new_l2.append(newwww)
            
        for i in range(len(new_l2)):
            for j in range(len(new_l2[i])):
                new_l2[i][j] = str(new_l2[i][j])

        new_dfff = {
            "user_id" : user_number,
            "sequence_movie_ids" : new_l2,
            "sequence_ratings" : ratings_data_transformed1.iloc[0].sequence_ratings,
            "sex" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].sex,
            "age_group" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].age_group,
            "occupation" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].occupation
        }

        nnnnn =pd.DataFrame.from_dict(new_dfff)
        nnnnn.sequence_movie_ids = nnnnn.sequence_movie_ids.apply(lambda x : ','.join(x))
        overall_dfs.update({str(user_number) : nnnnn})

    return overall_dfs

def loading_data(csv):
    data_batch = MovieLensDataset(csv_file_sequences=csv)
    return torch.utils.data.DataLoader(data_batch, batch_size=128)


def top_10_recs(not_watched):
    # maeggg = torchmetrics.MeanAbsoluteError().to('cuda')
    overall_target = []
    overall_output  = []
    with torch.no_grad():
        # loss_test = 0
        for i, data in enumerate(not_watched):
            target_movie_id_nov = data[6]
            overall_target += target_movie_id_nov.tolist()
            output_v, target_v = model(data)
            overall_output += output_v.tolist()
    overall_output =list((itertools.chain(*overall_output)))
    pairings = {overall_target[i] : overall_output[i] for i in range(len(overall_target))}
    pairings = dict(sorted(pairings.items(), key=lambda item: item[1], reverse=True))
    # top_10_ratings =  list(dict(list(pairings.items())[:10]).values())
    return pairings


def make_10_reccomendations(dictionary_to_predict : dict):
    overall_preds = {}
    for key in dictionary_to_predict.keys():
        df_with_sequences_of_hidden_movies = dictionary_to_predict[key]
        df_with_sequences_of_hidden_movies.to_csv('./data/For_ndcg/temp_dataset_for_given_user.csv')
        data_loaded = loading_data(csv= './data/For_ndcg/temp_dataset_for_given_user.csv')
        top_10_reccomendations_id = top_10_recs(data_loaded)
        overall_preds.update({key : top_10_reccomendations_id})
    return overall_preds


def ndcg_calc(rating_ground : list, ratings_new : dict):
    dcg = 0
    idcg = 0
    for i in range(len(rating_ground)):
        j = i + 1
        dcg += ratings_new[rating_ground[i][0]] / math.log2(j + 1)
        idcg += rating_ground[i][1] / math.log2(j + 1)
    return dcg / idcg


initial = ground_truth_ratings()
dic_of_dfs = making_reccomendations_not_seen(initial)
final_preds = make_10_reccomendations(dic_of_dfs)

dcg_avg = 0
i = 0
for key in final_preds.keys():
    i +=1
    concatenated = sorted(dict(zip(initial[key][0], initial[key][1])).items(), key=lambda x:x[1], reverse=True)
    dcg_avg += ndcg_calc(rating_ground=concatenated, ratings_new=final_preds[key])

print('Average dcg for a random 10 users made on the top 10 predictions: ', dcg_avg / i)