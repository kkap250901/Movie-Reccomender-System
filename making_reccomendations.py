import itertools
from Dataloader import MovieLensDataset
import torch
import pandas as pd
from Model_architecture import BST

# Selecting the device to ensure the tensors stored in the right manner
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model loaded and shifted to device
# model = BST().to(device)

# Model loaded with pretrained parameters
# model.load_state_dict(torch.load('./new_bst'))

# Reading the dataframes necessary for the rest of the file
# users = pd.read_csv('../data/users.csv')
# ratings = pd.read_csv('../data/ratings.csv')
# movies = pd.read_csv('../data/movies.csv')


# This dataframe is for a list of list of sequences for each user necessary eg User 1 : [[1,2,3,4,5,][9,10,21,24,....]]
# This is necessary to get the reccomendations which have not been seen
# For the sake of efficiency data stored in a pickle file to preserve dtype
# ratings_data = pd.read_pickle('./data/For_reccomendations/Rating_sequences_list.pkl')

# This is the exploded view for the ratings per user dataframe generated during preprocessing but very useful during 
# Reccomendatios 
# ratings_data_transformed = pd.read_csv('./data/For_reccomendations/Sequences_seperated_for_user.csv')


##------------------------------------------How the reccomendations are made in this model taking a random user----------------------------#

# WE essentially first identify which movies the user not previously rated 
# For example USer 1 has not rated movies 10,20,30
# WE then take the last 7 movies the user has watched 
# For example taking user 1 last 7 movies [1,2,3,4,5,6,7]
# So we make this into a permutation with the movies user has not watched 
# seq 1 --> [1,2,3,4,5,6,7] + [10] input in model to get prediction of movie id 10 which user has not watched
# seq 2 --> [1,2,3,4,5,6,7] + [20] input in model to get prediction of movie id 20 which user has not watched
# seq 3 --> [1,2,3,4,5,6,7] + [30] input in model to get prediction of movie id 30 which user has not watched
# We do this for all movies user has not watched and we get a rating prediction for each of those movies for our model
# Then we sort them out and return the top 10 ratings as predicted by our model and return them as a prediction

##------------------------------------------How the reccomendations are made in this model taking a random user----------------------------#



def making_reccomendations_not_seen(user_number : str):
    '''
    Finding the movies the user has not rated previously aand outputting a dataframe of sequences wiht the lasy 7 movies the user 
    has watched with the 8th one being the movie has not rated previous;y
    This then loaded using the custom dataloader and then inputted into the model to get a prediction
    '''
    ratings = pd.read_csv('../data/ratings.csv')
    ratings_data_transformed = pd.read_csv('./data/For_reccomendations/Sequences_seperated_for_user.csv')
    ratings_data = pd.read_pickle('./data/For_reccomendations/Rating_sequences_list.pkl')
    
    # Just copy the list of movies rated copied to not change the original
    ratings_data1 = ratings_data.copy()

    # From this get the total list of all sequences watched by the user
    ratings_data1['movie_ids'] = ratings_data1['movie_ids'].map(lambda x : list((itertools.chain(*x))))


    # Copy original ratings dataframe as well to ensure no change to oringila
    ratingsg = ratings.copy()

    # Converting dtype of the movie_id column to make sure comparisons can be made correctly
    ratingsg.movie_id = ratingsg.movie_id.astype(str)

    # Getting a set of all movie ids the user has rated previously
    all_rated = set(ratingsg.movie_id.unique())

    # Making a new column which has a set of movies the user has not watched previouslt
    ratings_data1['not_watched'] = ratings_data1['movie_ids'].map(lambda x : all_rated.difference(set(x)))

    # Again copying here to ensure no change to gloabal dataframe
    ratings_data_transformed1 = ratings_data_transformed.copy()

    # This dataframe has a different row for each rating sequence of user
    # The user id is converted to string type format for comaprisons
    ratings_data_transformed1.user_id = ratings_data_transformed1.user_id.astype(str)

    # We only keep the latest sequence for the user and make permutations with latest 7 movies the user has watched 
    # and the 8th movie would be one of the movies the user has not watched
    ratings_data_transformed1 = ratings_data_transformed1.drop_duplicates(subset=['user_id'], keep='last')

    # Getting a list not watched movies for the user
    l1 = list(ratings_data1[ratings_data1['user_id'] == user_number].iloc[0].not_watched)

    # Getting the last sequence of wacthed movies for the user
    l2 = eval(ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].sequence_movie_ids)

    # Making a new list to concatenate the new sequences to 
    new_l2 = []

    # Iterate through the last 7 movies watched by the user
    for i in range(len(l1)):

        # This is the target movie we want to get a predicted rating for
        newnggg = l1[i]

        # This basically merged the last 7 movies watched by the user and the target movie (oen of the movies not watched)
        newwww = list(l2[:7]) + [newnggg]

        # Then create a sequences with last 7 movies watched by user + the movie not watched to get prediction
        new_l2.append(newwww)


    # Also this is to convert each movie inside into a string to keep it consistent with the dtypes
    for i in range(len(new_l2)):
        for j in range(len(new_l2[i])):
            new_l2[i][j] = str(new_l2[i][j])


    # This is the new dictionary which will be used to create a dataframe
    new_dfff = {
        # The user id is static
        "user_id" : user_number,
        
        # The new sequence of movie ids as suggested by the new multi dimensional list made 
        "sequence_movie_ids" : new_l2,

        # Sequence of ratings as mentioned the 8th rating here is irrelvant not taken into accoutn as we dont have the ratings for these movies
        "sequence_ratings" : ratings_data_transformed1.iloc[0].sequence_ratings,

        # The sex of user another static feature
        "sex" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].sex,

        # The age group of user another static feature
        "age_group" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].age_group,

        # The occupation of user another static feature
        "occupation" : ratings_data_transformed1[ratings_data_transformed1['user_id'] == user_number].iloc[0].occupation
    }

    # Converting into dataframe from the dicitonary above
    nnnnn =pd.DataFrame.from_dict(new_dfff)

    # Then also making sure the sequence of movie ids aree in the right format for the dataloader as designed before
    nnnnn.sequence_movie_ids = nnnnn.sequence_movie_ids.apply(lambda x : ','.join(x))

    # Returning the dataframe
    return nnnnn


def loading_data(csv):
    """
    This is used to laod the dataste with batch size 128 
    To load this data in a way to make predictions from the model 
    """
    data_batch = MovieLensDataset(csv_file_sequences=csv)
    return torch.utils.data.DataLoader(data_batch, batch_size=128)



def top_10_recs(not_watched):
    """
    This gives the top 10 reccomendations after getitng a predicted rating for each movie the user has not wacthed
    """

    model = BST().to(device)
    model.load_state_dict(torch.load('./new_bst'))

    # Initialising 2 lists the target stores the target movie id essentially the movie we are making a prediction for
    overall_target = []

    # Output is the score we are making a predicted rating for the movies the usre has nto watched 
    overall_output  = []

    # With no grad to save memory we make 
    with torch.no_grad():
        
        # Iterating throught the batch 
        for i, data in enumerate(not_watched):

            # Getting the target movie id
            target_movie_id_nov = data[6]

            # Appending to the list of movie ids
            overall_target += target_movie_id_nov.tolist()

            # Predicted ratings
            output_v, target_v = model(data)

            # Getting  the list of predicted ratings
            overall_output += output_v.tolist()

    # Makign a dictionary parings of these overall targets to predicted raitns
    overall_output =list((itertools.chain(*overall_output)))
    pairings = {overall_target[i] : overall_output[i] for i in range(len(overall_target))}

    # Sorting the dictionary based on the predicted ratings
    pairings = dict(sorted(pairings.items(), key=lambda item: item[1], reverse=True))
    top_10 = list(dict(list(pairings.items())[:10]).keys())
    return pairings


def make_10_reccomendations(user_number : str):

    # Getting the dataframe of sequences
    df_with_sequences_of_hidden_movies = making_reccomendations_not_seen(user_number)

    # Converting to csv 
    df_with_sequences_of_hidden_movies.to_csv('./data/For_reccomendations/temp_dataset_for_given_user.csv')

    # csv loaded to get data in batches for reccoemndations
    data_loaded = loading_data(csv= './data/For_reccomendations/temp_dataset_for_given_user.csv')

    # then get the reccomendations as the dicitronary
    top_10_reccomendations = top_10_recs(data_loaded)
    return top_10_reccomendations





