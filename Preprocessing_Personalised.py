
# Global Dataframes used for preprocessing 
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

seed = 1

def handling_data_types(usersdf : pd.DataFrame, moviesdf: pd.DataFrame, ratingsdf : pd.DataFrame) -> pd.DataFrame:
    # Converting the columns to category types 

    # This is converting or encoding sex into category and codes for future
    usersdf['sex'] = usersdf['sex'].astype('category').cat.codes
    usersdf["age_group"] = usersdf["age_group"].astype('category').cat.codes
    usersdf["occupation"] = usersdf["occupation"].astype('category').cat.codes
    usersdf["zip_code"] = usersdf["zip_code"].astype('category').cat.codes
    
    # Movies dataframe
    moviesdf['movie_id'] = moviesdf['movie_id'].astype('category').cat.codes

    # Ratings dataframe
    ratingsdf['unix_timestamp'] = pd.to_datetime(ratingsdf['unix_timestamp'])
    return usersdf, moviesdf, ratingsdf


# https://stackoverflow.com/questions/57917936/multilabelbinarizer-gives-individual-characters-instead-of-the-classes
def genre_encoding(moviesdf : pd.DataFrame) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    moviesdf['genres'] = moviesdf['genres'].map(lambda x : x.split("|"))
    moviesdf = moviesdf.join(pd.DataFrame(mlb.fit_transform(moviesdf.pop('genres')),columns=mlb.classes_,index=moviesdf.index))
    moviesdf['year'] = moviesdf['title'].map(lambda x : x[-5:-1])
    return moviesdf
                            

# This function here is to first sort ratings using the timestamp and then group the movie ids and ratings by user ids
# This is to put dataset in form so that it is ready to be made into sequences
def sorting_dataframe(ratingsdf : pd.DataFrame) -> pd.DataFrame :
    # Thsi code has been inspired and adapted from
    # https://keras.io/examples/structured_data/movielens_recommendations_transformers/
    ratings_sorted = ratingsdf.copy()
    ratings_sorted = ratings_sorted.sort_values(by=['unix_timestamp']).groupby('user_id')

    sequences_dict = {
        "user_id": list(ratings_sorted.groups.keys()),
        "movie_ids": list(ratings_sorted.movie_id.apply(list)),
        "ratings": list(ratings_sorted.rating.apply(list)),
        "timestamps": list(ratings_sorted.unix_timestamp.apply(list)),
    }

    grouped_data_by_user = pd.DataFrame.from_dict(sequences_dict)
    return grouped_data_by_user


def create_sequences(values : list, window_size : int, step_size : int) -> list[list]:
    """
    Now, let's split the movie_ids list into a set of sequences of a fixed length. 
    We do the same for the ratings. Set the sequence_length variable to change the 
    length of the input sequence to the model. You can also change the step_size to control the number of 
    sequences to generate for each user.
    """
    sequences = []
    start_index = 0
    while True:
        seq = values[start_index:(start_index + window_size)]
        
        if len(seq) < window_size and len(values[-window_size:]) == window_size:
            sequences.append(values[-window_size:])
            break
        elif len(seq) < window_size:
            break
        
        sequences.append(seq)
        start_index = start_index + step_size

    return sequences


def list_to_seq(input_list : list) -> str:
    input_list = [str(i) for i in input_list]
    str_sequence = ','.join(input_list)
    return str_sequence


def dataframe_with_sequences(ratingsdf: pd.DataFrame, window_size : int, step_size : int) -> pd.DataFrame:
    """
    This creates a list of list of different sequences for the movies rated using their ids and the ratings given to those movies as new ratings
    """
    user_ratings_grouped = sorting_dataframe(ratingsdf)
    user_ratings_grouped['movie_ids'] = user_ratings_grouped['movie_ids'].map(lambda movie_id: create_sequences(movie_id, window_size=window_size, step_size=step_size))
    user_ratings_grouped['ratings'] = user_ratings_grouped['ratings'].map(lambda rating: create_sequences(rating, window_size=window_size, step_size=step_size))
    return user_ratings_grouped


def seperate_sequences(usersdf : pd.DataFrame, dataframe_seq : pd.DataFrame) -> pd.DataFrame:
    """
    This is to seperate the sequences belonging to each user and then merging with the 
    users dataframe to get all the users metadata as well into one
    """

    transformed_df = dataframe_seq.explode(['movie_ids','ratings'])[['user_id','movie_ids','ratings']]
    transformed_df_merged = pd.merge(transformed_df, usersdf, on=['user_id'], how='inner')
    transformed_df_merged.movie_ids = transformed_df_merged.movie_ids.map(list_to_seq)
    transformed_df_merged.ratings = transformed_df_merged.ratings.map(list_to_seq)

    transformed_df_merged.drop(columns=['zip_code'], inplace=True)
    transformed_df_merged.columns = ['user_id','sequence_of_movies_watched', 'sequence_of_ratings_given', 'sex', 'age_group','occupation']
    
    return transformed_df_merged


def train_test_split(processed_df) -> pd.DataFrame:
    """
    This is to get test and train data for training and testing the model
    """
    training_data = processed_df.sample(frac=0.95, random_state=seed)
    testing_data = processed_df.drop(training_data.index)
    return training_data, testing_data


def transformer_preprocessing(users_dataframe : pd.DataFrame, ratings_dataframe, movies_dataframe, window_size : int, step_size : int) -> pd.DataFrame:
    users_dataframe1 = users_dataframe.copy()
    ratings_dataframe1 = ratings_dataframe.copy()
    movies_dataframe1 = movies_dataframe.copy()
    
    users_dataframe1, movies_dataframe1, ratings_dataframe1 = handling_data_types(users_dataframe1, movies_dataframe1, ratings_dataframe1)
    movies_dataframe1 = genre_encoding(movies_dataframe1)
    grouped_dataframe_by_user_with_sequences = dataframe_with_sequences(ratings_dataframe1, window_size, step_size)
    sequenced_dataframe = seperate_sequences(users_dataframe1, grouped_dataframe_by_user_with_sequences)
    training_dataframe, testing_dataframe = train_test_split(sequenced_dataframe)
    return training_dataframe, testing_dataframe


