import pandas as pd
from making_reccomendations import make_10_reccomendations
rated = pd.read_csv('./data/ratings.csv')

def loading_data(user : int): 
    movies_df = pd.read_csv("./data/movies.csv")
    # user = 1 # -> change to input later

    # Get recommendations for films 
    # dictionary
    model_ratings = make_10_reccomendations(user_number=str(user))
    

    # Map model ratings to dataframe, dropping nan values which are movies user has watched
    movies_df["model_ratings"] = movies_df["movie_id"].map(model_ratings)
    movies_df.dropna(inplace=True)
    
    print(movies_df.head(20))
    movies_df.to_csv(f"./For_per_user_rec/user_id_{user}.csv", index=False)
    return movies_df



two_hundred_random_users = range(1,201)
for user in two_hundred_random_users:
    checking = loading_data(str(user))
    # print(set(checking.keys()).intersection(set(set(rated[(rated.user_id == 2)].movie_id.unique()))))