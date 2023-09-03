
# from making_reccomendations import make_10_reccomendations
import pandas as pd
import reccomender_system
from Preprocessing_Personalised import genre_encoding
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")
import os
# user_number = input('What user number are you')
# print(make_10_reccomendations(user_number=str(user_number)))


# The aim of this is to tackle the privacy concerns for the user are addreessed here to let the user know what dtaa is being used
INFO_MESSAGE = """Information Page
1. How we use your data: 
    a. We take into account several of your personal sensotive demographic features including: 
    Sex, Location, Age, Occupation


2. Recommendation Systems: 
    a. Our personalised recommender system works by using your past viewage history and your demographic data to recommend you films. We use a state of the art transformer to do this 
    b. Our non-personalised system presents you the most popular films using a weighted-ranking system

3. How to use: 
    a. You select the system you like from the main page. From here you can now filter by genre through the input interface. 
    b. You can also suggest what decade of films you will like to watch specifically. To reset your choices you simply click clear
"""


def loading_data(user, save_file = False): 
    from making_reccomendations import make_10_reccomendations

    movies_df = pd.read_csv("./data/movies.csv")
     # -> change to input later

    # Get recommendations for films 
    # dictionary
    model_ratings = make_10_reccomendations(user_number=str(user))
    

    # Map model ratings to dataframe, dropping nan values which are movies user has watched
    movies_df["model_ratings"] = movies_df["movie_id"].map(model_ratings)
    movies_df.dropna(inplace=True)
    
    # print(movies_df.head(20))
    if save_file: 
        print(f"Saving user {user} Profile, data loaded...")
        movies_df.to_csv(f"./For_per_user_rec/user_id_{user}.csv", index=False)
    else: 
        print(f"{user} Profile loaded but not saved")
    return movies_df


def get_top_model(movies_df, k=5): 
    return movies_df.nlargest(k, "model_ratings")


def list_options(ranked): 
    for i in range(0,len(ranked)): 
        print(f"{i+1}. {ranked.iloc[i].title[:-6]}")



# Loads user data 
# Checks if user profile already created and loads dataframe
# Else creates dataframe by generating predictions from BST

def loading_user_df(user_number : str):
    
    file_path = f"user_id_{user_number}.csv"
    full_file_path = f"./For_per_user_rec/user_id_{user_number}.csv"

    if file_path in os.listdir("For_per_user_rec"):
        return pd.read_csv(full_file_path)


    print(f"User: {user_number} profile not found, making temp new profile...")
    return loading_data(user_number, False)



def setup_nonpers(): 
    pass

# # Global Variables 
# movies_df = pd.read_csv('../data/movies.csv')
# ratings_df = pd.read_csv('../data/ratings.csv')
#!Todo COnsider makign the movvies not in the upper 0.95 quantile ratings go to 0

# Getting the dummies for genres 
def genre_encoding(movies_df : pd.DataFrame) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    movies_df['genres'] = movies_df['genres'].map(lambda x : x.split("|"))
    movies_df = movies_df.join(pd.DataFrame(mlb.fit_transform(movies_df.pop('genres')),columns=mlb.classes_,index=movies_df.index))
    movies_df['year'] = movies_df['title'].map(lambda x : x[-5:-1])
    return movies_df

#---------------------------------------------------- Non personalised Reccomender-------------------------------------------------------------#
def weighted_rating(movies_df : pd.DataFrame) -> pd.Series:

    minimum_vote_requirement = movies_df["number_of_ratings"].quantile(0.92)
    total_mean_rating = movies_df["average_rating"].mean() # Average of Averages 

    first_term = movies_df['number_of_ratings']/(movies_df["number_of_ratings"] + minimum_vote_requirement) * movies_df["average_rating"]
    second_term = minimum_vote_requirement/(movies_df["number_of_ratings"] + minimum_vote_requirement) * total_mean_rating
    
    return first_term + second_term


def preproc_main(movies_df : pd.DataFrame, ratings_df : pd.DataFrame) -> pd.DataFrame:
    number_of_votes = ratings_df.movie_id.value_counts()
    average_rating = ratings_df.groupby("movie_id")["rating"].mean()

    movies_df['number_of_ratings'] = movies_df['movie_id'].map(number_of_votes)
    movies_df["number_of_ratings"].fillna(0, inplace=True)

    movies_df["average_rating"] = movies_df["movie_id"].map(average_rating)
    movies_df["average_rating"].fillna(0, inplace=True)
    
    movies_df["Weighted_rating"] = weighted_rating(movies_df)
    movies_df = genre_encoding(movies_df)
    return movies_df

#---------------------------------------------------- Non personalised Reccomender-------------------------------------------------------------#

if __name__ == "__main__": 
    

    print("Recommender System")
    try:
        while True:

            print("Select either info, pers, nonpers")                
                    
            choice = input("what system would you like? ").lower().strip()


            if choice == "info": 
                print(INFO_MESSAGE)

            elif choice == "pers":
                user = int(input("What user do you want to be? "))
                
                movies_df = loading_user_df(user)
                movies_df = genre_encoding(movies_df)
                genres = list(movies_df.columns)[3:] 
                reccomender_system.main(movies_df, genres, target_column="model_ratings")
            
            elif choice == "nonpers": 
                movies_df = pd.read_csv("./data/movies.csv")
                ratings_df = pd.read_csv("./data/ratings.csv")
                movies_df = preproc_main(movies_df, ratings_df)
                genres = list(movies_df.columns)[3:] 

                # print(movies_df.head())
                reccomender_system.main(movies_df, genres, target_column="Weighted_rating")


                print("nonpers")
            

    
    except KeyboardInterrupt: 
        print("ended")
        