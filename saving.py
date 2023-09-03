#------------------------------------------This has the personalised model architecture in the file--------------------------------------#

# Code inspiration from https://keras.io/examples/structured_data/movielens_recommendations_transformers/
# Code inspiration from https://github.com/jiwidi/Behavior-Sequence-Transformer-Pytorch
# Paper with orignal architecture https://arxiv.org/pdf/1905.06874.pdf
# Added funcitonality from these implementations by embedding movie genres as well as multiplying this concatenated 
# Feature with the corresponding ratings not done in the links above

# Reaidng necessary packages
import torch
import torch.nn as nn
import math
from Dataloader import MovieLensDataset
import pandas as pd
import Preprocessing_Personalised 
import torch


# Files needed in the embedding of the model
users = pd.read_csv('../data/users.csv')
ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies.csv')


# Paramters used in the model
model_paramters = {'dropout' : 0.2, 'batch_size' : 128, 'inc_movie_genre' : False, 'Window_size' : 8, 'step_size' : 1}


# Preprocessing the scanned file using the imported function from the preprocessing.py file in the same directoty
training_df, testing_df = Preprocessing_Personalised.transformer_preprocessing(users, ratings, movies, window_size=model_paramters['Window_size'], step_size=model_paramters['step_size'])

# This is done to one hot encode the movie genres
movies = Preprocessing_Personalised.genre_encoding(movies)

# Getting a list of the unique genres for the movies neeeded in the embedding layer later
genres = list(movies.columns)[3:]



# Positional Embedding for the movie ids 
class PositionalEmbedding(nn.Module):
    """
    Computes the positional encodng for the embeddings and repeats it for each batch
    """

    def __init__(self, max_vocab_size, model_dimensions):
        super().__init__()

        self.pe = nn.Embedding(max_vocab_size, model_dimensions)

    def forward(self, x):
        batch_size = model_paramters['batch_size']
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)



# Another class for embeddings of the different features
class Embeddings(nn.Module):
    '''
    Computes the embedding for each feature of the movie, user or rating
    '''
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.embedding = nn.Embedding(dim_in, dim_out)

    def forward(self, x):
        return self.embedding(x)



# The MLP model where all the features are gone through to get the predicted ratign 
class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(661,1024,),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.linear(x)



# The model it self
class BST(nn.Module):
    def __init__(self):
        super(BST, self).__init__()


#------------------------------------------User embedddings-------------------------------------------------------------------------------------#
        self.user_id_embedding = Embeddings(int(users.user_id.max())+1, int(math.sqrt(users.user_id.max()))+1)
        self.user_age_group_embedding = Embeddings(users.age_group.nunique(), int(math.sqrt(users.age_group.nunique())))
        self.user_sex_embedding = Embeddings(users.sex.nunique(), int(math.sqrt(users.sex.nunique())))
        self.user_occupation_embedding = Embeddings(users.occupation.nunique(), int(math.sqrt(users.occupation.nunique())))


#---------------------------------------Movie embedddings-------------------------------------------------------------------------------------#
        self.embeddings_movie_id = Embeddings(int(movies.movie_id.max())+1, int(math.sqrt(movies.movie_id.max()))+1)
        self.embeddings_movie_year = Embeddings(movies.year.nunique(), int(math.sqrt(movies.year.nunique())))
        genre_vectors = movies[genres].to_numpy()
        self.embeddings_movie_genre = Embeddings(int(movies.movie_id.max())+1, genre_vectors.shape[1])


#-----------------------------------------Positional Embeddings-------------------------------------------------------------------------------------#
        self.positional_embedding = PositionalEmbedding(8, 9)


#----------------------------------------Transformer Layers------------------------------------------------------------------------------------#
        self.transfomerlayer = nn.TransformerEncoderLayer(72, 3, dropout=model_paramters['dropout'])


#-----------------------------------------MLPS for dimension change and at the end ------------------------------------------------------------------------------------#
        self.linear = Mlp()

        # The second linear layer for reshaping the features
        self.linear2 = nn.Sequential(nn.Linear(81, 63), nn.ReLU())
        

# This function covers all the user feature embeddings
    def user_embeddings(self, user_id, sex, age_group, occupation):

        # Embedding the user id 
        user_id = self.user_id_embedding(user_id)

        # Embedding the user sex
        sex = self.user_sex_embedding(sex)

        # Embedding the age group for user
        age_group = self.user_age_group_embedding(age_group)

        # Embeddign the occupation of the user
        occupation = self.user_occupation_embedding(occupation)

        # Concatinating all across batch size and adding all these together to make an overall features for users
        user_features = torch.cat([user_id, sex, age_group,occupation], 1)


        return user_features


    # Embedding the watch history of the user
    def movie_history_genre_embeddings(self, movie_history, movie_history_ratings):

        # Get embedding for each movie id in the movie history eg movie history = 1,2,3,4,5,6,7
        movie_history_1 = self.embeddings_movie_id(movie_history) # 128, 7, 63

        # Embedding the genres for each movie id in the history
        movie_genre_history = self.embeddings_movie_genre(movie_history) # 128, 7, 18

        # Then concatinating these features to get an overall movie history tensor adding across the features
        movie_history = torch.cat([movie_history_1, movie_genre_history], dim=2) # 128, 7, 81

        # Multiplying the movie history tensor with the associated ratings of the movies made by the user
        movie_history = torch.mul(movie_history, movie_history_ratings.unsqueeze(-1))

        return movie_history

    # Getting the target movie embeddings just like the target movie embeddings
    def target_movie_genre_embeddings(self, target_movie_id):

        # GEtting an embedding for the target movie id first
        target_movie_1 = self.embeddings_movie_id(target_movie_id) # 128,63

        # Embedding the genre for the target movie id
        target_movie_genre = self.embeddings_movie_genre(target_movie_id) # 128,18

        # Concatinating these 2 features together, not multiplying rating as that is hidden from model
        target_movie = torch.cat([target_movie_1, target_movie_genre], dim=1).unsqueeze(1) # 128,81

        return target_movie


    # This is to get the features which are going to be inputted into the transformer
    def transformer_features(self, movie_history_embedding, target_movie_embedding):

        # First concatenate the movie history embeddings and target movie embeddings acorrs batch size
        transfomer_features = torch.cat((movie_history_embedding, target_movie_embedding),dim=1) # 128, 8, 91

        # Then put these transformer features trough a linear layer with a relu funciton to reshape them to righ size 81 -> 63
        transfomer_features = self.linear2(transfomer_features)
        return transfomer_features


    def encode_all_features(self,inputs):
        user_id, occupation, sex, age_group, movie_history, movie_history_ratings, target_movie_id, target_movie_rating = inputs

        #-------------------------------Movie history and genre encodings-----------------------------------------------------------------#

        movie_history = self.movie_history_genre_embeddings(movie_history, movie_history_ratings)

        #-------------------------------Target Movie and genre encodings-----------------------------------------------------------------#
        
        target_movie = self.target_movie_genre_embeddings(target_movie_id)

        #-------------------------------Transformer features (features into transformer layer)-----------------------------------------------#
        
        transfomer_features = self.transformer_features(movie_history, target_movie)

        #-------------------------------User embedddings-------------------------------------------------------------------------------------#
        user_features = self.user_embeddings(user_id, sex, age_group, occupation)

        return transfomer_features, user_features, target_movie_rating.float(), 
    

    # Forward to make predictions of the predicted ratings
    def forward(self, batch):

        # First getting the transformer features, the other features, and also the target movie rating to compare from the batch
        transfomer_features, user_features, target_movie_rating,  = self.encode_all_features(batch)

        # Then embed the position for these embeddings needed in transformers for paralelising
        positional_embedding = self.positional_embedding(transfomer_features)

        # Concatinating these transformer features and positional encodigns togther across the dim 2 as that represnets sequence lenght
        transfomer_features = torch.cat([transfomer_features, positional_embedding], dim=2) # 128,8, 72

        # The output from the transformer layer
        transformer_output = self.transfomerlayer(transfomer_features)

        # Then flatten the output for the ratings
        transformer_output = torch.flatten(transformer_output,start_dim=1)

        # Concatenate theis output with the user features
        features = torch.cat([transformer_output,user_features],dim=1)

        # Put it through the MLP as shown above and get the output predicted ratign
        output = self.linear(features)

        # Get the predicted and target movie rating 
        return output, target_movie_rating
        

# Setting up the datsaets for inputs into the model itsef
def setup_datasets(train : bool = True): 

    # For trainng dataset loader the csv path is defined already
    if train:
        return MovieLensDataset(csv_file_sequences="data/train_data.csv")

    # For test dataset loader
    else:
        return MovieLensDataset(csv_file_sequences='data/test_data.csv')


# Just geting the data loaded as well into the dataloader as well
def train_dataloader():
    return torch.utils.data.DataLoader(setup_datasets(train=True), batch_size=128)

# Test data being loaded as well
def test_dataloader():
    return torch.utils.data.DataLoader(setup_datasets(train=False), batch_size=128)

