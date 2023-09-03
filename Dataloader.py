# Code inspiration from https://keras.io/examples/structured_data/movielens_recommendations_transformers/
# Code inspiration from https://github.com/jiwidi/Behavior-Sequence-Transformer-Pytorch
# Paper with orignal architecture https://arxiv.org/pdf/1905.06874.pdf

# Importing the necessary packages
import torch
import pandas as pd


# Determing the device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Custom Dataloader to load the data in the right format for our model as mentioned above this 
# Dataloader has been inspired from the links mentioned above
class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_sequences):

        # This gets me the rating sequences dataframe after scanning the csv format
        self.rating_sequences = pd.read_csv(csv_file_sequences)

    # Just getting the length of the rating sequences dataframe
    def __len__(self):
        return len(self.rating_sequences)
    
    # Getting the item 
    def __getitem__(self, index):
        
        # This gets the right row
        specific_row = self.rating_sequences.iloc[index]
        #-----------------------------------------------------User Features------------------------------------------------------------#

        # User Features converted to tensor and stored in device
        user_id = torch.tensor(specific_row.user_id).to(device)

        # User occupation converted to tensor and stored in device
        user_occupation = torch.tensor(specific_row.occupation).to(device)

        # User sex converted to tensor and stored in device
        user_sex = torch.tensor(specific_row.sex).to(device)

        # Age group converted to tensor and stored in device
        age_group = torch.tensor(specific_row.age_group).to(device)

        #------------------------------------------------ Movie features----------------------------------------------------------#

        # Movie history the sequence of movies watched represented by list of movie_ids such as [1,2,3,4,5,6]
        # Evaluation to convert the list into a tuple to get access to the index system
        movie_his = eval(specific_row.sequence_movie_ids)[:-1]

        # Convert this list of movie histories into a Longtensor stored in cuda for running model in the future 
        # Need a long tensor as some movie_ids aer quite long so 64 bit integer
        movie_his = torch.LongTensor(movie_his).to(device)
 

        # Getting the history of movie ratings as well
        movie_his_ratings = torch.LongTensor(eval(specific_row.sequence_ratings)[:-1]).to(device)

        # Getting the target movie_id taking the last movie id as the target so first 7 as input sequences are used to predict the 8th movie
        # Converted to tensor and then stored in cuda
        target_movie_id = torch.tensor(eval(specific_row.sequence_movie_ids)[-1:][0]).to(device)
       

        # Target movie rating as well
        # This target movie rating converted to tensor and then stored in cuda device
        target_movie_rating = torch.tensor(eval(specific_row.sequence_ratings)[-1:][0]).to(device)

        # Returning all the necessary features 
        return user_id, user_occupation, user_sex, age_group, movie_his, movie_his_ratings, target_movie_id, target_movie_rating
