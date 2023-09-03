### DISCLAIMER : 
Most files take a long time to run especially on cpu, as this model is trained on NCC GPUs and the model is very deep therefore in the info below I will mention the files which would take very long to run, and should only be run on CUDA supported environments.

#### For_per_user_rec
This folder stored dataframes for users 1-200 with the predicted ratigns of the movies those users have not watched, the resaon for this as explained below and above is that it would not be possible to retrain the model, and make predictions on a latop with not a dedicated graphics card.

#### Accuray_metric.py
Calculates the RMSE loss for the model on the test set which is split as 75:25, stratified by user. This would just return a float for the accuracy metric of the BST model

#### Dataloader.py
This is the custom dataloader designed to make data into batches of 128 and feed the input such that it is in the right format for our transformer model, as mentioned in the code this dataloader is inspired from a previous implementation and has been cited

#### main.py
This contains the File you need to run to get the interface and the reccomendations
#### Imporatant point here 
The interface would give an option of 'pers' : Personalised and 'nonpers' : Non-Personalised
This means that if you choose 'pers' you will login as user and also can input 'info' to access what inforation/context is being taken as input from your user profile
#### Important thing about personalised reccomendation
When getting a personalised reccomendation from scratch it takes around 8s on GPU RTX3070, but would take a lot longer on the machine outlined in the mark scheme as it would be on CPU.
Hence to make the process more streamlined for testing purposes, I have stored predicted ratings for Users 1 - 200 already so if you choose any user number from [1,200] you will be able to get a personalised reccomendation instantly, however to test if my model actually can create the new reccomendations you can choose any number outside that range to test that, but it would <b> Take a significant amount of time if ran on your machine locally </b>. 
Finally, once entered the interface the filtering you can perform:
--> Filter by Genres
--> Filter by decade for eg if you want films reccomended from the 1990s you just have to input '1990s'
--> To clear filters just input 'clear' 
--> To exit press ctrl + c to exit the PS and then you can enter the NP system

#### making_reccomendations.py
This file makes reccomendations for a given user with the use of the BST model

#### Model_architecture.py
This is where the architecture of the model lies, the code has been adapted from some sources and has added functionality to it, but this is the file I use to import the architecture of the model

#### nDCG.py
This is to calculate the nDCG metric for randomly selected 100 unique users and then the mean is taken to be a determinaiton of how the BST model performs in terms of ranking and predicting the ground truth of rankings.

#### newbst
This is the file that saves all the paramters/weights of the model, these are the parameters we get after training the model for 30-40 mins
this is loaded in several files in order to make predicitons

#### Novelty.py
This is to calculate the novelty as an evaluation metric for the model, takes a random of 100 users and gives predictions and then calculates the avg novelty, the novelty funcytion has been taken from the Recmetrics library/ repositry

#### Personalised.py
This is the personalised reccomender system

#### Processing_Personalised.py
These are the functions required to preprocess data in the correct format required by the BST model

#### reccomender_system.py
The reccomender system itself
#### saving.py
This was used to save user reccomendations such that not too much time was taken to generate new reccomendatiosn
#### training.py
This is for training the model, take a <b> Takes a long time even on NCC 4 mins per epoch, only train if CUDA is available </b>