
# Code inspired from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html, pytorch documentation

from Model_architecture import BST
from Model_architecture import train_dataloader
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BST().to(device)
criterion = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
train_loader = train_dataloader()


def train_one_epoch():
    local_loss = 0.
    global_avg_loss = 0.
    for i, data in enumerate(train_loader):
        batch = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output, target_movie_rating = model(batch)
        output = output.flatten()

        # Computing the loss
        loss = criterion(output, target_movie_rating) 

        # Backpropahation
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        local_loss += loss.item()
        if i % 1000 == 999:
            global_avg_loss = local_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, global_avg_loss))
            local_loss = 0.
    return global_avg_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
current_epoch_num = 0

Max_epochs = 3
losees = {}
for epoch in range(Max_epochs):
    print('EPOCH {}:'.format(current_epoch_num + 1))

#-------------------------------------------- ---Training the model=------------------------------------------------------------------------##-
    model.train(True)
    avg_loss = train_one_epoch()
    losees.update({epoch : avg_loss})
    current_epoch_num += 1

### If trained this will the save the model here 
torch.save(model.state_dict(), './new_bst_trained_again')