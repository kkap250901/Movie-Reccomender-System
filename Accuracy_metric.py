# To get the accuracy metric fo rour model chosen as the RMSE, just run this and get the score 
# On the test dataset
from Model_architecture import BST  
from Model_architecture import test_dataloader
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BST().to(device)
model.load_state_dict(torch.load('./new_bst'))
tesing = test_dataloader()

maeggg = nn.L1Loss().to(device)
with torch.no_grad():
    loss_test = 0
    for i, data in enumerate(tesing):
        output_v, target_v = model(data)
        output_v = output_v.flatten()
        loss_test += maeggg(output_v, target_v)
    loss_avg = loss_test/i

print('RMSE on the test set: ', loss_avg)

