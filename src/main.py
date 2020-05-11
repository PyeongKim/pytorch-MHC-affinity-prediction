from model import *
from customlossfunction import *
from preprocessing import *
from helper_module import *
import torch

target_dic = set_target_dic("/content/gdrive/My Drive/Colab Notebooks/training_data.csv", 
                            "/content/gdrive/My Drive/Colab Notebooks/test_data.csv")
                            
train_data = PeptideData("/content/gdrive/My Drive/Colab Notebooks/training_data.csv", target_dic)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           num_workers = 16,
                                           batch_size = 20,
                                           shuffle = True)  

model = resnet18().cuda()
criterion = CustomMSE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
epoch = 10000

for i in range(epoch): 
  train_the_model(train_loader, model, criterion, optimizer, epoch)
