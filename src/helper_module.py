def set_target_dic(df_path_1, df_path_2):
  df_1 = pd.read_csv(df_path_1)
  df_2 = pd.read_csv(df_path_2)
  data_target_1 = df_1.iloc[:,1]
  data_target_2 = df_2.iloc[:,1]
  unique_targets_1 = list(set(data_target_1))
  unique_targets_2 = list(set(data_target_2))
  unique_targets = set(unique_targets_1+ unique_targets_2)
  target_dic = {}
  for n, target in enumerate(unique_targets):
    target_dic[target] = n
  return target_dic


def train_the_model(dataloader, model, criterion, optimizer, epoch):
  loss_sum = 0
  for batch, (peptide, label, target) in enumerate(dataloader):
      #! nvidia-smi
      peptide = Variable(peptide.cuda())
      target = Variable(target.cuda())
      outputs = model(peptide)
      loss = criterion(outputs, label[:,0], label[:,1], target)
      optimizer.zero_grad()
      loss.backward()
      # Update weights
      optimizer.step()
      loss_sum += loss
  print("epoch: {} | loss: {}".format(epoch, loss_sum))
