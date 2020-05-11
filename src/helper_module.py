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
