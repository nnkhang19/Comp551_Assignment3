import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from mlp import MLP

class AddOne:
  def __call__(self, x):
    x_new = np.ones(x.shape[0] + 1)
    x_new[:x.shape[0]] = x
    return x_new

class ToOneHot:
  def __init__(self, num_class):
    self.num_class = num_class
  def __call__(self, y):
    return torch.nn.functional.one_hot(y.long(), num_classes = self.num_class)

def create_model(args):
    net = MLP(args.lr, args.num_class, args.reg_coeff, args.reg_type)
    for (dim, act) in zip(args.hidden_dims, args.acts):
      net.add_layer(int(dim), act)
    net.add_layer(args.num_class, 'softmax')
    return net

def plot_loss(losses, title):
  plt.plot(losses, marker = 'x')
  plt.xlabel("Epochs")
  plt.ylabel("Average Epoch Loss")
  plt.title("Training Loss")
  plt.savefig(f"logs/plots/{title}.png", format = "png", bbox_inches = "tight")

def train_model(net, train_loader, train_dataset, args):
  all_loss = []
  for epoch in range(args.num_epochs):
    epoch_loss = 0
    for batch in tqdm(train_loader):
      x, y = batch
      y = torch.nn.functional.one_hot(y.squeeze().long(), num_classes = args.num_class)
      x = x.numpy()
      y = y.numpy()
      all_x = net.forward(x)
      y_pred = all_x[-1]
      grad_list = net.backward(y, all_x)
      loss = net.compute_loss(y, y_pred)
      epoch_loss += loss * len(y)
      net.update_weight(grad_list)
    avg_epoch_loss = epoch_loss / len(train_dataset)
    print(f"Epoch: {epoch}, Loss: {avg_epoch_loss}")
    all_loss.append(avg_epoch_loss)
    plot_loss(all_loss, args.exp_name)
  return net, all_loss

def evaluate(net, test_loader, evalutor, split):
  y_preds = []
  y_gts = []
  for batch in tqdm(test_loader):
      x, y = batch
      x = x.numpy()
      y = y.numpy()
      y_pred = net.predict(x)
      y_gts.append(y.reshape(len(y)))
      y_preds.append(y_pred)
  y_gts = np.concatenate(y_gts, axis = 0)
  y_preds = np.concatenate(y_preds, axis = 0)
  metrics = evalutor.evaluate(y_preds)
  return metrics[0], metrics[1]

def train_cnn(net, loss_fn, optimizer, train_loader, train_dataset, args):
  net.train()
  all_loss = []
  for epoch in range(args.num_epochs):
    epoch_loss = 0
    for batch in tqdm(train_loader):
      optimizer.zero_grad()
      x, y = batch
      x = x.cuda()
      y = y.cuda()
      y_hat = net(x)
      loss = loss_fn(y_hat, y.view(-1).long())
      loss.backward()
      optimizer.step()
      epoch_loss += loss.detach().item() * len(y)
    avg_epoch_loss = epoch_loss / len(train_dataset)
    print(f"Epoch: {epoch}, Loss: {avg_epoch_loss}")
    all_loss.append(avg_epoch_loss)
    plot_loss(all_loss, args.exp_name)
  return net, all_loss

@torch.no_grad()
def evaluate_cnn(net, test_loader, evaluator, split):
  net.eval() 
  y_true = torch.tensor([])
  y_score = torch.tensor([])
  with torch.no_grad():
    for inputs, targets in test_loader:
      inputs = inputs.cuda()
      outputs = net(inputs)      
      targets = targets.squeeze().long()
      outputs = outputs.softmax(dim=-1).cpu()
      targets = targets.float().resize_(len(targets), 1).cpu()
      y_true = torch.cat((y_true, targets), 0)
      y_score = torch.cat((y_score, outputs), 0)
    y_true = y_true.numpy()
    y_score = y_score.detach().numpy() 
    metrics = evaluator.evaluate(y_score) 
    print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))     
    return metrics[0], metrics[1]