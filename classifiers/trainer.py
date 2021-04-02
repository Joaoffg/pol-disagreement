import pandas as pd
import numpy as np
import statistics
from tqdm.notebook import tqdm

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

class Trainer():

  def __init__(
    self,
    model,
    train_dataset,
    eval_dataset,
    loss_function = None,
    loss_weights = None,
    batch_size = 16
  ):

    self.model = model
    self.train_dataset= train_dataset
    self.eval_dataset = eval_dataset
    self.loss_function = loss_function 
    self.loss_weights = loss_weights
    self.metrics = {"Training": {}, "Eval": {}}
    self.batch_size = batch_size
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


  def compute_metrics(self, y_true, y_pred):
    precision, recall, fscore, support = score(y_true, y_pred)
    accuracy = accuracy_score(y_true,y_pred)
    return {
        "Accuracy" : accuracy,
        "Precision": precision,
        "Recall": recall,
        "Fscore": fscore,
        "Support": support,
        "Macro F1": f1_score(y_true, y_pred, average='macro')
    }

  def create_data_loader(self, dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

  def eval_op(self):
    y_true = []
    y_pred = []
    with torch.no_grad():
      for dl in self.test_dataloader:
        input_ids = dl['input_ids'].to(self.device)
        attention_mask = dl['attention_mask'].to(self.device)
        targets = dl['labels'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        y_true += targets.cpu().numpy().tolist()
        y_pred += nn.Sigmoid()(outputs.logits).round().squeeze().cpu().tolist()
      print('EVAL ACCURACY: {}'.format(accuracy_score(y_true,y_pred)),
            'EVAL F1: {}'.format(f1_score(y_true, y_pred, average='macro')))
    return (y_true, y_pred)

  def fit(self, epochs = 2, learning_rate = 2e-5, use_scheduler = True):

    self.train_dataloader = self.create_data_loader(self.train_dataset, self.batch_size)
    self.test_dataloader = self.create_data_loader(self.eval_dataset,self.batch_size)
    total_steps = len(self.train_dataloader)
    print('TOTAL STEPS:', total_steps)

    optimizer = AdamW(self.model.parameters(),  lr=learning_rate)
    if use_scheduler:
      scheduler = get_linear_schedule_with_warmup(optimizer, total_steps, (epochs-1) * total_steps)

    for epoch in range(epochs):
      step = 0 
      losses = []
      y_true = []
      y_pred = []
      print("EPOCH {}".format(epoch+1))
      for dl in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc="Training... "):
        optimizer.zero_grad()
        step += 1
        input_ids = dl['input_ids'].to(self.device)
        attention_mask = dl['attention_mask'].to(self.device)
        targets = dl['labels'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        y_true += targets.cpu().numpy().tolist()
        y_pred += nn.Sigmoid()(outputs.logits).round().squeeze().cpu().tolist()
        loss = self.loss_function(outputs.logits, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if use_scheduler:
          scheduler.step()
    
        losses.append(loss.detach().cpu().item())
        if step%100 == 0:
          print("TRAINING LOSS: {}".format(statistics.mean(losses)),
                "TRAIN ACCURACY: {}".format(accuracy_score(y_true,y_pred)))
          eval_y_true, eval_y_pred = self.eval_op()
          self.model.train()

      self.metrics["Training"]["Epoch {}".format(epoch)] = self.compute_metrics(y_true, y_pred)
      self.metrics["Eval"]["Epoch {}".format(epoch)] = self.compute_metrics(eval_y_true, eval_y_pred)