import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import numpy as np
import pandas as pd
import json
import csv
#=========================Kaggle report============================
#I use BERT pretrained model,
#than I create the classifier to classify the emotions.
#I run the model 10 epochs, and save the best model with the highest val acc,
#The last thing is to generate the test .csv with test data.
#I tried to balance the data, but the result isn't good enough.
#I also tried to use KFold to cross_val the model,but it just generate the second best score.
#Otherwise,I use different model like albert and roberta,but it got poor results than any other model trained based on BERT.
#The following is the baseline method which I got the highest final score.
#=========================Kaggle report============================


#=======================data preprocessing=========================

# open file: tweets_DM.json
file = open ('tweets_DM.json','r')
data = []
for line in file.readlines():
    dic = json.loads(line)
    data.append( [dic['_source']['tweet']['tweet_id'],dic['_source']['tweet']['text']] )
df_orig = pd.DataFrame (data, columns = ['id','tweet'])
df = pd.DataFrame (data, columns = ['id','tweet'])


# open file: emotion.csv<br>
# delete data which [emotion] is NaN
emo = []
with open('emotion.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        emo.append(row)
df_emo = pd.DataFrame (emo, columns = ['id','emotion'])    
df['emotion'] = df.id.map(df_emo.set_index('id')['emotion'])

print("Sum of NaN in [emotion]:",df['emotion'].isna().sum(),"\nSum of data:",len(df))


#delete the data which emotion is NaN
df = df[df.emotion.notnull()]
print("Sum of NaN in [emotion]:",df['emotion'].isna().sum(),"\nSum of data:",len(df))
df[:5]



# open file: data_identification.csv
# create df_test
ide = []
with open('data_identification.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        ide.append(row)
df_ide = pd.DataFrame (ide, columns = ['id','identification'])    
df['identification'] = df.id.map(df_ide.set_index('id')['identification'])
df_orig['identification'] = df_orig.id.map(df_ide.set_index('id')['identification'])
print("Sum of NaN in [identification]:",df['identification'].isna().sum(),"\nSum of data:",len(df))
df_test = df_orig[df_orig['identification']=="test"]
df[:5],df.groupby("emotion").size()



#label encoder
from sklearn import preprocessing, metrics, decomposition, pipeline, dummy
mlb = preprocessing.LabelEncoder()
mlb.fit(df.emotion)
emotion_names = mlb.classes_
print(emotion_names)
df['NUMotion'] = mlb.transform(df['emotion']).tolist()
df[:5]

#=======================data preprocessing=========================

#please install the transformers,watermark,Pytorch
#!pip install -qq transformers
#!pip install watermark
#!pip install barbar
#%reload_ext watermark
#%watermark -v -p numpy,pandas,torch,transformers

#=====================create training dataset======================

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from barbar import Bar


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# select GPU to run.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# it will take "some" time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# to calculate the MAX_LEN 
'''
token_lens = []
for txt in df.tweet:
    tokens = tokenizer.encode(txt)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 180]);
plt.xlabel('Token count');
'''
MAX_LEN = 100


#  create a PyTorch dataset
class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True,
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

# split training data
df_train, df_val = train_test_split(
  df,
  test_size=0.1,
  random_state=42
)
'''
#test with small training dataset

df_train, df_other = train_test_split(
  df_train,
  test_size=0.95,
  random_state=42
)
df_val, df_other = train_test_split(
  df_other,
  test_size=0.99,
  random_state=42
)
'''
df_train.shape, df_val.shape


# create data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.tweet.to_numpy(),
    targets=df.NUMotion.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
    )


BATCH_SIZE = 16
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

#=====================create training dataset======================

#==========================create model============================

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


# create the classifier
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = EmotionClassifier(len(emotion_names))
# I use two GPUs to run
if torch.cuda.device_count()>1:
      model=nn.DataParallel(model,device_ids=[0,1])
model = model.to(device)


EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
    ):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in Bar(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in Bar(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
#==========================create model============================


#==========================train state=============================
best_accuracy = 0
history = defaultdict(list)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'final_best_model_state.bin')
        best_accuracy = val_acc
#==========================train state=============================


#===========================val state==============================
#load model
#model.load_state_dict(torch.load('final_best_model_state.bin'))
print("# of test data: ",len(df_test))
df_test[:5]

# output the test file to upload the .csv 
with open('FinalSubmission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'emotion'])
    for index, row in df_test.iterrows():
        #print(row['id'],row['tweet'])
        #the .csv used as my final score didn't use model.eval()
        #so when in the predict state , the model is still on the the training state.
        #model.eval()
        encoded_tweet = tokenizer.encode_plus(
          row['tweet'],
          max_length=MAX_LEN,
          add_special_tokens=True,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True,
        )
        input_ids = encoded_tweet['input_ids'].to(device)
        attention_mask = encoded_tweet['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        writer.writerow( [row['id'] , emotion_names[prediction]] ) 
#===========================val state==============================
