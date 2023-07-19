import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)

RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

TRAIN_PATH = ""
TEST_PATH = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset into a pandas dataframe.
records_train = json.loads(open(TRAIN_PATH).read())
df_train = pd.DataFrame.from_dict(records_train)
records_test = json.loads(open(TEST_PATH).read())
df_test = pd.DataFrame.from_dict(records_test)


mlb = MultiLabelBinarizer()

df_train = df_train.join(pd.DataFrame(mlb.fit_transform(df_train.pop('labels')),
                          columns=mlb.classes_,
                          index=df_train.index))
df_test = df_test.join(pd.DataFrame(mlb.fit_transform(df_test.pop('labels')),
                          columns=mlb.classes_,
                          index=df_test.index))
label_columns = df_train.columns[2:]

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

class IntentDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
      data_row = self.data.iloc[index]
      text = data_row['text']
      labels = data_row[label_columns]

      inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
      ids = inputs['input_ids']
      mask = inputs['attention_mask']
      token_type_ids = inputs["token_type_ids"]


      return {
          'ids': torch.tensor(ids, dtype=torch.long),
          'mask': torch.tensor(mask, dtype=torch.long),
          'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
          'labels': torch.FloatTensor(labels)
      }
training_data = IntentDataset(df_train, tokenizer, max_len=MAX_LEN)
test_data = IntentDataset(df_test, tokenizer, max_len=MAX_LEN)
train_params = {'batch_size': 8,
                'shuffle': True,
                'num_workers': 2
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_data, **train_params)
testing_loader = DataLoader(test_data, **test_params)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 10)
    
    def forward(self, ids, mask, token_type_ids):
        output= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output.pooler_output)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        targets = data['labels'].to(device)


        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('training started')
for epoch in range(EPOCHS):
    train(epoch)


def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['labels'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"F1 Score (Macro) = {f1_score_macro}")
