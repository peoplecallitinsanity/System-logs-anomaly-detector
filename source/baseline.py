import os
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score



def make_df(directory):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):  
            file_path = os.path.join(directory, filename)
            # print(file_path)
            with open(file_path, 'r') as file:
                # print(file)
                data = json.load(file)

            row_data = {'index_json': filename.split('.')[0]}    
            def flatten_json(y, prefix=''):
                if isinstance(y, dict):
                    for k, v in y.items():
                        flatten_json(v, prefix + k + '_')
                elif isinstance(y, list):
                    for index, item in enumerate(y):
                        flatten_json(item, prefix + str(index) + '_')
                else:
                    row_data[prefix[:-1]] = y
            flatten_json(data)
            data_list.append(row_data)

    return pd.DataFrame(data_list)

df_train = make_df('/usr/src/app/source/train')
df_test = make_df('/usr/src/app/source/test')

df_combined = pd.concat([df_train, df_test], ignore_index=True)

def drop_columns_with_nan(df, threshold=0.97):
    if 'index_json' in df.columns:
        index_json_col = df['index_json'].copy()
    else:
        index_json_col = None

    nan_percentage = df.drop(columns=['index_json'], errors='ignore').isna().mean()
    columns_to_keep = nan_percentage[nan_percentage < threshold].index
    df = df[columns_to_keep]

    if index_json_col is not None:
        df['index_json'] = index_json_col

    return df

def encode_text_columns(df):
    if 'index_json' in df.columns:
        index_json_col = df['index_json'].copy()
    else:
        index_json_col = None

    df_encoded = df.drop(columns=['index_json'], errors='ignore')
    for column in df_encoded.columns:
        df_encoded[column] = df_encoded[column].apply(lambda x: str(x) if isinstance(x, bool) else x)
        if df_encoded[column].dtype == 'object' and all(isinstance(val, str) or pd.isna(val) for val in df_encoded[column]):
            le = LabelEncoder()
            df_encoded[column] = df_encoded[column].fillna("NaN_Category")
            df_encoded[column] = le.fit_transform(df_encoded[column])
    df_encoded = df_encoded.fillna(-1)

    if index_json_col is not None:
        df_encoded['index_json'] = index_json_col

    return df_encoded
df_combined = drop_columns_with_nan(df_combined)
df_combined_encoded = encode_text_columns(df_combined)

df_train_encoded = df_combined_encoded.iloc[:len(df_train)].reset_index(drop=True)
df_test_encoded = df_combined_encoded.iloc[len(df_train):].reset_index(drop=True)

df_train_encoded = df_train_encoded.drop(columns=['index_json'], errors='ignore')

X = df_train_encoded.drop(columns=['label']) 
y = df_train_encoded['label']

X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=42, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 350)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(350, 220)
        self.fc3 = nn.Linear(220, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        return out

input_size = X.shape[1]
hidden_size = 32 
num_classes = len(y.unique())


model = SimpleNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0006)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad() 
            outputs = model(features) 
            loss = criterion(outputs, labels)  
            loss.backward() 
            optimizer.step()  

train_model(model, train_loader, criterion, optimizer, num_epochs=110)

y_test = df_test_encoded['label']
X_test = df_test_encoded.drop(columns=['label', 'index_json'])
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=42, shuffle=False)
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    return predictions


predictions = predict(model, test_loader)
prediction_df = pd.DataFrame({
    'Index': df_test_encoded.index_json,
    'Prediction': predictions,
    'actual' : df_test_encoded.label
})

prediction_df.set_index('Index', inplace=True)
prediction_df.drop(columns=['actual'], inplace=True, errors='ignore') 
json_string = prediction_df['Prediction'].to_json(orient='index')
with open('/usr/src/app/output/labels', 'w') as file:
    file.write(json_string)

accuracy = accuracy_score(y_test, predictions)
print(f"Acuratețea modelului pe setul de testare: {accuracy * 100:.2f}%")