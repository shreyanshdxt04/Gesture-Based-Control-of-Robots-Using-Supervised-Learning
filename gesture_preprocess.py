import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

dataset_dir = 'dataset'
all_data = []

for file in os.listdir(dataset_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(dataset_dir, file))
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
X = data.drop('label', axis=1)
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, 'label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Preprocessing complete.")
