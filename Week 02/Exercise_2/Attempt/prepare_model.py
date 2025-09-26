# prepare_model.py

#import libraries
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Collect data
url = "https://www.pro-football-reference.com/years/2023/"
tables = pd.read_html(url)

afc = tables[0]
nfc = tables[1]

print("AFC columns:", afc.columns)
print("NFC columns:", nfc.columns)

# combine AFC and NFC
df = pd.concat([afc, nfc], ignore_index=True)

# remove unnecessary columns
df = df[['Tm', 'W', 'L', 'PF', 'PA', 'PD', 'SoS']]
df.rename(columns={'Tm': 'Team'}, inplace=True)

# convert to numeric
for col in ['W', 'L', 'PF', 'PA', 'PD', 'SoS']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# drop rows with missing values
df = df.dropna(subset=['W', 'L', 'PF', 'PA', 'PD', 'SoS'])

# add WinningSeason column
df['WinningSeason'] = (df['W'] > 8).astype(int)

# Step 2: Save to SQL
conn = sqlite3.connect("NFL.db")
df.to_sql("stats", conn, if_exists="replace", index=False)
conn.close()

# Step 3: Train model
X = df[['PF', 'PA', 'PD', 'SoS']]
y = df['WinningSeason']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save the model
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")
