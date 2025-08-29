#
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Scrape data
url = "https://www.pro-football-reference.com/years/2023/"
tables = pd.read_html(url)

afc = tables[0]
nfc = tables[1]

print("AFC columns:", afc.columns)
print("NFC columns:", nfc.columns)

# Combine
df = pd.concat([afc, nfc], ignore_index=True)

# Keep only relevant columns
df = df[['Tm', 'W', 'L', 'PF', 'PA', 'PD', 'SoS']]
df.rename(columns={'Tm': 'Team'}, inplace=True)

# Convert numeric cols
for col in ['W', 'L', 'PF', 'PA', 'PD', 'SoS']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 🚨 Drop rows with NaN values (summary rows, etc.)
df = df.dropna(subset=['W', 'L', 'PF', 'PA', 'PD', 'SoS'])

# Target: Winning Season (more than 8 wins out of 17 games)
df['WinningSeason'] = (df['W'] > 8).astype(int)

# Step 2: Save to SQL
conn = sqlite3.connect("NFL.db")
df.to_sql("stats", conn, if_exists="replace", index=False)
conn.close()

# Step 3: Train model
X = df[['PF', 'PA', 'PD', 'SoS']]
y = df['WinningSeason']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")
