# train_classifier.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------
data_path = './data.pickle'  # Make sure this file is in the same folder
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

print(f"Original number of samples: {len(data_dict['data'])}")

# -----------------------------
# Handle variable-length feature vectors
# -----------------------------
max_len = max(len(x) for x in data_dict['data'])
print(f"Padding all feature vectors to length: {max_len}")

data_fixed = np.array([
    np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else np.array(x[:max_len])
    for x in data_dict['data']
])

labels = np.array(data_dict['labels'])

print(f"Data shape after padding: {data_fixed.shape}")
print(f"Labels shape: {labels.shape}")

# -----------------------------
# Split into training and test sets
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data_fixed, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# -----------------------------
# Train Random Forest classifier
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# -----------------------------
# Evaluate the model
# -----------------------------
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
print("Classification report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save the trained model
# -----------------------------
model_file = 'model.p'
with open(model_file, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"\nRandom Forest model saved to '{model_file}'")