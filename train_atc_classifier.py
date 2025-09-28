import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the CSV measurements and your labeled ATC scores
# CSV should have columns: ['body_length', 'height_at_withers', 'chest_width', 'rump_angle', 'atc_label']
data = pd.read_csv('animal_measurements_with_labels.csv')

# Basic preprocessing - drop rows with missing data
data.dropna(inplace=True)

X = data[['body_length', 'height_at_withers', 'chest_width', 'rump_angle']]
y = data['atc_label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# Save model using joblib if needed
import joblib
joblib.dump(clf, 'atc_classifier_model.joblib')
