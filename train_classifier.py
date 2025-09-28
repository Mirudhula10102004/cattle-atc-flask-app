import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv('animal_measurements_clustered.csv').dropna()

X = data[['body_length','height_at_withers','chest_width']]
y = data['cluster_label']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

joblib.dump(clf, 'atc_classifier.joblib')
