import pandas as pd

data = pd.read_csv('animal_measurements_clustered.csv')

print("CSV file info:")
print(data.info())

print("\nShape after dropping rows with missing key features:")
print(data[['body_length', 'height_at_withers', 'chest_width']].dropna().shape)

print("\nFirst 10 rows:")
print(data.head(10))
