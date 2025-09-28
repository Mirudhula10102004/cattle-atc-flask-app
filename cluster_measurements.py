import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your measurements CSV
data = pd.read_csv('animal_measurements.csv')
data = data.dropna()

features = ['body_length', 'height_at_withers', 'chest_width', 'rump_angle']
X = data[features]

# Choose number of clusters - start with 2 (cattle vs buffalo)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster_label'] = kmeans.fit_predict(X)

data.to_csv('animal_measurements_clustered.csv', index=False)
print(data[['image_name', 'cluster_label']].head())

# Optional visualization
plt.scatter(data['body_length'], data['chest_width'], c=data['cluster_label'])
plt.xlabel('Body Length')
plt.ylabel('Chest Width')
plt.title('Clustering of Animal Measurements')
plt.show()
