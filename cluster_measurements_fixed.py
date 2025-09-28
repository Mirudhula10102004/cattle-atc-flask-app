import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('animal_measurements_debug.csv').dropna(subset=['body_length','height_at_withers','chest_width'])

features = ['body_length', 'height_at_withers', 'chest_width']
X = data[features]

kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster_label'] = kmeans.fit_predict(X)

data.to_csv('animal_measurements_clustered.csv', index=False)

plt.scatter(data['body_length'], data['chest_width'], c=data['cluster_label'])
plt.xlabel('Body Length')
plt.ylabel('Chest Width')
plt.title('KMeans clusters for animal measurements')
plt.savefig('cluster_plot.png')
plt.show()
