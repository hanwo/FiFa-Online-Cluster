from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

players = pd.read_csv("../DataSet/player_data.csv")

feature = players[["speed", "shoot", "pass", "dribble", "defense", "physical"]]

model = KMeans(n_clusters=3)
model.fit(feature)
result_kmeans = model.predict(feature)

predict = pd.DataFrame(result_kmeans)
predict.columns = ['predict']

r = pd.concat([feature, predict], axis=1)
for idx, i in enumerate(r['predict']):
    if i == 0:
        r['predict'][idx] = "Striker"
    elif i == 1:
        r['predict'][idx] = "MidField"
    elif i == 2 :
        r['predict'][idx] = "Defense"

# Kmeans 시각화
g = sns.scatterplot(x="speed", y="dribble", hue = 'predict', style = 'predict', data=r)
plt.title('Speed & Dribble')

# Feature 전체 클러스터링에 대한 시각화
sns.pairplot(r, hue="predict", markers=["o", "s", "D"])

plt.show()