from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

players = pd.read_csv("player_data.csv")

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
feature_df = pd.DataFrame(feature)

transformed = TSNE(n_components=2).fit_transform(feature_df)

xs = transformed[:,0]
ys = transformed[:,1]
g = sns.scatterplot(x=xs, y=ys, hue = 'predict', style = 'predict', data=r)
g.set_title("t-SNE")
plt.legend(loc='best', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()