from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

players = pd.read_csv("player_data.csv")

feature = players[["speed", "shoot", "pass", "dribble", "defense", "physical"]]
name = players["name"]

model = KMeans(n_clusters=3)
model.fit(feature)
result_kmeans = model.predict(feature)

predict = pd.DataFrame(result_kmeans)
predict.columns = ['predict']

r = pd.concat([feature, predict], axis=1)
print(r.head(15))
for idx, i in enumerate(r['predict']):
    if i == 0:
        r['predict'][idx] = "Striker"
    elif i == 1:
        r['predict'][idx] = "MidField"
    elif i == 2 :
        r['predict'][idx] = "Defense"

g = sns.scatterplot(x="speed", y="dribble", hue = 'predict', style = 'predict', data=r)
# sns.pairplot(r, hue="predict", markers=["o", "s", "D"])
# plt.xlabel('Speed')
# plt.ylabel('dribble')
g.set_title("speed & dribble")
# g.legend(loc='lower right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()