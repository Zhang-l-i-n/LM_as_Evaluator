import numpy as np
import umap
import umap.plot
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

X = np.array([[0, 1, 11, 2, 34, 434, 45, 56],
[0, 1, 11, 2, 34, 434, 45, 56],
[0, 1, 11, 2, 34, 434, 0, 56],
[0, 1, 11, 2, 34, 434, 0, 56]])
y = np.array([0, 0, 1, 1])

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset')
plt.show()

# pipe = make_pipeline(SimpleImputer(), QuantileTransformer())
# X_processed = pipe.fit_transform(X)
# manifold = umap.UMAP().fit(X_processed, y)
# umap.plot.points(manifold, labels=y, theme="fire")
