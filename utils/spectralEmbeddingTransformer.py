from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import SpectralEmbedding
import numpy as np


class SpectralEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        n_neighbors=5,
        affinity="nearest_neighbors",
        gamma=None,
        random_state=42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        print(f"Entrenando Spectral Embedding con {X.shape[0]} muestras...")

        se = SpectralEmbedding(
            n_components=self.n_components,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            gamma=self.gamma,
            random_state=self.random_state,
            eigen_solver="arpack",
            n_jobs=-1,
        )

        self.embedding_ = se.fit_transform(X)
        self.X_train_ = X

        self.knn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
            algorithm="auto",
            n_jobs=-1,
        )
        self.knn_.fit(X)

        print(f"✓ Spectral Embedding completado. Shape: {self.embedding_.shape}")

        return self

    def transform(self, X):
        if not hasattr(self, "embedding_"):
            raise ValueError(
                "El transformador no ha sido entrenado aún. Ejecuta fit() primero."
            )

        print(f"Proyectando {X.shape[0]} muestras al espacio embebido...")

        distances, indices = self.knn_.kneighbors(X)
        distances = distances + 1e-10

        weights = 1.0 / distances
        weights /= weights.sum(axis=1, keepdims=True)

        X_transformed = np.zeros((X.shape[0], self.n_components))
        for i, neighbor_indices in enumerate(indices):
            X_transformed[i] = np.average(
                self.embedding_[neighbor_indices], axis=0, weights=weights[i]
            )

        print(f"✓ Proyección completada. Shape: {X_transformed.shape}")

        return X_transformed
