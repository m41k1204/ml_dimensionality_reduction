# Proyecto 2: Reducción de la dimensionalidad

import pandas as pd
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gdown
import numpy as np
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import SpectralEmbedding


def download_data(file_id, name_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, name_file, quiet=False)
    return name_file


def read_labels(file_path):
    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    with open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    df = pd.DataFrame(labels, columns=["label"])
    df["class_name"] = df["label"].map(class_names)
    return df


def extrar_feature_images(file_path):

    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("Número de imágenes:", num_images)
        print("Dimensiones de cada imagen:", rows, "x", cols)
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        X = images.reshape(num_images, rows * cols)
        print("Forma de la matriz final:", X.shape)
    return X


def Show_Image(X, nro_imagen):
    if nro_imagen < 0 or nro_imagen >= X.shape[0]:
        raise IndexError(
            f"El índice {nro_imagen} está fuera de rango. Debe estar entre 0 y {X.shape[0]-1}"
        )

    img = X[nro_imagen].reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.title(f"Imagen #{nro_imagen}")
    plt.axis("off")
    plt.show()


def normalize_train(x):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    return x_norm, scaler


def normalize_test(x, scaler):
    x_norm = scaler.transform(x)
    return x_norm


def spectral_embedding(
    X,
    n_components=2,
    n_neighbors=10,
    affinity="nearest_neighbors",
    gamma=None,
    random_state=42,
    eigen_solver="arpack",
    n_jobs=-1,
):

    se = SpectralEmbedding(
        n_components=n_components,
        affinity=affinity,
        n_neighbors=n_neighbors,
        gamma=gamma,
        random_state=random_state,
        eigen_solver=eigen_solver,
        n_jobs=n_jobs,
    )

    X_embedded = se.fit_transform(X)

    return X_embedded


def visualizar_reduccion_2d(X_reduced, y, metodo_nombre, class_names_dict):
    plt.figure(figsize=(12, 10))

    for clase in np.unique(y):
        indices = y == clase
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            label=class_names_dict[clase],
            alpha=0.6,
            s=20,
        )

    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.title(f"Visualización 2D - {metodo_nombre}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def addBias(x):
    return np.column_stack([np.ones(x.shape[0]), x])


def generar_matriz_confusion(
    y_true, y_pred, modelo_nombre, class_names=None, save_path=None
):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - {modelo_nombre}")
    plt.tight_layout()

    if save_path is None:
        modelo_safe = modelo_nombre.replace(" ", "_").replace("/", "_")
        save_path = f"confusion_matrix_{modelo_safe}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Matriz de confusión guardada en: {save_path}")

    plt.close(fig)

    return cm


def evaluar_modelos(
    X_train, y_train, X_test, y_test, mostrar_matrices=True, save_dir="resultados"
):
    modelos = {
        "SVM": SVC(kernel="rbf", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    class_names = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    if mostrar_matrices:
        os.makedirs(save_dir, exist_ok=True)

    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        resultados.append(
            {
                "Modelo": nombre,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "F1-Score": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }
        )

        if mostrar_matrices:
            modelo_safe = nombre.replace(" ", "_")
            save_path = os.path.join(save_dir, f"confusion_{modelo_safe}.png")
            generar_matriz_confusion(y_test, y_pred, nombre, class_names, save_path)

    return pd.DataFrame(resultados)


def main():
    # Descargando la data solo la primera vez
    file_train_X = download_data("1enziBIpqiv_t95KQcifsclNH2BdR8lAd", "train_X")
    file_test_X = download_data("1Jeax6tnQ6Nmr2PTNXdQqzKnN0YqtrLe4", "test_X")
    file_train_Y = download_data("1MZtn2iA5cgiYT1i3O0ECuR01oD0kGHh7", "train_Y")
    file_test_Y = download_data("1K5pxwk2s3RDYsYuwv8RftJTXZ-RGR7K4", "test_Y")

    train_X = extrar_feature_images(file_train_X)
    test_X = extrar_feature_images(file_test_X)
    train_Y = read_labels(file_train_Y)
    test_Y = read_labels(file_test_Y)

    print("Data train original: ", train_X.shape)
    print("Label train original: ", train_Y.shape)
    print("Data test original: ", test_X.shape)
    print("Label test original: ", test_Y.shape)

    # Fase 1: Preprocesamiento
    print("\n=== FASE 1: PREPROCESAMIENTO ===")
    train_X_norm, scaler = normalize_train(train_X)
    test_X_norm = normalize_test(test_X, scaler)
    print(f"Rango antes de normalizar: [{train_X.min()}, {train_X.max()}]")
    print(f"Rango después de normalizar: [{train_X_norm.min()}, {train_X_norm.max()}]")

    # Agregar bias
    train_X_norm = addBias(train_X_norm)
    test_X_norm = addBias(test_X_norm)
    print(
        f"Shape después de agregar bias - Train: {train_X_norm.shape}, Test: {test_X_norm.shape}"
    )

    # Diccionario de nombres de clases para visualización
    class_names_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    print("\n=== FASE 2: REDUCCIÓN DE DIMENSIONALIDAD ===")
    print("Combinando train y test para Spectral Embedding...")

    X_combined = np.vstack([train_X_norm, test_X_norm])
    y_combined = np.concatenate([train_Y["label"].values, test_Y["label"].values])

    print(f"Shape combinado: {X_combined.shape}")
    print(
        "⚠️ Nota: Spectral Embedding no tiene transform(), por eso combinamos train+test"
    )

    print("\n--- Generando visualización 2D ---")
    n_viz = min(3000, len(X_combined))
    viz_indices = np.random.choice(len(X_combined), n_viz, replace=False)
    X_viz = X_combined[viz_indices]
    y_viz = y_combined[viz_indices]

    print(f"Aplicando Spectral Embedding 2D en {n_viz} muestras...")
    X_spectral_2d = spectral_embedding(
        X_viz, n_components=2, n_neighbors=10, eigen_solver="arpack", n_jobs=-1
    )
    print(f"Shape después de Spectral Embedding (2D): {X_spectral_2d.shape}")

    # Visualización
    visualizar_reduccion_2d(
        X_spectral_2d,
        y_viz,
        "Spectral Embedding",
        class_names_dict,
    )

    n_components_clf = 80
    print(f"\n--- Aplicando Spectral Embedding con {n_components_clf} componentes ---")
    print(f"Procesando {len(X_combined)} muestras...")

    X_combined_spectral = spectral_embedding(
        X_combined,
        n_components=n_components_clf,
        n_neighbors=10,
        eigen_solver="arpack",
        n_jobs=-1,
    )
    print(f"✓ Embedding completado. Shape: {X_combined_spectral.shape}")

    print("\n--- Separando train y test ---")
    train_X_spectral = X_combined_spectral[:60000]
    test_X_spectral = X_combined_spectral[60000:]

    print(f"Train spectral shape: {train_X_spectral.shape}")
    print(f"Test spectral shape: {test_X_spectral.shape}")

    print("\n=== FASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS ===")
    print("--- Evaluación con Spectral Embedding ---")

    resultados_spectral = evaluar_modelos(
        train_X_spectral,
        train_Y["label"].values,
        test_X_spectral,
        test_Y["label"].values,
        mostrar_matrices=True,
    )

    print("\n--- Resultados con Spectral Embedding ---")
    print(resultados_spectral.to_string(index=False))

    print("\n✓ Proceso completado exitosamente!")


if __name__ == "__main__":
    main()
