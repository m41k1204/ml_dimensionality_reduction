import pandas as pd
import matplotlib
import os
from spectralEmbeddingTransformer import SpectralEmbeddingTransformer

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gdown
import numpy as np
import struct
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.manifold import Isomap, SpectralEmbedding


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
    scaler = MinMaxScaler()
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


def isomap_embedding(
    X,
    n_components=2,
    n_neighbors=5,
    eigen_solver="auto",
    n_jobs=-1,
):
    iso = Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors,
        eigen_solver=eigen_solver,
        n_jobs=n_jobs,
    )

    X_embedded = iso.fit_transform(X)

    return iso, X_embedded


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
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f")
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
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        # "Logistic Regression": LogisticRegression(
        #     max_iter=5000, solver="saga", class_weight="balanced", random_state=42
        # ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
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


def cargar_datos_locales(train_X_path, test_X_path, train_Y_path, test_Y_path):
    print("Cargando datos desde archivos locales...")

    train_X = extrar_feature_images(train_X_path)
    test_X = extrar_feature_images(test_X_path)
    train_Y = read_labels(train_Y_path)
    test_Y = read_labels(test_Y_path)

    return train_X, test_X, train_Y, test_Y


def main():
    train_X, test_X, train_Y, test_Y = cargar_datos_locales(
        "train_X", "test_X", "train_Y", "test_Y"
    )

    print("Data train original: ", train_X.shape)
    print("Label train original: ", train_Y.shape)
    print("Data test original: ", test_X.shape)
    print("Label test original: ", test_Y.shape)

    print("\n=== FASE 1: PREPROCESAMIENTO ===")
    train_X_norm, scaler = normalize_train(train_X)
    test_X_norm = normalize_test(test_X, scaler)
    print(f"Rango antes de normalizar: [{train_X.min()}, {train_X.max()}]")
    print(f"Rango después de normalizar: [{train_X_norm.min()}, {train_X_norm.max()}]")

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

    # print("\n=== FASE 2A: SPECTRAL EMBEDDING (2D para visualización) ===")
    # print("Aplicando Spectral Embedding para visualización 2D...")

    # n_viz = min(60000, len(train_X_norm))
    # viz_indices = np.random.choice(len(train_X_norm), n_viz, replace=False)
    # X_viz = train_X_norm[viz_indices]
    # y_viz = train_Y["label"].values[viz_indices]

    # print(f"\nAplicando Spectral Embedding 2D en {n_viz} muestras...")

    # se_2d = SpectralEmbeddingTransformer(
    #     n_components=2, n_neighbors=10, affinity="nearest_neighbors", random_state=42
    # )
    # X_spectral_2d = se_2d.fit_transform(X_viz)
    # print(f"Shape después de Spectral Embedding (2D): {X_spectral_2d.shape}")

    # visualizar_reduccion_2d(
    #     X_spectral_2d,
    #     y_viz,
    #     "Spectral Embedding",
    #     class_names_dict,
    # )

    print("\n=== FASE 2B: SPECTRAL EMBEDDING (250 componentes para clasificación) ===")

    n_components_clf = 250
    n_train_samples = min(60000, len(train_X_norm))

    print(f"Procesando {n_train_samples} muestras de entrenamiento...")

    train_indices = np.random.choice(len(train_X_norm), n_train_samples, replace=False)
    X_train_sample = train_X_norm[train_indices]

    se_clf = SpectralEmbeddingTransformer(
        n_components=n_components_clf,
        n_neighbors=15,
        affinity="nearest_neighbors",
        random_state=42,
    )

    train_X_spectral_sample = se_clf.fit_transform(X_train_sample)
    print(f"✓ Spectral Embedding completado. Shape: {train_X_spectral_sample.shape}")

    print(f"\nTransformando todos los datos de entrenamiento...")
    train_X_spectral = se_clf.transform(train_X_norm)
    print(f"Train Spectral Embedding shape: {train_X_spectral.shape}")

    print(f"Transformando datos de test...")
    test_X_spectral = se_clf.transform(test_X_norm)
    print(f"Test Spectral Embedding shape: {test_X_spectral.shape}")

    print("\n=== FASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS ===")
    print("--- Evaluación con Spectral Embedding ---")

    resultados_spectral = evaluar_modelos(
        train_X_spectral,
        train_Y["label"].values,
        test_X_spectral,
        test_Y["label"].values,
        mostrar_matrices=True,
        save_dir="resultados_spectral",
    )

    print("\n--- Resultados con Spectral Embedding ---")
    print(resultados_spectral.to_string(index=False))

    print("\n✓ Proceso completado exitosamente!")


if __name__ == "__main__":
    main()
