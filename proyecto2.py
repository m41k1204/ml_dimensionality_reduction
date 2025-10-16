# Proyecto 2: Reducción de la dimensionalidad

import pandas as pd
import matplotlib

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


def normalizar(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler


def add_bias(X):
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    X_bias = np.hstack((bias_column, X))
    return X_bias


def generar_matriz_confusion(y_true, y_pred, modelo_nombre, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - {modelo_nombre}")
    plt.tight_layout()
    plt.show()
    return cm


def evaluar_modelos(X_train, y_train, X_test, y_test, mostrar_matrices=True):
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

    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        resultados.append(
            {
                "Modelo": nombre,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            }
        )

        if mostrar_matrices:
            generar_matriz_confusion(y_test, y_pred, nombre, class_names)

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

    print("Data train : ", train_X.shape)
    print("Label train : ", train_Y.shape)
    print("Data test : ", test_X.shape)
    print("Label test : ", test_Y.shape)

    # Fase 1: Preprocesamiento - Normalización
    train_X_norm, scaler = normalizar(train_X)
    test_X_norm = scaler.transform(test_X)
    print(f"Rango antes de normalizar: [{train_X.min()}, {train_X.max()}]")
    print(f"Rango después de normalizar: [{train_X_norm.min()}, {train_X_norm.max()}]")

    # Agregando columna de bias (opcional)
    train_X_bias = add_bias(train_X_norm)
    test_X_bias = add_bias(test_X_norm)
    print(f"Shape antes de bias: {train_X_norm.shape}")
    print(f"Shape después de bias: {train_X_bias.shape}")
    print(f"Primera columna (bias): {train_X_bias[0, 0]}")

    # Fase 3: Clasificación y evaluación
    print("\n=== FASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS ===")
    resultados = evaluar_modelos(
        train_X_norm, train_Y["label"].values, test_X_norm, test_Y["label"].values
    )
    print("\n--- Resultados ---")
    print(resultados.to_string(index=False))

    # image_number = 45
    # Show_Image(train_X, image_number)


if __name__ == "__main__":
    main()
