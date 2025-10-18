import pandas as pd
import matplotlib
import os
import numpy as np
import struct
from scipy import ndimage
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


# ============= FUNCIONES DE CARGA DE DATOS =============


def read_labels(file_path):
    """Lee los labels del archivo binario"""
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
    """Extrae características de imágenes del archivo binario"""
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("Número de imágenes:", num_images)
        print("Dimensiones de cada imagen:", rows, "x", cols)
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        X = images.reshape(num_images, rows * cols)
        print("Forma de la matriz final:", X.shape)
    return X


def cargar_datos_locales(train_X_path, test_X_path, train_Y_path, test_Y_path):
    """Carga todos los datos desde archivos locales"""
    print("Cargando datos desde archivos locales...")
    train_X = extrar_feature_images(train_X_path)
    test_X = extrar_feature_images(test_X_path)
    train_Y = read_labels(train_Y_path)
    test_Y = read_labels(test_Y_path)
    return train_X, test_X, train_Y, test_Y


# ============= FUNCIONES DE VISUALIZACIÓN =============

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


def visualizar_clase(train_X, train_Y, clase_id, num_imagenes=10):
    """
    Visualiza imágenes de una clase específica

    Args:
        train_X: matriz de características (n_samples, 784)
        train_Y: DataFrame con labels y class_name
        clase_id: número de clase (0-9)
        num_imagenes: cuántas imágenes mostrar
    """
    indices = np.where(train_Y["label"] == clase_id)[0]

    indices_seleccionados = np.random.choice(
        indices, min(num_imagenes, len(indices)), replace=False
    )

    num_cols = 5
    num_rows = (len(indices_seleccionados) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    clase_nombre = class_names[clase_id]

    for i, idx in enumerate(indices_seleccionados):
        img = train_X[idx].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{clase_nombre}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Clase: {clase_nombre} ({len(indices)} imágenes disponibles)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def comparar_clases(train_X, train_Y, clase1_id, clase2_id, num_pares=5):
    """
    Compara dos clases lado a lado

    Args:
        train_X: matriz de características
        train_Y: DataFrame con labels
        clase1_id: ID primera clase
        clase2_id: ID segunda clase
        num_pares: cuántos pares mostrar
    """
    indices1 = np.where(train_Y["label"] == clase1_id)[0]
    indices2 = np.where(train_Y["label"] == clase2_id)[0]

    indices1_sel = np.random.choice(indices1, num_pares, replace=False)
    indices2_sel = np.random.choice(indices2, num_pares, replace=False)

    fig, axes = plt.subplots(num_pares, 2, figsize=(8, 3 * num_pares))

    nombre1 = class_names[clase1_id]
    nombre2 = class_names[clase2_id]

    for i in range(num_pares):
        img1 = train_X[indices1_sel[i]].reshape(28, 28)
        axes[i, 0].imshow(img1, cmap="gray")
        axes[i, 0].set_title(f"{nombre1}", fontweight="bold")
        axes[i, 0].axis("off")

        img2 = train_X[indices2_sel[i]].reshape(28, 28)
        axes[i, 1].imshow(img2, cmap="gray")
        axes[i, 1].set_title(f"{nombre2}", fontweight="bold")
        axes[i, 1].axis("off")

    plt.suptitle(f"Comparación: {nombre1} vs {nombre2}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def visualizar_original_vs_sobel(train_X, train_Y, clase_id, num_imagenes=5):
    """
    Visualiza imágenes originales y sus versiones con filtro Sobel

    Args:
        train_X: matriz de características
        train_Y: DataFrame con labels
        clase_id: ID de la clase a visualizar
        num_imagenes: cuántas imágenes mostrar
    """
    indices = np.where(train_Y["label"] == clase_id)[0]
    indices_sel = np.random.choice(indices, num_imagenes, replace=False)

    clase_nombre = class_names[clase_id]

    fig, axes = plt.subplots(num_imagenes, 2, figsize=(10, 3 * num_imagenes))

    for i, idx in enumerate(indices_sel):
        img = train_X[idx].reshape(28, 28)

        # Filtros Sobel
        sx = ndimage.sobel(img, axis=0)
        sy = ndimage.sobel(img, axis=1)
        sobel_mag = np.sqrt(sx**2 + sy**2)

        # Original
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"{clase_nombre} - Original", fontweight="bold")
        axes[i, 0].axis("off")

        # Sobel
        axes[i, 1].imshow(sobel_mag, cmap="hot")
        axes[i, 1].set_title(f"{clase_nombre} - Sobel (Bordes)", fontweight="bold")
        axes[i, 1].axis("off")

    plt.suptitle(
        f"Comparación: Original vs Detección de Bordes (Sobel)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def visualizar_todas_clases(train_X, train_Y, num_por_clase=5):
    """
    Visualiza todas las clases en una sola imagen con los nombres de las clases en el eje Y

    Args:
        train_X: matriz de características
        train_Y: DataFrame con labels
        num_por_clase: cuántas imágenes por clase mostrar
    """
    num_clases = 10
    fig = plt.figure(figsize=(12, 22))

    # Crear grid: 1 columna para labels + num_por_clase columnas para imágenes
    gs = fig.add_gridspec(
        num_clases,
        num_por_clase + 1,
        width_ratios=[0.5] + [1] * num_por_clase,
        wspace=0.01,
        hspace=0.15,
        left=0.08,
        right=0.99,
        top=0.96,
        bottom=0.02,
    )

    for clase_id in range(num_clases):
        # Agregar label en la primera columna
        ax_label = fig.add_subplot(gs[clase_id, 0])
        ax_label.text(
            0.95,
            0.5,
            class_names[clase_id],
            fontweight="bold",
            fontsize=11,
            va="center",
            ha="right",
            transform=ax_label.transAxes,
        )
        ax_label.axis("off")

        # Agregar imágenes
        indices = np.where(train_Y["label"] == clase_id)[0]
        indices_sel = np.random.choice(indices, num_por_clase, replace=False)

        for col, idx in enumerate(indices_sel):
            ax_img = fig.add_subplot(gs[clase_id, col + 1])
            img = train_X[idx].reshape(28, 28)
            ax_img.imshow(img, cmap="gray")
            ax_img.axis("off")

    plt.suptitle(
        "Dataset Fashion-MNIST - Todas las clases",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.show()


def enumerar_labels_unicos(train_Y):
    print("\n" + "=" * 60)
    print("LABELS ÚNICOS EN EL DATASET DE ENTRENAMIENTO")
    print("=" * 60)

    labels_unicos = sorted(train_Y["label"].unique())

    for label in labels_unicos:
        class_name = class_names[label]
        count = (train_Y["label"] == label).sum()
        print(f"{label}: {class_name:20s} ({count:,} imágenes)")

    print("=" * 60)
    print(f"Total de clases: {len(labels_unicos)}")
    print(f"Total de imágenes: {len(train_Y):,}")
    print("=" * 60 + "\n")


# ============= MAIN =============


def main():
    print("=" * 60)
    print("VISUALIZADOR DE DATASET FASHION-MNIST")
    print("=" * 60)

    # Cargar datos
    print("\n--- Cargando datos ---")
    train_X, test_X, train_Y, test_Y = cargar_datos_locales(
        "train_X", "test_X", "train_Y", "test_Y"
    )

    print(f"\nDatos cargados:")
    print(f"  Train X shape: {train_X.shape}")
    print(f"  Test X shape: {test_X.shape}")
    print(f"  Train Y shape: {train_Y.shape}")
    print(f"  Test Y shape: {test_Y.shape}")

    # Menú de opciones
    while True:
        print("\n" + "=" * 60)
        print("SELECCIONA UNA OPCIÓN:")
        print("=" * 60)
        print("1. Visualizar clase específica (T-shirt)")
        print("2. Visualizar clase específica (Shirt)")
        print("3. Comparar T-shirt vs Shirt")
        print("4. Ver original vs Sobel (T-shirt)")
        print("5. Ver original vs Sobel (Shirt)")
        print("6. Visualizar todas las clases")
        print("7. Enumerar labels únicos de y_train")
        print("8. Salir")
        print("=" * 60)

        opcion = input("Ingresa tu opción (1-8): ").strip()

        if opcion == "1":
            print("\nVisualizando T-shirt (clase 0)...")
            visualizar_clase(train_X, train_Y, clase_id=0, num_imagenes=10)

        elif opcion == "2":
            print("\nVisualizando Shirt (clase 6)...")
            visualizar_clase(train_X, train_Y, clase_id=6, num_imagenes=10)

        elif opcion == "3":
            print("\nComparando T-shirt vs Shirt...")
            comparar_clases(train_X, train_Y, clase1_id=0, clase2_id=6, num_pares=10)

        elif opcion == "4":
            print("\nVisualizando T-shirt original vs Sobel...")
            visualizar_original_vs_sobel(train_X, train_Y, clase_id=0, num_imagenes=5)

        elif opcion == "5":
            print("\nVisualizando Shirt original vs Sobel...")
            visualizar_original_vs_sobel(train_X, train_Y, clase_id=6, num_imagenes=5)

        elif opcion == "6":
            print("\nVisualizando todas las clases...")
            visualizar_todas_clases(train_X, train_Y, num_por_clase=5)

        elif opcion == "7":
            enumerar_labels_unicos(train_Y)

        elif opcion == "8":
            print("\n¡Hasta luego!")
            break

        else:
            print("\n⚠️  Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
