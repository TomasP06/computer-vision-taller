"""
cvtools.color
=============
Módulo de espacios de color y cuantización.

Funciones:
    - rgb_to_hsv: convierte imagen BGR a HSV
    - rgb_to_lab: convierte imagen BGR a LAB
    - plot_color_histogram: calcula y grafica histograma de colores
    - quantize_image: cuantización por K-means
    - reduce_image_weight: cuantiza y retorna tamaño en KB
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen de espacio BGR a HSV.

    Args:
        image (np.ndarray): Imagen en formato BGR (como carga OpenCV), forma (H, W, 3).

    Returns:
        np.ndarray: Imagen convertida al espacio HSV, misma forma que la entrada.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> hsv = rgb_to_hsv(img)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen de espacio BGR a CIE LAB.

    El espacio LAB separa la luminosidad (L) de la información de color (a, b),
    siendo más uniforme perceptualmente que BGR/RGB.

    Args:
        image (np.ndarray): Imagen en formato BGR, forma (H, W, 3).

    Returns:
        np.ndarray: Imagen convertida al espacio LAB, misma forma que la entrada.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> lab = rgb_to_lab(img)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def plot_color_histogram(image: np.ndarray) -> None:
    """
    Calcula y grafica el histograma de los tres canales de color (BGR) de una imagen.

    Args:
        image (np.ndarray): Imagen en formato BGR, forma (H, W, 3).

    Returns:
        None. Muestra la figura con matplotlib.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> plot_color_histogram(img)
    """
    colors = ('b', 'g', 'r')
    labels = ('Canal B (Azul)', 'Canal G (Verde)', 'Canal R (Rojo)')

    plt.figure(figsize=(10, 4))
    plt.title("Histograma de Colores - color.py", fontsize=13, fontweight='bold')

    for i, (color, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=label, linewidth=1.5)

    plt.xlabel("Intensidad de píxel (0–255)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def quantize_image(image: np.ndarray, n_colors: int) -> np.ndarray:
    """
    Aplica cuantización simple de colores usando K-means.

    Reduce la imagen a un número limitado de colores representativos.

    Args:
        image (np.ndarray): Imagen en formato BGR, forma (H, W, 3).
        n_colors (int): Número de colores destino (ej. 256, 64, 16).

    Returns:
        np.ndarray: Imagen cuantizada de misma forma (H, W, 3), dtype uint8.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> q16 = quantize_image(img, n_colors=16)
    """
    h, w = image.shape[:2]
    # Reshape a lista de píxeles
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        n_colors,
        None,
        criteria,
        attempts=5,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(h, w, 3)


def reduce_image_weight(image: np.ndarray, n_colors: int) -> tuple:
    """
    Cuantiza la imagen reduciendo el número de colores y calcula su tamaño en KB.

    Args:
        image (np.ndarray): Imagen en formato BGR, forma (H, W, 3).
        n_colors (int): Número de colores destino para la cuantización.

    Returns:
        tuple: (imagen_cuantizada, tamaño_kb) donde:
            - imagen_cuantizada (np.ndarray): Imagen cuantizada (H, W, 3).
            - tamaño_kb (float): Tamaño aproximado de la imagen cuantizada en KB
                                 (calculado como nbytes / 1024).

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> q_img, size_kb = reduce_image_weight(img, n_colors=16)
        >>> print(f"Nuevo tamaño: {size_kb:.2f} KB")
    """
    quantized = quantize_image(image, n_colors)
    size_kb = quantized.nbytes / 1024
    return quantized, size_kb
