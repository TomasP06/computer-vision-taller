"""
cvtools.filters
===============
Módulo de convoluciones y filtros clásicos de visión por computador.

Funciones:
    - apply_convolution: convolución genérica con kernel arbitrario
    - sobel_x: filtro Sobel horizontal (detecta bordes verticales)
    - sobel_y: filtro Sobel vertical (detecta bordes horizontales)
    - canny_detector: detector de bordes Canny
    - laplacian_filter: filtro Laplaciano (realza cambios bruscos de intensidad)
"""

import cv2
import numpy as np


def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica una convolución genérica a una imagen usando un kernel arbitrario.

    Args:
        image (np.ndarray): Imagen de entrada en escala de grises o BGR.
        kernel (np.ndarray): Kernel de convolución (matriz 2D de flotantes).

    Returns:
        np.ndarray: Imagen resultante de la convolución, misma forma que la entrada.

    Example:
        >>> img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
        >>> blur_kernel = np.ones((5, 5), np.float32) / 25
        >>> blurred = apply_convolution(img, blur_kernel)
    """
    kernel = np.array(kernel, dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def sobel_x(image: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Sobel horizontal para detectar bordes verticales.

    El filtro Sobel en X resalta los cambios de intensidad en dirección horizontal,
    es decir, detecta bordes orientados verticalmente.
    Kernel Sobel X:
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]

    Args:
        image (np.ndarray): Imagen en escala de grises (H, W) o BGR (H, W, 3).
                            Si es BGR se convierte internamente a gris.

    Returns:
        np.ndarray: Imagen con bordes resaltados en dirección X, tipo uint8.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> edges_x = sobel_x(img)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    return cv2.convertScaleAbs(sx)


def sobel_y(image: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Sobel vertical para detectar bordes horizontales.

    El filtro Sobel en Y resalta los cambios de intensidad en dirección vertical,
    es decir, detecta bordes orientados horizontalmente.
    Kernel Sobel Y:
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]

    Args:
        image (np.ndarray): Imagen en escala de grises (H, W) o BGR (H, W, 3).
                            Si es BGR se convierte internamente a gris.

    Returns:
        np.ndarray: Imagen con bordes resaltados en dirección Y, tipo uint8.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> edges_y = sobel_y(img)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(sy)


def canny_detector(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Aplica el detector de bordes Canny a una imagen.

    El detector Canny es un algoritmo multi-etapa que incluye:
    suavizado gaussiano, cálculo de gradiente, supresión de no-máximos
    e histéresis mediante dos umbrales.

    Args:
        image (np.ndarray): Imagen de entrada en escala de grises o BGR.
                            Si es BGR se convierte internamente a gris.
        low (int): Umbral inferior para la histéresis (por defecto 50).
        high (int): Umbral superior para la histéresis (por defecto 150).

    Returns:
        np.ndarray: Imagen binaria (0 o 255) con los bordes detectados.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> edges = canny_detector(img, low=50, high=150)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Laplaciano a una imagen.

    El filtro Laplaciano calcula la segunda derivada de la intensidad.
    Resalta regiones donde la intensidad cambia abruptamente en cualquier dirección,
    por lo que es especialmente sensible a bordes y esquinas.
    Es isótropo: no depende de la orientación del borde.

    Kernel Laplaciano:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]

    Args:
        image (np.ndarray): Imagen de entrada en escala de grises o BGR.
                            Si es BGR se convierte internamente a gris.

    Returns:
        np.ndarray: Imagen con bordes y cambios bruscos de intensidad resaltados,
                    tipo uint8.

    Note:
        Resalta principalmente: bordes pronunciados, esquinas, cambios de textura bruscos.
        Es sensible al ruido; se recomienda suavizar antes de aplicarlo.

    Example:
        >>> img = cv2.imread('imagen.jpg')
        >>> lap = laplacian_filter(img)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)
