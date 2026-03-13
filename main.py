"""
main.py - Script demostrativo de la libreria cvtools

Usa imagen1.jpg e imagen2.jpg de la carpeta data/ para demostrar
todas las funciones de los modulos camera, color y filters.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Importar modulos de cvtools
from cvtools.camera import apply_radial_distortion, vary_focal_length
from cvtools.color import (
    rgb_to_hsv, rgb_to_lab,
    plot_color_histogram,
    quantize_image, reduce_image_weight
)
from cvtools.filters import (
    apply_convolution,
    sobel_x, sobel_y,
    canny_detector, laplacian_filter
)

# Rutas fijas a las imagenes
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMG1_PATH = os.path.join(DATA_DIR, 'imagen1.jpg')
IMG2_PATH = os.path.join(DATA_DIR, 'imagen2.jpg')


def load_images():
    """Carga imagen1.jpg e imagen2.jpg desde data/."""
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    if img1 is None:
        raise FileNotFoundError(f"No se encontro: {IMG1_PATH}")
    if img2 is None:
        raise FileNotFoundError(f"No se encontro: {IMG2_PATH}")
    print(f"[main] imagen1.jpg cargada: {img1.shape}")
    print(f"[main] imagen2.jpg cargada: {img2.shape}")
    return img1, img2


# ==============================================================
# DEMO 1 - camera.py
# ==============================================================
def demo_camera():
    print("\n" + "="*60)
    print("  DEMO: cvtools.camera")
    print("="*60)

    # Grilla de puntos para distorsion radial
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distorsion Radial - cvtools.camera", fontsize=13, fontweight='bold')

    configs = [
        (0.0,  0.0,  "Sin distorsion\n(k1=0, k2=0)"),
        (0.4,  0.05, "Barril\n(k1=0.4, k2=0.05)"),
        (-0.4, 0.05, "Cojin\n(k1=-0.4, k2=0.05)"),
    ]
    for ax, (k1, k2, title) in zip(axes, configs):
        d = apply_radial_distortion(points, k1, k2)
        ax.scatter(d[:, 0], d[:, 1], s=12, color='steelblue', alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Variacion de longitud focal
    np.random.seed(42)
    pts3d = np.column_stack([
        np.random.uniform(-2, 2, 40),
        np.random.uniform(-2, 2, 40),
        np.random.uniform(3, 8, 40),
    ])
    focal_lengths = [100, 300, 700]
    projections = vary_focal_length(pts3d, focal_lengths, cx=320, cy=240)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Variacion de Longitud Focal - cvtools.camera", fontsize=13, fontweight='bold')
    for ax, f, proj in zip(axes, focal_lengths, projections):
        ax.scatter(proj[:, 0], proj[:, 1], s=30, color='tomato', alpha=0.8)
        ax.set_title(f"f = {f} px")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("[camera] OK - Distorsion radial y variacion de focal mostradas.")


# ==============================================================
# DEMO 2 - color.py  (usa imagen1.jpg)
# ==============================================================
def demo_color(image):
    print("\n" + "="*60)
    print("  DEMO: cvtools.color  (imagen1.jpg)")
    print("="*60)

    hsv = rgb_to_hsv(image)
    lab = rgb_to_lab(image)
    print(f"[color] Shape original: {image.shape}")
    print(f"[color] Shape HSV:      {hsv.shape}")
    print(f"[color] Shape LAB:      {lab.shape}")

    # Conversion de espacios de color
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Conversion de Espacios de Color - cvtools.color", fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (RGB)")
    axes[1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[1].set_title("HSV")
    axes[2].imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    axes[2].set_title("LAB")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Histograma de colores
    print("[color] Mostrando histograma de colores...")
    plot_color_histogram(image)

    # Cuantizacion con distintos numeros de colores
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle("Cuantizacion de Color (K-means) - cvtools.color", fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    for ax, nc in zip(axes[1:], [64, 16, 4]):
        q, kb = reduce_image_weight(image, n_colors=nc)
        ax.imshow(cv2.cvtColor(q, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nc} colores\n({kb:.1f} KB)")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("[color] OK - Conversiones, histograma y cuantizacion mostrados.")


# ==============================================================
# DEMO 3 - filters.py  (usa imagen2.jpg)
# ==============================================================
def demo_filters(image):
    print("\n" + "="*60)
    print("  DEMO: cvtools.filters  (imagen2.jpg)")
    print("="*60)

    # Kernel de enfoque (sharpening)
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    sharpened = apply_convolution(image, sharpen_kernel)

    sx  = sobel_x(image)
    sy  = sobel_y(image)
    can = canny_detector(image, low=50, high=150)
    lap = laplacian_filter(image)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Filtros y Convoluciones - cvtools.filters", fontsize=13, fontweight='bold')

    datos = [
        ("Original (gris)",          cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 'gray'),
        ("Convolucion - Sharpening",  cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB), None),
        ("Sobel X (bordes vert.)",    sx,  'gray'),
        ("Sobel Y (bordes horiz.)",   sy,  'gray'),
        ("Canny (detector bordes)",   can, 'gray'),
        ("Laplaciano (intensidad)",   lap, 'gray'),
    ]

    for ax, (title, img_data, cmap) in zip(axes.ravel(), datos):
        ax.imshow(img_data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print("[filters] OK - Todos los filtros aplicados y mostrados.")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 52)
    print("  cvtools - Demo Principal (main.py)")
    print("=" * 52)

    img1, img2 = load_images()

    demo_camera()           # usa puntos sinteticos (no imagen)
    demo_color(img1)        # imagen1.jpg
    demo_filters(img2)      # imagen2.jpg

    print("\n[OK] Demo completado. Todos los modulos funcionaron correctamente.")
