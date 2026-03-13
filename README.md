# cvtools — Librería de Visión por Computador

Librería modular en Python para tareas fundamentales de visión por computador.

## Estructura

```
cvtools/
├── __init__.py
├── camera.py     # Modelo de cámara y proyecciones
├── color.py      # Espacios de color y cuantización
└── filters.py    # Convoluciones y filtros clásicos

tests/
├── test_camera.py
├── test_color.py
└── test_filters.py

data/             # Imágenes de prueba
main.py           # Script demostrativo
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```python
from cvtools.camera import apply_radial_distortion, vary_focal_length
from cvtools.color import rgb_to_hsv, rgb_to_lab, plot_color_histogram, quantize_image, reduce_image_weight
from cvtools.filters import apply_convolution, sobel_x, sobel_y, canny_detector, laplacian_filter
```

## Ejecutar demo

```bash
python main.py
```

## Ejecutar tests

```bash
python -m pytest tests/ -v
```

## Módulos

### camera.py
- `apply_radial_distortion(points, k1, k2)` — Aplica distorsión radial con parámetros k₁, k₂
- `vary_focal_length(points_3d, focal_lengths, cx, cy)` — Proyecta puntos con distintas focales

### color.py
- `rgb_to_hsv(image)` — Convierte BGR→HSV
- `rgb_to_lab(image)` — Convierte BGR→LAB
- `plot_color_histogram(image)` — Grafica histograma de colores
- `quantize_image(image, n_colors)` — Cuantización por K-means
- `reduce_image_weight(image, n_colors)` — Reduce peso y retorna tamaño en KB

### filters.py
- `apply_convolution(image, kernel)` — Convolución genérica
- `sobel_x(image)` — Filtro Sobel horizontal
- `sobel_y(image)` — Filtro Sobel vertical
- `canny_detector(image, low, high)` — Detector de bordes Canny
- `laplacian_filter(image)` — Filtro Laplaciano (resalta cambios bruscos de intensidad)
