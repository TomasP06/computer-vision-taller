# Carpeta de imágenes de prueba

Coloca aquí las imágenes que quieras usar con `cvtools`.

## Formatos soportados
- `.jpg` / `.jpeg`
- `.png`
- `.bmp`

## Uso en main.py
El script `main.py` busca automáticamente la primera imagen válida en esta carpeta.
Si no encuentra ninguna, usa una imagen sintética de prueba.

## Ejemplos de uso manual
```python
import cv2
from cvtools.color import rgb_to_hsv, plot_color_histogram

img = cv2.imread('data/tu_imagen.jpg')
hsv = rgb_to_hsv(img)
plot_color_histogram(img)
```
