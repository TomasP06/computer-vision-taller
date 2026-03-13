import numpy as np


def apply_radial_distortion(points: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """
    Aplica distorsión radial a un conjunto de puntos 2D normalizados.

    La distorsión radial modela la deformación introducida por la lente de la cámara.
    El modelo utilizado es:
        x_d = x * (1 + k1*r^2 + k2*r^4)
        y_d = y * (1 + k1*r^2 + k2*r^4)
    donde r^2 = x^2 + y^2.

    Args:
        points (np.ndarray): Array de forma (N, 2) con coordenadas normalizadas (x, y).
        k1 (float): Primer coeficiente de distorsión radial.
        k2 (float): Segundo coeficiente de distorsión radial.

    Returns:
        np.ndarray: Array de forma (N, 2) con los puntos distorsionados.

    Example:
        >>> pts = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> distorted = apply_radial_distortion(pts, k1=0.1, k2=0.01)
    """
    points = np.array(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]

    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2

    factor = 1 + k1 * r2 + k2 * r4

    x_distorted = x * factor
    y_distorted = y * factor

    return np.stack([x_distorted, y_distorted], axis=1)


def vary_focal_length(
    points_3d: np.ndarray,
    focal_lengths: list,
    cx: float = 0.0,
    cy: float = 0.0
) -> list:
    """
    Proyecta puntos 3D usando distintas longitudes focales y devuelve las proyecciones.

    Utiliza el modelo de proyección en perspectiva (pinhole):
        u = f * (X / Z) + cx
        v = f * (Y / Z) + cy

    Args:
        points_3d (np.ndarray): Array de forma (N, 3) con puntos 3D (X, Y, Z).
        focal_lengths (list): Lista de valores de longitud focal a probar.
        cx (float): Coordenada principal en x (por defecto 0).
        cy (float): Coordenada principal en y (por defecto 0).

    Returns:
        list: Lista de arrays (N, 2), uno por cada focal length, con los puntos proyectados.

    Example:
        >>> pts3d = np.array([[1.0, 2.0, 5.0], [-1.0, 0.5, 3.0]])
        >>> projections = vary_focal_length(pts3d, focal_lengths=[100, 200, 500])
    """
    points_3d = np.array(points_3d, dtype=np.float64)
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    projections = []
    for f in focal_lengths:
        u = f * (X / Z) + cx
        v = f * (Y / Z) + cy
        projections.append(np.stack([u, v], axis=1))

    return projections

