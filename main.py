"""Pixel aprende a ver: laboratorio de convoluciones (CPU, NumPy + Pygame)."""

import numpy as np
import pygame

WIDTH, HEIGHT = 1160, 720
FPS = 30
BG = (13, 20, 34)
WHITE = (235, 241, 250)
MUTED = (157, 175, 198)
CYAN = (55, 210, 222)

KERNELS = {
    "Vertical": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    "Horizontal": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
    "Desenfoque": np.ones((3, 3), dtype=float) / 9,
    "Realce": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
}
SHAPES = ("Cuadrado", "Cruz", "Diagonal", "Circulo")


def make_image(name, noisy):
    """Construye una imagen 16 x 16 en escala de grises, sin dataset externo."""
    image = np.zeros((16, 16), dtype=float)
    yy, xx = np.indices(image.shape)
    if name == "Cuadrado":
        image[4:12, 4:12] = 1.0
    elif name == "Cruz":
        image[3:13, 7:9] = 1.0
        image[7:9, 3:13] = 1.0
    elif name == "Diagonal":
        image[np.abs(yy - xx) <= 1] = 1.0
    elif name == "Circulo":
        distance = (xx - 7.5) ** 2 + (yy - 7.5) ** 2
        image[(distance >= 16) & (distance <= 29)] = 1.0
    if noisy:
        rng = np.random.default_rng(42)
        image = np.clip(image + rng.normal(0, 0.22, image.shape), 0, 1)
    return image


def convolve(image, kernel, stride, padding):
    """Correlacion cruzada usada habitualmente en capas CNN; no invierte el kernel."""
    padded = np.pad(image, padding, mode="constant")
    rows = (padded.shape[0] - 3) // stride + 1
    cols = (padded.shape[1] - 3) // stride + 1
    result = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            patch = padded[r * stride : r * stride + 3, c * stride : c * stride + 3]
            result[r, c] = np.sum(patch * kernel)
    return result


def max_pool(image):
    """Max pooling 2 x 2 con stride 2; descarta bordes impares."""
    rows, cols = image.shape
    result = np.zeros((rows // 2, cols // 2), dtype=float)
    for r in range(rows // 2):
        for c in range(cols // 2):
            result[r, c] = np.max(image[2 * r : 2 * r + 2, 2 * c : 2 * c + 2])
    return result


def write(surface, text, x, y, font, color=WHITE):
    surface.blit(font.render(text, True, color), (x, y))


def draw_grid(surface, matrix, x, y, size=305, signed=False):
    """Muestra valores, normalizados solo para el color (no altera los cálculos)."""
    rows, cols = matrix.shape
    cell = min(size / cols, size / rows)
    scale = max(float(np.max(np.abs(matrix))), 1e-9) if signed else 1.0
    for r in range(rows):
        for c in range(cols):
            value = float(matrix[r, c])
            if signed:
                strength = min(abs(value) / scale, 1.0)
                color = (
                    (int(30 + 210 * strength), int(51 + 100 * (1 - strength)), 85)
                    if value < 0
                    else (20, int(50 + 175 * strength), int(72 + 165 * strength))
                )
            else:
                strength = float(np.clip(value, 0, 1))
                color = (
                    int(27 + 140 * strength),
                    int(44 + 178 * strength),
                    int(68 + 180 * strength),
                )
            rect = pygame.Rect(
                round(x + c * cell),
                round(y + r * cell),
                max(1, round(cell) - 1),
                max(1, round(cell) - 1),
            )
            pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(
        surface,
        MUTED,
        (x - 1, y - 1, round(cols * cell) + 2, round(rows * cell) + 2),
        1,
    )
    return cell


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PIXEL | Laboratorio CNN - clase 1")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("dejavusans", 31, bold=True)
    font = pygame.font.SysFont("dejavusans", 19)
    small = pygame.font.SysFont("dejavusans", 16)
    shape_idx = 0
    kernel_idx = 0
    stride = 1
    padding = 0
    noisy = False
    pooling = True
    show_scan = True
    step = 0
    frames = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    shape_idx = event.key - pygame.K_1
                    step = 0
                elif event.key == pygame.K_k:
                    kernel_idx = (kernel_idx + 1) % len(KERNELS)
                elif event.key == pygame.K_s:
                    stride = 2 if stride == 1 else 1
                elif event.key == pygame.K_p:
                    padding = 1 if padding == 0 else 0
                elif event.key == pygame.K_n:
                    noisy = not noisy
                elif event.key == pygame.K_m:
                    pooling = not pooling
                elif event.key == pygame.K_SPACE:
                    show_scan = not show_scan
                elif event.key == pygame.K_RIGHT:
                    step += 1
                elif event.key == pygame.K_LEFT:
                    step -= 1

        image = make_image(SHAPES[shape_idx], noisy)
        kernel_name = tuple(KERNELS)[kernel_idx]
        kernel = KERNELS[kernel_name]
        features = convolve(image, kernel, stride, padding)
        activated = np.maximum(features, 0)  # ReLU: elimina las activaciones negativas.
        pooled = max_pool(activated)
        result = pooled if pooling else activated
        out_rows, out_cols = features.shape
        total = out_rows * out_cols
        frames += 1
        if show_scan and frames % 12 == 0:
            step += 1
        step %= total
        scan_r, scan_c = divmod(step, out_cols)

        screen.fill(BG)
        write(screen, "PIXEL: aprende a ver", 28, 17, title_font, CYAN)
        write(
            screen,
            "Experimento: un filtro transforma pixeles en un mapa de caracteristicas",
            30,
            63,
            small,
            MUTED,
        )

        labels = [
            ("IMAGEN ORIGINAL", 28),
            ("CONVOLUCION (sin ReLU)", 415),
            ("ReLU + POOLING" if pooling else "ReLU (sin pooling)", 802),
        ]
        for label, x in labels:
            write(screen, label, x, 120, font)
        cell = draw_grid(screen, image, 28, 160, signed=False)
        draw_grid(screen, features, 415, 160, signed=True)
        draw_grid(screen, result, 802, 160, signed=False)

        # Cuadro: ventana 3 x 3 sobre la imagen original.
        top_r, top_c = scan_r * stride, scan_c * stride
        if padding == 0:
            left, top = 28 + top_c * cell, 160 + top_r * cell
        else:
            left, top = 28 + (top_c - padding) * cell, 160 + (top_r - padding) * cell
        pygame.draw.rect(
            screen,
            (255, 194, 65),
            (round(left), round(top), round(3 * cell), round(3 * cell)),
            3,
        )
        write(
            screen, f"Posicion de salida: ({scan_r}, {scan_c})", 30, 485, small, MUTED
        )
        write(
            screen,
            f"Valor en esa posicion: {features[scan_r, scan_c]:.2f}",
            416,
            485,
            small,
            MUTED,
        )
        write(
            screen,
            f"Tamano final: {result.shape[0]} x {result.shape[1]}",
            804,
            485,
            small,
            MUTED,
        )

        pygame.draw.line(screen, (66, 88, 112), (28, 525), (1130, 525), 1)
        write(
            screen,
            f"Figura: {SHAPES[shape_idx]} | Kernel: {kernel_name} | Stride: {stride} | Padding: {padding}",
            29,
            543,
            font,
        )
        write(
            screen,
            f"Ruido: {'SI' if noisy else 'NO'} | Pooling: {'SI' if pooling else 'NO'} | Animacion: {'SI' if show_scan else 'PAUSA'}",
            29,
            575,
            font,
        )
        write(
            screen,
            "1-4 figura   K filtro   S stride   P padding   N ruido   M pooling",
            30,
            623,
            small,
            CYAN,
        )
        write(
            screen,
            "ESPACIO pausa   Flechas paso a paso   ESC salir",
            30,
            654,
            small,
            CYAN,
        )
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()
