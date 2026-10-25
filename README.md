# Pixel CNN

Nuestra primer CNN paa validar como funciona:

[ ] Misión 1 · Detectar bordes

Utiliza un cuadrado. Compara el filtro vertical con el horizontal y explica qué zonas resaltan en cada caso.
Identifica qué características detecta cada kernel.


[ ] Misión 2 · Menos información

Modifica stride de 1 a 2. Compara las dimensiones de salida y observa qué detalles desaparecen.
Explica el efecto del desplazamiento del filtro.


[ ] Misión 3 · El problema de los bordes

Activa y desactiva padding. Compara el tamaño de los mapas y cómo se procesan los extremos de la imagen.
Comprende el propósito del padding.


[ ] Misión 4 · Pixel bajo presión

Activa el ruido. Compara los kernels e identifica cuáles suavizan la imagen y cuáles resaltan las variaciones.
Analiza la sensibilidad de los filtros al ruido.


[ ] Misión 5 · Reducir sin olvidar

Activa max pooling. Compara la matriz antes y después de reducirla, considerando que primero se aplica ReLU.
Identifica qué información se conserva y cuál se pierde.

## Para usar el repo

Clonarlo

Acceder

Sincronizar librerias:
```bash
uv sync
```

Ejecutar:
```bash
uv run main.py
```
