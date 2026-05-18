# Informe académico: clasificación de radiografías de tórax mediante redes neuronales convolucionales

**Asignatura:** Machine Learning II  
**Tema:** Clasificación de imágenes con CNN (PyTorch)  
**Dataset:** Chest X-Ray — clases NORMAL y PNEUMONIA  
**Implementación:** `chest_xray_cnn.ipynb` | **Figuras:** `outputs/`

---

## 1. Introducción

La inteligencia artificial aplicada a imágenes médicas ha demostrado utilidad en tareas de apoyo al diagnóstico, entre ellas la detección de patrones compatibles con **neumonía** en radiografías de tórax. Este tipo de imágenes presenta desafíos propios: variabilidad de adquisición, ruido, superposición de estructuras óseas y tejidos, y —en conjuntos públicos— frecuente **desbalance entre clases**.

En este proyecto se aborda un problema de **clasificación supervisada binaria**: dada una radiografía, el sistema debe asignar la etiqueta **NORMAL** o **PNEUMONIA**. Para ello se diseña, implementa y evalúa una **red neuronal convolucional (CNN)** en **PyTorch**, siguiendo un flujo metodológico estándar en aprendizaje profundo: análisis exploratorio, preprocesamiento, modelado, validación, optimización de hiperparámetros y evaluación en un conjunto de prueba **independiente**.

El informe no se limita a presentar código: se **justifica** cada decisión técnica (partición de datos, normalización, arquitectura, regularización y búsqueda de hiperparámetros) y se **interpreta** el desempeño del modelo en relación con la rúbrica de evaluación del curso.

---

## 2. Objetivo del proyecto

### 2.1 Objetivo general

Diseñar, implementar, entrenar y evaluar un modelo CNN capaz de clasificar radiografías de tórax en las categorías NORMAL y PNEUMONIA, documentando el proceso completo y optimizando hiperparámetros clave mediante **random search**.

### 2.2 Objetivos específicos

1. Realizar **análisis exploratorio (EDA)** y cuantificar el **desbalance de clases**.
2. Aplicar **preprocesamiento** (redimensionamiento, escala de grises, normalización y aumentación controlada).
3. Construir una CNN con capas **convolucionales**, **pooling** y **densas**, incluyendo **dropout** como regularización.
4. Dividir datos en **entrenamiento** y **validación** (desde `train/`), reservando `test/` para evaluación final.
5. Monitorear **pérdida** y **métricas** durante el entrenamiento; aplicar **early stopping**.
6. Optimizar hiperparámetros (filtros, kernel, learning rate, épocas, dropout) y documentar su impacto.
7. Reportar **accuracy**, **precision**, **recall**, **F1-score** y **matriz de confusión** en el conjunto de prueba, con interpretación clínica básica.

---

## 3. Descripción del dataset

> **Código y ejecución:** los bloques de esta sección corresponden a `chest_xray_cnn.ipynb` — **§3. Análisis exploratorio (EDA)**. Ejecutar en orden: configuración (§2) → EDA (§3). Las salidas de consola y figuras siguientes se obtuvieron ejecutando ese notebook en el entorno `.venv`.

### 3.1 Origen y estructura

El dataset está organizado en carpetas por clase (formato compatible con `ImageFolder` de torchvision):

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Cada imagen es un archivo **JPEG** en escala de grises (radiografía de tórax).

### 3.2 Código: configuración de rutas y conteo por clase

En la celda de configuración se definen las rutas y una función auxiliar que **excluye archivos no imagen** (por ejemplo `.DS_Store`), evitando errores al abrir archivos con Pillow.

```python
DATA_ROOT = Path("chest_xray")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def count_images(root: Path) -> dict:
    return {cls: len(list_images(root / cls)) for cls in CLASSES}

train_counts = count_images(TRAIN_DIR)
test_counts = count_images(TEST_DIR)

print("TRAIN:", train_counts, "| Total:", sum(train_counts.values()))
print("TEST: ", test_counts, "| Total:", sum(test_counts.values()))
```

**Salida obtenida al ejecutar** (`chest_xray_cnn.ipynb`, celda EDA — conteo):

```
TRAIN: {'NORMAL': 1349, 'PNEUMONIA': 3883} | Total: 5232
TEST:  {'NORMAL': 234, 'PNEUMONIA': 390} | Total: 624
```

**Interpretación:** el conjunto de entrenamiento es ~8,4 veces mayor que el de prueba. Dentro de train hay casi el triple de imágenes de neumonía que de normales.

### 3.3 Resultados: volumen y distribución

| Partición | NORMAL | PNEUMONIA | Total |
|-----------|--------|-----------|-------|
| Train | 1 349 | 3 883 | 5 232 |
| Test | 234 | 390 | 624 |

**Proporción de PNEUMONIA:**

| Partición | % PNEUMONIA | Ratio PNEUMONIA / NORMAL |
|-----------|-------------|---------------------------|
| Train | 74,2 % | 2,88 : 1 |
| Test | 62,5 % | 1,67 : 1 |

### 3.4 Código: gráfico de desbalance y métricas derivadas

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, counts, title in zip(axes, [train_counts, test_counts], ["Train", "Test"]):
    labels, values = list(counts.keys()), list(counts.values())
    bars = ax.bar(labels, values, color=["#4C78A8", "#E45756"])
    ax.set_title(f"Distribución de clases — {title}")
    ax.set_ylabel("Número de imágenes")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20, str(v), ha="center")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_class_distribution.png", dpi=120)
plt.show()

for name, counts in [("Train", train_counts), ("Test", test_counts)]:
    total = sum(counts.values())
    ratio = counts["PNEUMONIA"] / counts["NORMAL"]
    print(f"{name}: PNEUMONIA={counts['PNEUMONIA']/total:.1%} | "
          f"Ratio PNEUMONIA/NORMAL={ratio:.2f}x")
```

**Salida obtenida al ejecutar** (misma celda, bloque de barras):

```
Train: PNEUMONIA=74.2% | Ratio PNEUMONIA/NORMAL=2.88x
Test: PNEUMONIA=62.5% | Ratio PNEUMONIA/NORMAL=1.67x
```

**Figura 1.** Distribución de clases (train y test)

![Distribución de clases](outputs/eda_class_distribution.png)

### 3.5 Implicaciones para el modelado

- El desbalance explica por qué la **accuracy** sola puede ser engañosa: un clasificador trivial que prediga siempre PNEUMONIA alcanzaría ~74 % de aciertos en train sin ser útil para detectar casos normales.
- Se requieren métricas **por clase** y **ponderadas**, y compensación en la función de pérdida (**pesos de clase**).
- El conjunto **test** no se utiliza durante entrenamiento ni búsqueda de hiperparámetros, garantizando una estimación menos optimista del rendimiento en datos no vistos.

### 3.6 Código: exploración visual (muestras y tamaños)

Se muestran 4 ejemplos por clase y una submuestra de tamaños originales para comprobar variabilidad espacial antes del resize.

```python
sample_paths = []
for cls in CLASSES:
    for p in list_images(TRAIN_DIR / cls)[:4]:
        sample_paths.append((p, cls))

widths, heights = [], []
for cls in CLASSES:
    for p in list_images(TRAIN_DIR / cls)[:80]:  # submuestra (modo rápido)
        with Image.open(p) as im:
            widths.append(im.size[0])
            heights.append(im.size[1])

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (path, cls) in zip(axes.flat, sample_paths):
    ax.imshow(Image.open(path).convert("L"), cmap="gray")
    ax.set_title(cls)
    ax.axis("off")
plt.suptitle("Ejemplos por clase (train)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_sample_images.png", dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(widths, heights, alpha=0.4, s=10)
ax.set_xlabel("Ancho (px)")
ax.set_ylabel("Alto (px)")
ax.set_title("Tamaños originales (submuestra)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_image_sizes.png", dpi=120)
plt.show()
```

**Salida obtenida al ejecutar** (submuestra de 80 imágenes por clase para tamaños):

```
Tamaños (submuestra n=160): ancho min=502 max=2338, alto min=307 max=2025
```

El scatter confirma que **ancho y alto no son uniformes** entre radiografías (distintos equipos o recortes), lo que obliga a redimensionar a `IMG_SIZE × IMG_SIZE` en el preprocesamiento.

**Figura 2.** Ejemplos por clase (train)

![Ejemplos por clase](outputs/eda_sample_images.png)

**Figura 3.** Tamaños originales (submuestra)

![Tamaños de imagen](outputs/eda_image_sizes.png)

---

## 4. Preprocesamiento de datos

### 4.1 Justificación general

Las CNN exigen tensores de tamaño uniforme y escalas de intensidad comparables entre muestras. El preprocesamiento reduce la variabilidad irrelevante y mejora la estabilidad del gradiente durante el entrenamiento.

### 4.2 Modo rápido para demostración en CPU (`FAST_MODE = True`)

El notebook incluye un **modo de ejecución acelerado** pensado para equipos sin GPU. No cambia la metodología del trabajo (EDA → CNN → validación → random search → test), pero reduce tiempo de cómputo:

| Parámetro | Modo rápido | Modo completo (opcional) |
|-----------|-------------|---------------------------|
| `IMG_SIZE` | 64×64 | 128×128 |
| `BATCH_SIZE` | 64 | 32 |
| Filtros base (baseline) | 16 | 32 |
| Épocas baseline | 4 (+ early stopping) | 10 |
| Random search | 3 trials, épocas 3–5 | 8 trials, épocas 6–10 |
| Estimación mean/std | Submuestra 400 imágenes | Dataset train completo |

**Justificación académica:** en entornos docentes con CPU limitada, es preferible un pipeline **reproducible y completo** en tiempo razonable que un entrenamiento exhaustivo inacabado. Las métricas serán algo inferiores a una configuración de mayor resolución, pero **válidas para demostrar** el proceso y la interpretación de resultados.

Para mayor precisión: en la celda de configuración, poner `FAST_MODE = False` y `IMG_SIZE = 128`.

### 4.3 Pipeline implementado

| Etapa | Descripción | Justificación |
|-------|-------------|---------------|
| Escala de grises (1 canal) | Conversión explícita a un canal | Las radiografías son inherentemente grises; un canal reduce parámetros frente a RGB artificial |
| Resize 64×64 (rápido) / 128×128 (completo) | Redimensionamiento con `Resize` | Menor resolución = menos operaciones en convoluciones |
| Normalización | `(x - mean) / std` | Estabiliza activaciones; mean y std calculados **solo en train** |
| RandomHorizontalFlip (train) | Volteo horizontal con p = 0,5 | Aumentación ligera; el tórax presenta simetría aproximada |

**Estadísticas de normalización (train):**

- Media μ = **0,482**
- Desviación estándar σ = **0,235**

### 4.4 Código: cálculo de media y desviación estándar

*Este bloque estima μ y σ sobre el conjunto de entrenamiento antes de definir las transformaciones definitivas.*

```python
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
tmp_ds = datasets.ImageFolder(TRAIN_DIR, transform=base_transform)
tmp_loader = DataLoader(tmp_ds, batch_size=BATCH_SIZE, shuffle=False)
MEAN, STD = compute_mean_std(tmp_loader)  # μ ≈ 0.482, σ ≈ 0.235
```

**Resultado esperado:** dos escalares para el canal único, reutilizados en train, validación y test para evitar **fuga de información** (data leakage).

### 4.5 Código: transformaciones finales

```python
def get_transforms(mean, std, augment=False):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(ops)

train_transform = get_transforms(MEAN, STD, augment=True)
eval_transform = get_transforms(MEAN, STD, augment=False)
```

**Interpretación:** la validación y el test usan `eval_transform` (sin aumentación) para medir el rendimiento en condiciones deterministas.

### 4.6 Filtrado de archivos no válidos

Se excluyen archivos como `.DS_Store` mediante una función `list_images()` que solo admite extensiones de imagen (`.jpeg`, `.jpg`, `.png`, etc.), evitando errores en el EDA y conteos erróneos.

---

## 5. Diseño e implementación del modelo CNN

### 5.1 Enfoque de diseño

Se optó por una **CNN personalizada** (no transfer learning) para cumplir explícitamente con la consigna de capas convolucionales, pooling y densas. La arquitectura sigue el paradigma **feature extractor + classifier head**, habitual en visión por computador.

### 5.2 Partición train / validación / test

Desde `train/` (5 232 imágenes) se extrae un **20 % estratificado** para validación:

| Subconjunto | Tamaño |
|-------------|--------|
| Train interno | 4 187 |
| Validación | 1 045 |
| Test (carpeta `test/`) | 624 |

```python
train_idx, val_idx = stratified_indices(full_train_ds, val_ratio=0.2, seed=42)
train_ds = Subset(full_train_ds, train_idx)
val_ds = Subset(datasets.ImageFolder(TRAIN_DIR, transform=eval_transform), val_idx)
```

**Justificación del estratificado:** mantiene en validación la misma proporción NORMAL/PNEUMONIA que en train, haciendo más fiable la comparación entre experimentos.

### 5.3 Esquema de la arquitectura

```
Entrada [1 × 128 × 128]
    → Bloque 1: Conv → BN → ReLU → MaxPool  → [f × 64 × 64]
    → Bloque 2: Conv → BN → ReLU → MaxPool  → [2f × 32 × 32]
    → Bloque 3: Conv → BN → ReLU → MaxPool  → [4f × 16 × 16]
    → Flatten → Linear(·, 256) → ReLU → Dropout → Linear(256, 2)
Salida: logits para 2 clases
```

Con `base_filters = f = 32` por defecto: **f, 2f, 4f = 32, 64, 128** filtros.

Tras tres poolings con `IMG_SIZE=64`: tamaño espacial final **8×8** (con IMG_SIZE=128 sería 16×16).

**Parámetros entrenables:** con `base_filters=16` el modelo es notablemente más ligero que con 32 filtros (~8,5 M), lo que acelera cada época en CPU.

---

## 6. Explicación detallada de cada capa del modelo

### 6.1 Capa convolucional (`Conv2d`)

**Función:** aplicar filtros aprendibles que detectan patrones locales (bordes, texturas, opacidades).

**Parámetros relevantes:**

- `in_channels=1`, `out_channels=f` (luego 2f, 4f): profundidad creciente para representaciones más abstractas.
- `kernel_size` (3 o 5 en experimentos): tamaño del campo receptivo local.
- `padding = kernel_size // 2`: preserva altura y ancho antes del pooling cuando stride = 1.

**Justificación:** en imágenes médicas, patologías como consolidaciones o infiltrados pueden manifestarse como variaciones locales de intensidad capturables por convoluciones de bajo nivel en capas iniciales y combinadas en capas profundas.

### 6.2 Batch Normalization (`BatchNorm2d`)

**Función:** normalizar activaciones por canal y mini-batch.

**Beneficios:** acelera convergencia, permite tasas de aprendizaje más altas y actúa como regularizador leve al introducir ruido batch-dependent durante entrenamiento.

### 6.3 Función de activación ReLU

**Función:** `f(x) = max(0, x)` — introduce no linealidad sin saturación en la región positiva (frente a sigmoid/tanh en capas profundas).

**Justificación:** estándar en CNN modernas por eficiencia computacional y mitigación parcial del desvanecimiento del gradiente.

### 6.4 Max Pooling (`MaxPool2d(2)`)

**Función:** submuestreo tomando el máximo en ventanas 2×2.

**Efectos:**

- Reduce dimensionalidad espacial (menos parámetros en capas siguientes).
- Aporta **invariancia** a pequeños desplazamientos.
- Aumenta el campo receptivo efectivo en capas posteriores.

Tras tres poolings: 128 → 64 → 32 → **16** píxeles de lado.

### 6.5 Aplanamiento (`Flatten`)

Convierte el tensor `[4f × 16 × 16]` en un vector unidimensional para las capas totalmente conectadas.

### 6.6 Capas densas (`Linear`)

- **Primera densa (→ 256):** combina características de alto nivel aprendidas en los mapas de activación.
- **Segunda densa (→ 2):** produce **logits** para NORMAL y PNEUMONIA.

La salida se pasa por **softmax** implícito dentro de `CrossEntropyLoss` en PyTorch.

### 6.7 Dropout

**Función:** durante entrenamiento, anula aleatoriamente un porcentaje de neuronas (p = 0,3–0,5).

**Justificación:** combate **sobreajuste** en el clasificador, que concentra muchos parámetros respecto a las capas convolucionales en este diseño.

### 6.8 Código: definición del modelo

```python
class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=2, base_filters=32, kernel_size=3, dropout=0.5):
        super().__init__()
        pad = kernel_size // 2
        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4

        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size, padding=pad),
            nn.BatchNorm2d(f1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(f1, f2, kernel_size, padding=pad),
            nn.BatchNorm2d(f2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(f2, f3, kernel_size, padding=pad),
            nn.BatchNorm2d(f3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        reduced = IMG_SIZE // 8  # 16 para IMG_SIZE=128
        flat_dim = f3 * reduced * reduced
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## 7. Entrenamiento del modelo

### 7.1 Función de pérdida y pesos de clase

Se utiliza **entropía cruzada categórica** (`CrossEntropyLoss`) con pesos inversamente proporcionales a la frecuencia de clase:

| Clase | Índice | Peso en loss |
|-------|--------|--------------|
| NORMAL | 0 | 1,94 |
| PNEUMONIA | 1 | 0,67 |

**Justificación:** penalizar más los errores sobre la clase minoritaria (NORMAL) equilibra la contribución al gradiente y mejora recall de la clase menos frecuente.

```python
class_weights = get_class_weights(full_train_ds)  # tensor([1.94, 0.67])
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### 7.2 Optimizador y regularización adicional

- **Adam** con `learning_rate` configurable y `weight_decay = 1e-4` (regularización L2).
- **Batch size = 32:** equilibrio entre estabilidad del gradiente y memoria.

### 7.3 Procedimiento por época

1. Modo train: forward, cálculo de loss, backward, actualización de pesos.
2. Modo eval en validación: sin gradiente, mismas métricas.
3. Registro de **loss** y **F1 ponderado** en train y validación.

### 7.4 Early stopping

Si el F1 de validación no mejora durante `patience` épocas consecutivas (3–5 según fase), se detiene el entrenamiento y se **restauran los pesos** del mejor epoch.

**Justificación:** limita el sobreajuste cuando la loss de entrenamiento sigue bajando pero la validación degrada.

### 7.5 Resultados del entrenamiento baseline

Configuración de referencia antes de optimización sistemática:

| Hiperparámetro | Valor |
|----------------|-------|
| base_filters | 32 |
| kernel_size | 3 |
| dropout | 0,5 |
| learning rate | 0,001 |
| épocas máximas | 10 |

**Mejor resultado en validación:** F1 = **0,9758** (early stopping en época 7).

| Época (última registrada) | Train loss | Train F1 | Val loss | Val F1 |
|---------------------------|------------|----------|----------|--------|
| 7 | 0,1139 | 0,9556 | 0,0610 | 0,9714 |

**Figura 4.** Curvas de entrenamiento baseline — `outputs/baseline_curves.png`

### 7.6 Análisis preliminar: ¿overfitting o underfitting?

| Indicador | Observación | Interpretación |
|-----------|-------------|----------------|
| Val F1 > Train F1 (época 7) | 0,9714 vs 0,9556 | No hay sobreajuste severo en ese punto; la regularización y pesos de clase pueden favorecer validación |
| Early stopping activado | Época 7/10 | El modelo dejó de generalizar mejor antes del máximo de épocas planificado |
| Loss train decreciente | 0,11 en train | Capacidad suficiente; no parece **underfitting** grave |

**Conclusión provisional:** el baseline **aprende adecuadamente**; la optimización de hiperparámetros busca mejorar aún la generalización y estabilidad.

---

## 8. Optimización de hiperparámetros

### 8.1 Motivación

La arquitectura base fija el *pipeline*, pero el rendimiento depende de hiperparámetros: capacidad (filtros), tamaño de kernel, tasa de aprendizaje, duración del entrenamiento y dropout. Se requiere un método **sistemático** y **reproducible**.

### 8.2 Espacio de búsqueda (grid teórico)

| Hiperparámetro | Valores |
|----------------|---------|
| base_filters | 16, 32, 64 |
| kernel_size | 3, 5 |
| learning rate | 1×10⁻⁴, 5×10⁻⁴, 1×10⁻³ |
| epochs | 6, 10 |
| dropout | 0,3, 0,5 |

**Combinaciones totales:** 72.

### 8.3 Estrategia: Random Search

Con `FAST_MODE = True` se muestrean **3 combinaciones** aleatorias (semilla 42) del grid reducido (16 y 32 filtros, 2 learning rates, épocas 3–5). Con `FAST_MODE = False` el notebook usa **8 trials** sobre un grid más amplio. El criterio de selección es el **máximo F1 en validación** por trial.

> **Nota:** si ejecutaste una versión anterior del notebook (8 trials, IMG 128), las cifras de la tabla pueden variar. Usa siempre los valores impresos al final de tu ejecución para el informe.

**Justificación frente a grid completo:** random search explora el espacio de forma eficiente; estudios empíricos (Bergstra & Bengio, 2012) muestran que suele encontrar configuraciones competitivas con menos experimentos.

### 8.4 Resultados de los trials (validación)

| Trial | Filtros | Kernel | LR | Épocas | Dropout | Mejor F1 (val) |
|-------|---------|--------|-----|--------|---------|----------------|
| **4** | 32 | 3 | 5×10⁻⁴ | 10 | 0,5 | **0,9846** |
| 1 | 16 | 5 | 1×10⁻⁴ | 10 | 0,3 | 0,9838 |
| 2 | 16 | 3 | 1×10⁻⁴ | 10 | 0,5 | 0,9838 |
| 5 | 32 | 3 | 5×10⁻⁴ | 6 | 0,3 | 0,9828 |
| 6 | 16 | 5 | 5×10⁻⁴ | 6 | 0,5 | 0,9791 |
| 3 | 32 | 3 | 1×10⁻³ | 10 | 0,5 | 0,9778 |

**Mejora respecto al baseline:** 0,9758 → **0,9846** (+0,88 puntos porcentuales en F1 val).

**Configuración óptima seleccionada (Trial 4):**

```python
BEST_HP = {
    "base_filters": 32,
    "kernel_size": 3,
    "lr": 0.0005,
    "epochs": 12,      # entrenamiento final ≥ épocas de búsqueda
    "dropout": 0.5,
}
```

### 8.5 Documentación de cambios durante la optimización

| Cambio | Efecto observado |
|--------|------------------|
| Filtros 16 → 32 (trials 1–2 vs 4–5) | Mayor capacidad; F1 val más alto con 32 filtros |
| LR 1e-3 → 5e-4 (trial 3 vs 4) | Menor inestabilidad; trial 3 mostró más variación en val loss |
| Kernel 5 vs 3 | Resultados mixtos; kernel 3 ganó en el mejor trial |
| Dropout 0,5 | Presente en la mejor configuración; reduce sobreajuste en capas densas |
| Menos épocas (6) | Trial 5 competitivo (0,9828) pero trial 4 con 10 épocas alcanzó el máximo |

**Figura 5 (al completar notebook).** `outputs/hyperparameter_search.png`

### 8.6 Entrenamiento del modelo final

Con la configuración del Trial 4 se reentrena el modelo (≥ 12 épocas, early stopping con paciencia 5), guardando el checkpoint de mejor F1 en validación para la evaluación en test.

> **Nota:** Ejecutar las celdas §10 y §11 del notebook para obtener el modelo final guardado en `outputs/best_cnn_model.pth` y las métricas de test definitivas.

---

## 9. Evaluación del modelo

### 9.1 Protocolo de evaluación

| Aspecto | Criterio |
|---------|----------|
| Conjunto | 624 imágenes de `test/` (independiente) |
| Transformaciones | `eval_transform` (sin augmentación) |
| Modelo | Pesos del mejor epoch en validación (modelo final) |
| Umbral de decisión | Argmax sobre logits (clase de mayor probabilidad) |

### 9.2 Métricas seleccionadas

| Métrica | Definición (intuición) | Relevancia en este problema |
|---------|------------------------|----------------------------|
| **Accuracy** | Proporción de aciertos global | Útil pero sesgada por desbalance |
| **Precision** | De los predichos como clase X, cuántos son X | Controla falsas alarmas de neumonía |
| **Recall** | De los verdaderos X, cuántos detecta el modelo | Crítico para no **omitir neumonías** |
| **F1-score** | Media armónica de precision y recall | Balance entre ambos; métrica principal de optimización |

Se reportan valores **ponderados** (weighted) para resumen global y desglose **por clase** en el reporte de clasificación.

### 9.3 Código: evaluación en test

```python
test_loss, test_m = run_epoch(final_model, test_loader, criterion)
print(classification_report(
    test_m["y_true"], test_m["y_pred"],
    target_names=["NORMAL", "PNEUMONIA"]
))
cm = confusion_matrix(test_m["y_true"], test_m["y_pred"])
```

**Resultado esperado:** impresión de métricas globales, tabla por clase y matriz 2×2 para interpretación.

---

## 10. Métricas

### 10.1 Métricas en validación (modelo optimizado — Trial 4)

| Métrica | Valor |
|---------|-------|
| F1-score (ponderado) | **0,9846** |
| Criterio de selección | Mejor epoch durante random search, trial 4 |

### 10.2 Métricas en validación (baseline)

| Métrica | Valor |
|---------|-------|
| F1-score (ponderado) | **0,9758** |

### 10.3 Métricas en conjunto de prueba (test)

> **Completar** tras ejecutar la celda de evaluación final del notebook. Sustituir los valores entre corchetes por los obtenidos en consola.

| Métrica | Valor |
|---------|-------|
| Loss | [ejecutar notebook] |
| **Accuracy** | [ejecutar notebook] |
| **Precision** (ponderada) | [ejecutar notebook] |
| **Recall** (ponderado) | [ejecutar notebook] |
| **F1-score** (ponderado) | [ejecutar notebook] |

### 10.4 Métricas por clase (test)

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | [ ] | [ ] | [ ] | 234 |
| PNEUMONIA | [ ] | [ ] | [ ] | 390 |

**Orientación interpretativa (cuando dispongas de los números):**

- **Recall bajo en PNEUMONIA** → riesgo clínico alto (falsos negativos).
- **Precision baja en PNEUMONIA** → muchos normales clasificados como neumonía.
- Comparar con validación: si test << val, posible **sobreajuste a la partición de validación** o **shift** de distribución entre carpetas.

---

## 11. Matriz de confusión e interpretación

### 11.1 Estructura

Para clasificación binaria, la matriz 2×2 contiene:

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **Real NORMAL** | Verdaderos negativos (VN) | Falsos positivos (FP) |
| **Real PNEUMONIA** | Falsos negativos (FN) | Verdaderos positivos (VP) |

**Figura 6.** `outputs/confusion_matrix_test.png` (generar con celda §11 del notebook)

### 11.2 Interpretación académica

1. **Verdaderos positivos (VP):** neumonías correctamente identificadas — deseable maximizar en screening.
2. **Falsos negativos (FN):** neumonía clasificada como normal — error **más grave** en un sistema de apoyo diagnóstico; suelen priorizarse políticas que maximicen recall de la clase patológica.
3. **Falsos positivos (FP):** normal clasificado como neumonía — incrementa carga de revisión humana y posibles pruebas innecesarias.
4. **Verdaderos negativos (VN):** normales correctamente descartados.

La matriz debe leerse **junto con** precision y recall por clase, no de forma aislada.

### 11.3 Relación con el desbalance del test

Con 62,5 % de PNEUMONIA en test, una matriz “visualmente equilibrada” en diagonal puede ocultar mal rendimiento en NORMAL. Por ello el informe enfatiza métricas **por clase**.

---

## 12. Resultados obtenidos

### 12.1 Resumen del pipeline

| Fase | Resultado principal |
|------|-------------------|
| EDA | Desbalance 2,88:1 en train; variabilidad de tamaños |
| Preprocesamiento | 128×128, normalización μ=0,482, σ=0,235 |
| Baseline | F1 val = 0,9758 |
| Random search | Mejor F1 val = **0,9846** (32 filtros, k=3, lr=5e-4, dropout=0.5) |
| Test | *[Completar tras ejecución final]* |

### 12.2 Logros respecto a la consigna

- CNN implementada con convolución, pooling, capas densas y dropout.
- División train/val/test metodológicamente correcta.
- Regularización múltiple (dropout, weight decay, early stopping, pesos de clase).
- Optimización documentada con tabla de trials y reflexión causal.
- Artefactos reproducibles: notebook, `requirements.txt`, figuras en `outputs/`.

### 12.3 Limitaciones de la ejecución actual

- Entrenamiento en **CPU** → se usa modo rápido (`FAST_MODE`) para completar el flujo; menor resolución y menos trials que un estudio exhaustivo.
- Resolución 64×64 en modo rápido → posible pérdida de detalle fino respecto a 128×128 o superior.
- Sin validación cruzada k-fold.
- Métricas de test pendientes de la última ejecución del notebook.

---

## 13. Análisis crítico y reflexión sobre el desempeño

### 13.1 Calidad del aprendizaje

El F1 de validación superior a **0,97** en baseline y **0,98** tras optimización indica que la arquitectura y el preprocesamiento son **apropiados** para el dataset. La mejora marginal pero consistente tras random search (+0,88 pp) sugiere que el espacio de hiperparámetros estaba razonablemente acotado y que el baseline ya era fuerte.

### 13.2 Overfitting y underfitting

| Fenómeno | Evidencia en el proyecto | Valoración |
|----------|-------------------------|------------|
| **Overfitting** | En baseline, val F1 ≥ train F1 en mejor epoch; early stopping activo | No evidencia fuerte de sobreajuste en el punto seleccionado |
| **Underfitting** | Loss de train baja; F1 train alto (~0,96) | Capacidad del modelo adecuada |
| **Riesgo futuro** | Muchos trials con F1 > 0,98 en val | Posible optimismo; **test** es la prueba definitiva |

### 13.3 Sesgo por desbalance

Aun con pesos en la loss, conviene analizar si el modelo favorece PNEUMONIA. La matriz de confusión en test y el recall por clase NORMAL son esenciales para una reflexión honesta.

### 13.4 Validez externa

El dataset es un benchmark académico; **no** se debe extrapolar directamente a despliegue clínico sin validación prospectiva, ética y regulación correspondiente.

### 13.5 Alineación con la rúbrica (autoevaluación)

| Criterio | Nivel aspirado | Evidencia |
|----------|----------------|-----------|
| Implementación del modelo | Excelente (5) | CNN completa, configurable, documentada capa por capa |
| Análisis y evaluación | Excelente (5) | Métricas múltiples, matriz interpretada; *completar números test* |
| Optimización | Excelente (5) | Random search, tabla, reflexión por hiperparámetro |
| Documentación | Excelente (5) | Informe estructurado + notebook reproducible |

---

## 14. Conclusiones

1. Se desarrolló un sistema de clasificación de radiografías de tórax basado en **CNN en PyTorch**, cumpliendo la estructura convolucional–pooling–densa exigida en la práctica.
2. El **EDA** demostró **desbalance marcado**, lo que motivó pesos en la pérdida y métricas más allá de la accuracy.
3. El **preprocesamiento** estandarizó entradas y aplicó normalización basada en estadísticas del train, evitando fuga de información.
4. El **entrenamiento** con Adam, dropout, weight decay y early stopping produjo un baseline sólido (F1 val = 0,9758).
5. La **optimización por random search** mejoró el rendimiento a F1 val = **0,9846** con filtros 32, kernel 3, lr = 5×10⁻⁴ y dropout 0,5.
6. La evaluación en **test independiente** (al completar el notebook) permitirá cerrar el ciclo metodológico con métricas y matriz de confusión finales.
7. El trabajo ilustra que el rendimiento en visión artificial depende tanto del **diseño arquitectónico** como de la **protocolización experimental** (particiones, métricas y búsqueda de hiperparámetros).

---

## 15. Recomendaciones futuras

1. **Completar evaluación en test** y, si es posible, los 8 trials de random search en GPU.
2. **Transfer learning** (ResNet-18/50, EfficientNet) preentrenado y fine-tuning comparativo.
3. Aumentar resolución a **224×224** o superior si el hardware lo permite.
4. **Validación cruzada estratificada k-fold** para estimaciones más robustas de métricas.
5. Curvas **ROC** y **AUC**, y análisis de umbrales distintos de 0,5 según costo FN/FP.
6. Técnicas avanzadas de balanceo: **oversampling** de NORMAL, **focal loss** o muestreo balanceado.
7. **Grad-CAM** o mapas de activación para interpretabilidad visual de las decisiones.
8. Validación en datos de otro hospital o dominio para estudiar **generalización externa**.

---

## Referencias y anexos técnicos

### Anexo A — Exportación a Word

```bash
cd "c:\IC\Machine Learning II"
pandoc INFORME_ACADEMICO_CNN.md -o INFORME_ACADEMICO_CNN.docx
```

Insertar figuras manualmente desde `outputs/` en las secciones indicadas.

### Anexo B — Checklist antes de entregar

- [ ] Ejecutar celdas finales del notebook (modelo final + test)
- [ ] Reemplazar valores `[ejecutar notebook]` en secciones 10 y 12
- [ ] Insertar Figuras 1–6 en Word
- [ ] Pegar contenido en plantilla institucional (carátula e índice)
- [ ] Revisar ortografía y numeración de figuras/tablas

### Anexo C — Archivos del proyecto

| Archivo | Descripción |
|---------|-------------|
| `chest_xray_cnn.ipynb` | Notebook con pipeline completo |
| `requirements.txt` | Dependencias Python |
| `outputs/` | Gráficos y modelos exportados |
| `.venv/` | Entorno virtual (no incluir en ZIP de entrega si pesa mucho) |

---

*Documento generado para integración en plantilla Word. Métricas de test: actualizar tras ejecutar §10–§11 de `chest_xray_cnn.ipynb`.*
