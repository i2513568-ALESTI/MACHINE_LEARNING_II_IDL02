# Informe académico: clasificación de radiografías de tórax mediante redes neuronales convolucionales

**Asignatura:** Machine Learning II  
**Tema:** Clasificación de imágenes con CNN (PyTorch)  
**Dataset:** Chest X-Ray — clases NORMAL y PNEUMONIA  
**Implementación:** `chest_xray_cnn.ipynb` | **Figuras:** `outputs/`

---

## 1. Introducción

La inteligencia artificial aplicada a imágenes médicas ha demostrado utilidad en tareas de apoyo al diagnóstico, entre ellas la detección de patrones compatibles con **neumonía** en radiografías de tórax. Este tipo de imágenes presenta desafíos propios: variabilidad de adquisición, ruido, superposición de estructuras óseas y tejidos, y —en conjuntos públicos— frecuente **desbalance entre clases**.

En este proyecto se aborda un problema de **clasificación supervisada binaria**: dada una radiografía, el sistema debe asignar la etiqueta **NORMAL** o **PNEUMONIA**. Para ello se diseña, implementa y evalúa una **red neuronal convolucional (CNN)** en **PyTorch**, siguiendo el flujo definido en el notebook: análisis exploratorio, preprocesamiento, modelado, validación, búsqueda de hiperparámetros y evaluación en un conjunto de prueba **independiente**.

El informe documenta las decisiones técnicas del notebook y **interpreta** el desempeño obtenido, incluyendo la brecha entre validación y test.

---

## 2. Objetivo del proyecto

### 2.1 Objetivo general

Diseñar, implementar, entrenar y evaluar un modelo CNN capaz de clasificar radiografías de tórax en las categorías NORMAL y PNEUMONIA, documentando el proceso completo y optimizando hiperparámetros mediante **random search**.

### 2.2 Objetivos específicos

1. Realizar **análisis exploratorio (EDA)** y cuantificar el **desbalance de clases**.
2. Aplicar **preprocesamiento** (redimensionamiento a 128×128, escala de grises, normalización y aumentación controlada).
3. Construir una CNN con capas **convolucionales**, **pooling** y **densas**, incluyendo **dropout**.
4. Dividir datos en **entrenamiento** y **validación** (desde `train/`), reservando `test/` para evaluación final.
5. Monitorear **pérdida** y **métricas** durante el entrenamiento; aplicar **early stopping**.
6. Optimizar hiperparámetros mediante **random search** y reflexionar sobre su impacto.
7. Reportar **accuracy**, **precision**, **recall**, **F1-score** y **matriz de confusión** en test, con interpretación clínica básica.

---

## 3. Descripción del dataset

> **Referencia:** `chest_xray_cnn.ipynb` — §2 Configuración y §3 Análisis exploratorio (EDA).

### 3.1 Origen y estructura

El dataset está organizado en carpetas por clase (formato `ImageFolder` de torchvision):

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

### 3.2 Configuración y conteo por clase

En el notebook se definen las rutas, `IMG_SIZE = 128`, `BATCH_SIZE = 32`, `VAL_RATIO = 0.2` y una función `list_images()` que **excluye archivos no imagen** (por ejemplo `.DS_Store`).

**Salida de consola (celda EDA — conteo):**

```
TRAIN: {'NORMAL': 1349, 'PNEUMONIA': 3883} | Total: 5232
TEST:  {'NORMAL': 234, 'PNEUMONIA': 390} | Total: 624
```

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

### 3.4 Visualización del desbalance

**Salida de consola (gráfico de barras):**

```
Train: PNEUMONIA=74.2% | Ratio PNEUMONIA/NORMAL=2.88x
Test: PNEUMONIA=62.5% | Ratio PNEUMONIA/NORMAL=1.67x
```

**Figura 1.** Distribución de clases (train y test) — `outputs/eda_class_distribution.png`

![Distribución de clases](outputs/eda_class_distribution.png)

### 3.5 Implicaciones para el modelado

- El desbalance explica por qué la **accuracy** sola puede ser engañosa: un clasificador que prediga siempre PNEUMONIA alcanzaría ~62,5 % de aciertos en test sin ser clínicamente útil.
- Se emplean **pesos de clase** en la pérdida y métricas **por clase** además de las ponderadas.
- La carpeta **test/** no interviene en entrenamiento ni en la búsqueda de hiperparámetros.

### 3.6 Exploración visual

El notebook muestra ejemplos por clase y un scatter de tamaños originales (submuestra), confirmando **variabilidad espacial** entre radiografías antes del resize a 128×128.

**Figura 2.** Ejemplos por clase — `outputs/eda_sample_images.png`

![Ejemplos por clase](outputs/eda_sample_images.png)

**Figura 3.** Tamaños originales — `outputs/eda_image_sizes.png`

![Tamaños de imagen](outputs/eda_image_sizes.png)

---

## 4. Preprocesamiento de datos

> **Referencia:** `chest_xray_cnn.ipynb` — §4 Preprocesamiento.

### 4.1 Justificación general

Las CNN requieren tensores de tamaño uniforme y escalas comparables. El preprocesamiento reduce variabilidad irrelevante y estabiliza el entrenamiento.

### 4.2 Pipeline implementado

| Etapa | Descripción | Justificación |
|-------|-------------|---------------|
| Escala de grises (1 canal) | `Grayscale(num_output_channels=1)` | Las radiografías son inherentemente grises |
| Resize 128×128 | `Resize((IMG_SIZE, IMG_SIZE))` | Tamaño fijo definido en configuración (`IMG_SIZE = 128`) |
| Normalización | `(x - mean) / std` | Estabiliza activaciones; estadísticas calculadas **solo en train** |
| RandomHorizontalFlip (train) | Volteo horizontal con p = 0,5 | Aumentación ligera; simetría aproximada del tórax |

**Estadísticas de normalización** (conjunto train completo, celda §4):

- Media μ ≈ **0,482**
- Desviación estándar σ ≈ **0,235**

Salida del notebook: `Mean/Std: [0.4823513627052307] [0.23516277968883514]`

### 4.3 Código representativo

```python
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
MEAN, STD = compute_mean_std(DataLoader(tmp_ds, ...))

def get_transforms(mean, std, augment=False):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)
```

Validación y test usan `eval_transform` (sin aumentación).

### 4.4 Entorno de ejecución

El notebook se ejecutó en **CPU** con **PyTorch 2.12.0+cpu** (`device = cpu`). Por ello el entrenamiento baseline, los trials de random search y el modelo final usan **pocas épocas** (1–2 por experimento), tal como indica el propio notebook en la configuración de §2 (`N_RANDOM_TRIALS = 3`, `HP_TRIAL_EPOCHS = [1, 2]`).

---

## 5. Diseño e implementación del modelo CNN

> **Referencia:** `chest_xray_cnn.ipynb` — §5 División train/val, §6 Arquitectura CNN.

### 5.1 Enfoque de diseño

Se implementa una **CNN personalizada** (`ChestXRayCNN`) con bloques convolucionales, batch normalization, ReLU, max pooling y cabeza densa con dropout, sin transfer learning.

### 5.2 Partición train / validación / test

Desde `train/` (5 232 imágenes) se extrae un **20 % estratificado** (`VAL_RATIO = 0.2`, `SEED = 42`):

| Subconjunto | Tamaño |
|-------------|--------|
| Train interno | 4 187 |
| Validación | 1 045 |
| Test (`test/`) | 624 |

Salida: `Train: 4187 | Val: 1045 | Test: 624`

### 5.3 Esquema de la arquitectura

Con `base_filters = f` (por defecto 32) y `IMG_SIZE = 128`:

```
Entrada [1 × 128 × 128]
    → Bloque 1: Conv(f) → BN → ReLU → MaxPool(2)  → [f × 64 × 64]
    → Bloque 2: Conv(2f) → BN → ReLU → MaxPool(2) → [2f × 32 × 32]
    → Bloque 3: Conv(4f) → BN → ReLU → MaxPool(2) → [4f × 16 × 16]
    → Flatten → Linear(4f·16·16, 256) → ReLU → Dropout → Linear(256, 2)
```

Con f = 32: filtros **32, 64, 128**; vector aplanado de dimensión **128 × 16 × 16 = 32 768**.

---

## 6. Explicación detallada de cada capa del modelo

### 6.1 Capa convolucional (`Conv2d`)

Detecta patrones locales (bordes, texturas, opacidades). `kernel_size` 3 o 5 en los experimentos; `padding = kernel_size // 2` preserva dimensiones antes del pooling.

### 6.2 Batch Normalization (`BatchNorm2d`)

Normaliza activaciones por canal y mini-batch; acelera convergencia y actúa como regularizador leve.

### 6.3 ReLU

Introduce no linealidad sin saturación en la región positiva.

### 6.4 Max Pooling (`MaxPool2d(2)`)

Reduce resolución espacial (128 → 64 → 32 → **16**), aporta invariancia local y amplía el campo receptivo.

### 6.5 Capas densas y dropout

La cabeza combina características globales (256 unidades) y produce **2 logits**. `Dropout` (0,3–0,5) mitiga sobreajuste en la parte totalmente conectada.

### 6.6 Función de pérdida

`CrossEntropyLoss` con pesos de clase calculados sobre train:

| Clase | Peso |
|-------|------|
| NORMAL | 1,94 |
| PNEUMONIA | 0,67 |

Salida del notebook: `Pesos de clase: tensor([1.9392, 0.6737])`

---

## 7. Entrenamiento del modelo

> **Referencia:** `chest_xray_cnn.ipynb` — §7 Entrenamiento, §8 Baseline.

### 7.1 Optimizador y procedimiento

- **Adam** con `weight_decay = 1e-4`.
- **Batch size = 32**.
- Por época: entrenamiento con gradiente; validación sin gradiente; registro de loss y F1 ponderado.
- **Early stopping:** si el F1 de validación no mejora durante `patience` épocas, se restauran los pesos del mejor epoch.

### 7.2 Entrenamiento baseline

Configuración (§8 del notebook):

| Hiperparámetro | Valor |
|----------------|-------|
| base_filters | 32 |
| kernel_size | 3 |
| dropout | 0,5 |
| learning rate | 0,001 |
| épocas | **1** |

**Salida:**

```
Época 1/1 | train loss=0.8486 f1=0.8970 | val loss=0.1767 f1=0.9261
Mejor F1 validación (baseline): 0.9261
```

**Figura 4.** Curvas baseline — `outputs/baseline_curves.png`

### 7.3 Lectura preliminar del baseline

Con una sola época el modelo ya alcanza F1 val ≈ 0,93, pero el entrenamiento es breve; conviene contrastar con test para valorar generalización real.

---

## 8. Optimización de hiperparámetros

> **Referencia:** `chest_xray_cnn.ipynb` — §9 Random search.

### 8.1 Espacio de búsqueda

Definido en el notebook (`HP_GRID`):

| Hiperparámetro | Valores |
|----------------|---------|
| base_filters | 16, 32 |
| kernel_size | 3, 5 |
| learning rate | 5×10⁻⁴, 1×10⁻³ |
| epochs (por trial) | 1, 2 |
| dropout | 0,3, 0,5 |

**Combinaciones totales:** 32. Se muestrean **3 trials** aleatorios (`N_RANDOM_TRIALS = 3`, semilla 42). Criterio: **máximo F1 en validación**.

Salida inicial: `Grid total: 32 | Trials: 3 | épocas por trial: [1, 2]`

### 8.2 Resultados de los trials

| Trial | Filtros | Kernel | LR | Épocas | Dropout | Mejor F1 (val) |
|-------|---------|--------|-----|--------|---------|----------------|
| **3** | 32 | 3 | 5×10⁻⁴ | 1 | 0,5 | **0,9675** |
| 1 | 16 | 3 | 1×10⁻³ | 2 | 0,5 | 0,9642 |
| 2 | 16 | 3 | 5×10⁻⁴ | 1 | 0,5 | 0,9352 |

**Mejora respecto al baseline:** 0,9261 → **0,9675** (+4,1 puntos porcentuales en F1 val).

**Configuración óptima (Trial 3):**

- base_filters = 32  
- kernel_size = 3  
- lr = 5×10⁻⁴  
- dropout = 0,5  
- epochs = 1 (en el trial ganador)

### 8.3 Reflexión sobre hiperparámetros

| Observación | Interpretación |
|-------------|----------------|
| 32 filtros supera a 16 en el mejor trial | Mayor capacidad ayuda con resolución 128×128 |
| Trial 1 con 2 épocas bajó F1 en la 2.ª época (0,9642 → 0,9387) | Riesgo de sobreajuste o inestabilidad con pocas épocas extra |
| lr = 5×10⁻⁴ mejor que 1×10⁻³ en trials comparables | Tasa moderada más estable en CPU con pocos pasos |

**Figura 5.** Random search — `outputs/hyperparameter_search.png`

### 8.4 Entrenamiento del modelo final

> **Referencia:** `chest_xray_cnn.ipynb` — §10 Entrenamiento final.

Se reentrena con los hiperparámetros del mejor trial (`BEST_HP` derivado de `best_row`). En la ejecución registrada:

```
Época 1/1 | train loss=0.4822 f1=0.9064 | val loss=0.1083 f1=0.9599
F1 validación final: 0.9599
```

**Figura 6.** Curvas modelo final — `outputs/final_curves.png`

El checkpoint se guarda en `outputs/best_cnn_model.pth`.

---

## 9. Evaluación del modelo

> **Referencia:** `chest_xray_cnn.ipynb` — §11 Evaluación en test.

### 9.1 Protocolo

| Aspecto | Criterio |
|---------|----------|
| Conjunto | 624 imágenes de `test/` |
| Transformaciones | `eval_transform` (sin augmentación) |
| Modelo | Pesos del mejor epoch en validación (entrenamiento final) |
| Decisión | Argmax sobre logits |

### 9.2 Métricas globales en test

**Salida del notebook:**

| Métrica | Valor |
|---------|-------|
| Loss | 1,1297 |
| **Accuracy** | **0,6875** (68,75 %) |
| **Precision** (ponderada) | **0,7917** |
| **Recall** (ponderado) | **0,6875** |
| **F1-score** (ponderado) | **0,6071** |

### 9.3 Métricas por clase (test)

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 1,00 | 0,17 | 0,29 | 234 |
| PNEUMONIA | 0,67 | 1,00 | 0,80 | 390 |

**Promedios (sklearn):** macro avg F1 ≈ 0,54; weighted avg F1 ≈ 0,61; accuracy reportada 0,69.

### 9.4 Interpretación clínica y técnica

El modelo en test muestra un patrón claro:

- **Recall de PNEUMONIA = 1,00:** clasifica casi todas las neumonías como positivas (pocos falsos negativos).
- **Recall de NORMAL = 0,17:** la mayoría de casos normales se clasifican como neumonía (muchos falsos positivos respecto a NORMAL).

Esto es coherente con el **desbalance** y con **pocas épocas de entrenamiento**: el clasificador aprende a favorecer la clase mayoritaria (PNEUMONIA), obteniendo buen F1 en validación (~0,96) pero **generalización limitada en test** (F1 ponderado ≈ 0,61).

La brecha **validación (0,96) vs test (0,61)** indica que las métricas de validación, con tan poco entrenamiento, no reflejan el comportamiento en la partición `test/` (posible **shift** de distribución: 74 % vs 62,5 % de PNEUMONIA).

**Figura 7.** Matriz de confusión — `outputs/confusion_matrix_test.png`

![Matriz de confusión test](outputs/confusion_matrix_test.png)

Estimación a partir del reporte: ~40 verdaderos negativos (NORMAL bien clasificados), ~194 falsos positivos (NORMAL predichos como PNEUMONIA), ~390 verdaderos positivos de PNEUMONIA.

---

## 10. Métricas (resumen)

| Fase | F1 (val) | Notas |
|------|----------|-------|
| Baseline (1 época) | 0,9261 | 32 filtros, lr 1e-3 |
| Mejor trial random search | **0,9675** | Trial 3 |
| Modelo final (1 época) | 0,9599 | Hiperparámetros del trial 3 |
| **Test (independiente)** | **0,6071** (ponderado) | Accuracy 68,75 % |

---

## 11. Matriz de confusión e interpretación

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **Real NORMAL** | VN (~40) | FP (~194) |
| **Real PNEUMONIA** | FN (~0) | VP (~390) |

**Lectura:** el error más frecuente es clasificar radiografías **NORMAL** como **PNEUMONIA**. En un contexto clínico, eso implica muchas alarmas falsas; el bajo recall de NORMAL (17 %) es el principal problema del modelo en esta ejecución.

---

## 12. Resultados obtenidos

### 12.1 Resumen del pipeline

| Fase | Resultado principal |
|------|---------------------|
| EDA | Desbalance 2,88:1 en train; 2,88× más PNEUMONIA que NORMAL |
| Preprocesamiento | 128×128, μ≈0,482, σ≈0,235, flip horizontal en train |
| Baseline | F1 val = 0,9261 (1 época) |
| Random search (3 trials) | Mejor F1 val = **0,9675** |
| Modelo final | F1 val = 0,9599 |
| Test | F1 ponderado = **0,6071**; fuerte sesgo hacia PNEUMONIA |

### 12.2 Logros respecto a la consigna

- CNN con convolución, pooling, capas densas y dropout implementada en PyTorch.
- División train/val/test metodológicamente correcta.
- Regularización: dropout, weight decay, early stopping, pesos de clase.
- Random search documentado con tabla de trials.
- Evaluación completa en test con métricas y matriz de confusión.

### 12.3 Limitaciones de esta ejecución

- Entrenamiento muy corto (1–2 épocas) por restricción de tiempo en CPU.
- Fuerte **desajuste validación–test**; el modelo no equilibra bien la clase NORMAL en test.
- Sin validación cruzada k-fold ni transfer learning.
- El dataset es académico; no es válido para despliegue clínico sin validación adicional.

---

## 13. Análisis crítico y reflexión

### 13.1 Validación vs test

El F1 en validación (~0,97) sugiere un modelo aparentemente excelente, pero en test cae a **0,61**. Esta discrepancia obliga a:

1. No confiar solo en validación con entrenamientos muy breves.
2. Analizar siempre **recall por clase**, especialmente de NORMAL.
3. Considerar más épocas, mejor balanceo o umbrales de decisión distintos de 0,5.

### 13.2 Overfitting y sesgo de clase

El trial 1 empeoró en la segunda época (F1 val 0,9642 → 0,9387), señal de posible sobreajuste al continuar entrenando. En test, el recall perfecto de PNEUMONIA junto al recall bajo de NORMAL indica **sesgo hacia la clase mayoritaria**, pese a los pesos en la loss.

### 13.3 Validez externa

El conjunto Chest X-Ray es un benchmark educativo. Cualquier uso clínico requeriría validación prospectiva, revisión ética y cumplimiento normativo.

---

## 14. Conclusiones

1. Se implementó un pipeline completo en `chest_xray_cnn.ipynb` para clasificar radiografías NORMAL vs PNEUMONIA con una CNN personalizada en PyTorch.
2. El EDA confirmó **desbalance marcado** (74 % PNEUMONIA en train), justificando pesos en la pérdida y métricas por clase.
3. El preprocesamiento estandarizó entradas a **128×128** con normalización basada solo en train.
4. El baseline (1 época) alcanzó F1 val = **0,9261**; el **random search** (3 trials) mejoró a F1 val = **0,9675** con 32 filtros, kernel 3, lr = 5×10⁻⁴ y dropout 0,5.
5. En **test independiente**, el rendimiento fue moderado (F1 ponderado **0,61**, accuracy **69 %**), con **recall de NORMAL muy bajo (17 %)** y recall de PNEUMONIA del 100 %.
6. El trabajo demuestra que métricas altas en validación no garantizan buen desempeño en test cuando el entrenamiento es limitado y las clases están desbalanceadas.

---

## 15. Recomendaciones futuras

1. Aumentar **épocas de entrenamiento** (con GPU si es posible) y monitorizar curvas train/val.
2. Probar **umbrales de decisión** distintos para priorizar recall de NORMAL o de PNEUMONIA según coste clínico.
3. **Transfer learning** (ResNet, EfficientNet) con fine-tuning.
4. **Oversampling** de NORMAL, focal loss o muestreo balanceado por batch.
5. **Validación cruzada estratificada** para estimaciones más robustas.
6. **Grad-CAM** para interpretabilidad visual.
7. Curvas **ROC/AUC** y análisis de generalización en datos de otro origen.

---

## Referencias y anexos

### Anexo A — Exportación a Word

```bash
cd "c:\IC\Machine_Learning_II_2026\MACHINE_LEARNING_II_IDL02"
pandoc INFORME_ACADEMICO_CNN.md -o INFORME_ACADEMICO_CNN.docx
```

Insertar figuras desde `outputs/` en las secciones indicadas.

### Anexo B — Archivos del proyecto

| Archivo | Descripción |
|---------|-------------|
| `chest_xray_cnn.ipynb` | Notebook con pipeline completo (EDA → CNN → random search → test) |
| `requirements.txt` | Dependencias Python |
| `outputs/` | Gráficos (`eda_*.png`, `baseline_curves.png`, `hyperparameter_search.png`, `final_curves.png`, `confusion_matrix_test.png`) y `best_cnn_model.pth` |

### Anexo C — Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| §1–2 | Imports y configuración (`IMG_SIZE=128`, rutas, semilla) |
| §3 | EDA y figuras de distribución |
| §4 | Preprocesamiento y normalización |
| §5–6 | División estratificada y arquitectura `ChestXRayCNN` |
| §7–8 | Funciones de entrenamiento y baseline |
| §9 | Random search (3 trials) |
| §10–11 | Modelo final y evaluación en test |
| §12 | Conclusiones del notebook |

---

*Informe alineado con la ejecución documentada en `chest_xray_cnn.ipynb` (PyTorch 2.12.0+cpu). Figuras generadas al ejecutar todas las celdas del notebook.*
