# 🃏 GAN FIFA Panini — FIFA World Cup 2026
### Centro de Competencias Digitales · UNAB · Bucaramanga, Colombia

> Proyecto educativo de **Deep Learning generativo**: entrena una DCGAN con rostros de futbolistas FIFA y genera tarjetas coleccionables estilo Panini usando una app web interactiva desplegada en Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Colab](https://img.shields.io/badge/Entrenamiento-Google%20Colab%20GPU-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/Licencia-MIT-green)

---

## 📋 Tabla de contenidos

1. [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
2. [Estructura del repositorio](#-estructura-del-repositorio)
3. [Requisitos previos](#-requisitos-previos)
4. [Parte 1 — Entrenamiento en Google Colab](#-parte-1--entrenamiento-en-google-colab)
5. [Parte 2 — Guardar el modelo en GitHub](#-parte-2--guardar-el-modelo-en-github)
6. [Parte 3 — Despliegue en Streamlit Cloud](#-parte-3--despliegue-en-streamlit-cloud)
7. [Uso de la aplicación](#-uso-de-la-aplicación)
8. [Conceptos clave](#-conceptos-clave)
9. [Créditos](#-créditos)

---

## 🎯 ¿Qué hace este proyecto?

El proyecto tiene **dos componentes** independientes que trabajan juntos:

| Componente | Herramienta | Descripción |
|---|---|---|
| **Entrenamiento** | Google Colab (GPU) | Notebook que entrena la DCGAN y exporta el generador |
| **Aplicación web** | Streamlit Cloud | App que genera tarjetas Panini usando el modelo entrenado |

### Flujo general

```
[Google Colab]                    [GitHub]                  [Streamlit Cloud]
Entrenar DCGAN  →  Subir modelo  →  Repositorio  →  App web pública
     GPU             .keras          + código           Tarjeta Panini
```

### ¿Cómo genera las caras?

La app **no necesita que subas una foto**. Usa una semilla numérica para generar un vector de ruido aleatorio que el Generador convierte en un rostro sintético de futbolista:

```
Semilla (ej: 1234)
      ↓
z = vector aleatorio ∈ ℝ¹⁰⁰   ← espacio latente
      ↓
GENERADOR DCGAN (entrenado con futbolistas FIFA)
      ↓
Imagen 64×64 RGB de "futbolista" sintético
      ↓
🃏 Tarjeta Panini personalizada
```

> **La misma semilla siempre produce la misma cara.** Cambia la semilla → cara diferente.

---

## 📁 Estructura del repositorio

```
tu-repositorio/
│
├── 📓 CCD_UNAB_GAN_Entrenamiento.ipynb   ← Notebook Colab (entrenamiento)
├── 🐍 app_panini_streamlit.py             ← App Streamlit (tarjeta Panini)
├── 📄 requirements.txt                    ← Dependencias Python para Streamlit
├── 📄 README.md                           ← Este archivo
│
└── 📦 generador_fifa_ccd.keras            ← Modelo entrenado (subirlo después)
                                              ⚠️ Archivo pesado (~50 MB)
                                              Usar Git LFS o Google Drive
```

> **Nota sobre el modelo:** El archivo `.keras` puede pesar entre 30–80 MB. Si supera 100 MB, usa **Git LFS** o alójalo en Google Drive (ver [Parte 2](#-parte-2--guardar-el-modelo-en-github)).

---

## ✅ Requisitos previos

Antes de comenzar necesitas tener:

- [ ] Cuenta de **Google** (para Google Colab y Google Drive)
- [ ] Cuenta de **GitHub** — [Crear cuenta gratis](https://github.com/signup)
- [ ] Cuenta de **Streamlit Community Cloud** — [Crear cuenta gratis](https://share.streamlit.io) (se vincula con GitHub)
- [ ] **Git** instalado en tu PC — [Descargar Git](https://git-scm.com/downloads)

> No necesitas instalar Python ni TensorFlow localmente. Todo el entrenamiento ocurre en la nube.

---

## 🧠 Parte 1 — Entrenamiento en Google Colab

### Paso 1.1 — Abrir el notebook en Colab

**Opción A — Desde GitHub (recomendado):**

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Haz clic en **`Archivo` → `Abrir cuaderno`**
3. Selecciona la pestaña **`GitHub`**
4. Pega la URL de tu repositorio:
   ```
   https://github.com/TU-USUARIO/TU-REPOSITORIO
   ```
5. Selecciona `CCD_UNAB_GAN_Entrenamiento.ipynb`
6. Haz clic en **Abrir**

**Opción B — Subir el archivo manualmente:**

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Haz clic en **`Archivo` → `Subir cuaderno`**
3. Selecciona el archivo `CCD_UNAB_GAN_Entrenamiento.ipynb` de tu PC

---

### Paso 1.2 — Activar GPU (¡obligatorio!)

> ⚠️ **Sin GPU el entrenamiento tardará horas.** Con GPU T4 tarda ~15-30 minutos.

1. En Colab, ve al menú **`Entorno de ejecución`**
2. Haz clic en **`Cambiar tipo de entorno de ejecución`**
3. En **Acelerador de hardware**, selecciona **`GPU`**
4. En **Tipo de GPU**, selecciona **`T4`** (gratuita)
5. Haz clic en **`Guardar`**

Verifica que la GPU esté activa ejecutando la primera celda:

```python
gpus = tf.config.list_physical_devices('GPU')
# Debe mostrar: ✅ GPU detectada: /physical_device:GPU:0
```

---

### Paso 1.3 — Ejecutar el notebook completo

1. Haz clic en **`Entorno de ejecución` → `Ejecutar todo`** (o `Ctrl + F9`)
2. El notebook descargará el dataset automáticamente desde Google Drive
3. Monitorea el progreso en la celda de entrenamiento:

```
🚀 Iniciando entrenamiento...
──────────────────────────────────────────────────
Época   1/100 | Loss G: 2.4312 | Loss D: 1.1823
Época   5/100 | Loss G: 1.8901 | Loss D: 0.9234
Época  10/100 | Loss G: 1.5234 | Loss D: 0.8712
...
Época 100/100 | Loss G: 0.9821 | Loss D: 0.7341
──────────────────────────────────────────────────
✅ Entrenamiento completado
```

> **Tip:** Cada 10 épocas verás una cuadrícula 4×4 con las imágenes generadas. Observa cómo mejoran progresivamente.

---

### Paso 1.4 — Ajustar número de épocas (opcional)

En la celda de entrenamiento puedes modificar:

```python
EPOCHS     = 100    # ← Cambia este número
                    # 50  = rápido, calidad básica
                    # 100 = recomendado (15-30 min con GPU T4)
                    # 200 = mejor calidad (~1 hora)

SHOW_EVERY = 10     # Mostrar imágenes cada N épocas
```

---

### Paso 1.5 — Descargar el modelo entrenado

Al final del notebook se ejecuta automáticamente:

```python
generador.save('generador_fifa_ccd.keras')
files.download('generador_fifa_ccd.keras')
```

El navegador iniciará la **descarga automática** del archivo `generador_fifa_ccd.keras`.

> Guarda este archivo en tu PC — lo necesitarás en la siguiente parte.

---

## 📦 Parte 2 — Guardar el modelo en GitHub

Tienes **dos opciones** según el tamaño del modelo. Primero verifica cuánto pesa:

- **Menos de 100 MB** → Opción A (GitHub directo)
- **Más de 100 MB** → Opción B (Git LFS) o Opción C (Google Drive)

---

### Opción A — Subir directamente a GitHub (< 100 MB)

#### Paso 2A.1 — Crear el repositorio en GitHub

1. Ve a [github.com](https://github.com) e inicia sesión
2. Haz clic en el botón **`+`** → **`New repository`**
3. Configura el repositorio:
   - **Repository name:** `gan-panini-fifa-ccd-unab` (o el nombre que prefieras)
   - **Description:** `DCGAN · Generador de tarjetas Panini FIFA 2026 · CCD·UNAB`
   - **Visibility:** `Public` ← importante para Streamlit Cloud gratuito
   - Activa **`Add a README file`**
4. Haz clic en **`Create repository`**

#### Paso 2A.2 — Clonar el repositorio en tu PC

Abre una terminal (Git Bash en Windows, Terminal en Mac/Linux):

```bash
git clone https://github.com/TU-USUARIO/gan-panini-fifa-ccd-unab.git
cd gan-panini-fifa-ccd-unab
```

#### Paso 2A.3 — Copiar los archivos al repositorio

Copia estos archivos a la carpeta del repositorio:

```
gan-panini-fifa-ccd-unab/
├── CCD_UNAB_GAN_Entrenamiento.ipynb   ← desde tu PC
├── app_panini_streamlit.py             ← desde tu PC
├── requirements.txt                    ← desde tu PC
├── generador_fifa_ccd.keras            ← descargado de Colab
└── README.md                           ← ya existe (reemplazar con este)
```

#### Paso 2A.4 — Subir todo a GitHub

```bash
git add .
git commit -m "feat: DCGAN entrenada + app Streamlit Panini FIFA 2026"
git push origin main
```

Verifica en tu repositorio de GitHub que aparezcan todos los archivos.

---

### Opción B — Git LFS para modelos grandes (> 100 MB)

Si el modelo supera 100 MB, GitHub bloqueará la subida. Usa **Git Large File Storage**:

#### Instalar Git LFS

```bash
# Windows (con Git for Windows ya incluido)
git lfs install

# Mac
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt install git-lfs
git lfs install
```

#### Configurar LFS para archivos .keras

Dentro de la carpeta del repositorio:

```bash
git lfs track "*.keras"
git add .gitattributes
git add generador_fifa_ccd.keras
git commit -m "feat: modelo DCGAN via Git LFS"
git push origin main
```

---

### Opción C — Hospedar el modelo en Google Drive (recomendado para modelos pesados)

Esta opción evita los límites de GitHub y es la más robusta:

#### Paso 2C.1 — Subir a Google Drive

1. Ve a [drive.google.com](https://drive.google.com)
2. Crea una carpeta llamada `modelos-ccd-unab`
3. Sube `generador_fifa_ccd.keras` a esa carpeta
4. Haz clic derecho sobre el archivo → **`Compartir`**
5. Cambia a **`Cualquier persona con el enlace`** → **`Lector`**
6. Haz clic en **`Copiar enlace`**

La URL tendrá esta forma:
```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnO/view?usp=sharing
                                  ↑
                          Este es tu FILE_ID
```

#### Paso 2C.2 — Actualizar la URL en la app

Abre `app_panini_streamlit.py` y reemplaza la variable `MODEL_URL`:

```python
# Antes:
MODEL_URL = "https://drive.google.com/uc?id=REEMPLAZA_CON_TU_FILE_ID"

# Después (usa tu FILE_ID real):
MODEL_URL = "https://drive.google.com/uc?id=1AbCdEfGhIjKlMnO"
```

Guarda el archivo y súbelo al repositorio de GitHub (sin el `.keras`).

---

## 🚀 Parte 3 — Despliegue en Streamlit Cloud

### Paso 3.1 — Crear cuenta en Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **`Sign up`**
3. Selecciona **`Continue with GitHub`**
4. Autoriza el acceso a tu cuenta de GitHub
5. Completa el formulario de registro (nombre, organización, etc.)

---

### Paso 3.2 — Crear la aplicación

1. En el dashboard de Streamlit, haz clic en **`New app`**
2. Selecciona **`From existing repo`**
3. Configura los campos:

| Campo | Valor |
|---|---|
| **Repository** | `TU-USUARIO/gan-panini-fifa-ccd-unab` |
| **Branch** | `main` |
| **Main file path** | `app_panini_streamlit.py` |
| **App URL** | `panini-ccd-unab` (personalizable) |

4. Haz clic en **`Deploy!`**

---

### Paso 3.3 — Esperar el despliegue

Streamlit instalará automáticamente las dependencias de `requirements.txt`. Verás los logs en tiempo real:

```
[18:32:01] 📦 Installing dependencies from requirements.txt
[18:32:15] ✓ streamlit==1.35.0
[18:32:28] ✓ tensorflow==2.15.0
[18:32:41] ✓ pillow==10.0.0
[18:32:45] ✓ numpy==1.24.0
[18:32:47] ✓ gdown==5.1.0
[18:32:48] 🚀 Starting app...
```

> El primer despliegue puede tardar **5–10 minutos** por la instalación de TensorFlow.

---

### Paso 3.4 — Primera ejecución de la app

Cuando la app arranque por primera vez:

1. Descargará automáticamente `generador_fifa_ccd.keras` desde GitHub o Google Drive
2. Cargará el modelo en memoria (caché con `@st.cache_resource`)
3. Mostrará la interfaz lista para usar

Tu app estará disponible en:
```
https://panini-ccd-unab.streamlit.app
```

> **Nota:** Streamlit Community Cloud gratuito puede "dormir" la app si no recibe visitas en varios días. Al volver a acceder, tardará ~30 segundos en despertar.

---

### Paso 3.5 — Actualizar la app (si cambias el código)

Cada vez que hagas `git push` al repositorio, Streamlit **actualizará la app automáticamente** en 1–2 minutos.

```bash
# Flujo de actualización
git add app_panini_streamlit.py
git commit -m "fix: mejora diseño tarjeta"
git push origin main
# → Streamlit detecta el cambio y redespliega automáticamente
```

---

## 🎮 Uso de la aplicación

Una vez desplegada, la interfaz tiene estas secciones:

### Panel izquierdo — Tu tarjeta

| Campo | Descripción | Ejemplo |
|---|---|---|
| **Nombre completo** | Aparece en la tarjeta en mayúsculas | `ALFREDO DÍAZ` |
| **Fecha de nacimiento** | Formato DD-MM-YYYY | `15-03-1990` |
| **Altura** | Texto libre | `1,78m` |
| **Peso** | Texto libre | `75 kg` |
| **Club / Universidad** | Equipo o institución | `UNAB FC (COL)` |
| **País (3 letras)** | Abreviatura ISO | `COL` |
| **Color acento** | Color de la banda lateral | Verde / Rojo / Azul |

### Slider de semilla

- Rango: **0 a 9999**
- Cada número produce un **rostro GAN diferente**
- La misma semilla siempre da la misma cara (reproducible)
- El rostro se actualiza en tiempo real mientras mueves el slider

### Panel derecho — Vista previa

- La tarjeta se regenera automáticamente al cambiar cualquier campo
- Botón **`⬇️ Descargar tarjeta (300 DPI)`** → PNG listo para imprimir

---

## 💡 Conceptos clave

### ¿Qué es una GAN?

Una **Red Generativa Adversaria** (*Generative Adversarial Network*) es una arquitectura de deep learning compuesta por dos redes que compiten entre sí:

```
Ruido z ~ N(0,1)
      ↓
 ┌─────────────┐        ┌───────────────────┐
 │  GENERADOR  │──────→ │  DISCRIMINADOR    │
 │  (crea      │  ¿real │  (¿es real o      │
 │   imágenes) │  o     │   generada?)      │
 └─────────────┘  falsa └───────────────────┘
                           ↑
                    Imágenes reales (FIFA)
```

- El **Generador** aprende a crear imágenes que engañen al Discriminador
- El **Discriminador** aprende a distinguir imágenes reales de falsas
- Al final del entrenamiento, el Generador produce imágenes realistas

### ¿Qué es `@tf.function`?

Decorador de TensorFlow que **compila** la función Python en un grafo computacional estático, logrando una aceleración de **2x–5x** en GPU respecto a la ejecución eager (línea por línea).

### ¿Qué es el espacio latente?

El vector `z` de 100 dimensiones es el **espacio latente** — un espacio matemático donde cada punto corresponde a una imagen posible. La semilla controla qué punto de ese espacio se muestrea, determinando el rostro generado.

### Label Smoothing

En lugar de entrenar con etiquetas duras (0 = falso, 1 = real), se usan valores suavizados:
- Real → **0.9** (en vez de 1.0)
- Falso → **0.1** (en vez de 0.0)

Esto evita que el Discriminador se vuelva demasiado seguro y estabiliza el entrenamiento.

---

## 📊 Arquitectura DCGAN

### Generador

| Capa | Salida | Activación |
|---|---|---|
| Dense + Reshape | 4 × 4 × 512 | ReLU |
| ConvTranspose2D + BN | 8 × 8 × 256 | ReLU |
| ConvTranspose2D + BN | 16 × 16 × 128 | ReLU |
| ConvTranspose2D + BN | 32 × 32 × 64 | ReLU |
| ConvTranspose2D | **64 × 64 × 3** | **Tanh** |

### Discriminador

| Capa | Salida | Activación |
|---|---|---|
| GaussianNoise | 64 × 64 × 3 | — |
| Conv2D + Dropout | 32 × 32 × 64 | LeakyReLU(0.2) |
| Conv2D + BN + Dropout | 16 × 16 × 128 | LeakyReLU(0.2) |
| Conv2D + BN + Dropout | 8 × 8 × 256 | LeakyReLU(0.2) |
| Conv2D + BN | 4 × 4 × 512 | LeakyReLU(0.2) |
| Flatten + Dense | **1 logit** | — |

---

## 🛠️ Solución de problemas frecuentes

### ❌ "No se detectó GPU" en Colab
→ Ve a `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)` y reconecta.

### ❌ El archivo .keras supera el límite de GitHub
→ Usa **Git LFS** (Opción B) o hospeda en **Google Drive** (Opción C).

### ❌ La app Streamlit no carga el modelo
→ Verifica que `MODEL_URL` en `app_panini_streamlit.py` tenga el FILE_ID correcto y que el archivo en Drive sea público.

### ❌ Streamlit muestra "ModuleNotFoundError"
→ Asegúrate de que `requirements.txt` esté en la raíz del repositorio y que los nombres de paquetes estén escritos correctamente.

### ❌ Las imágenes generadas se ven completamente grises o con ruido
→ El modelo necesita más épocas de entrenamiento. Aumenta `EPOCHS = 200` en el notebook.

---

## 👥 Créditos

| Rol | Detalle |
|---|---|
| **Institución** | Universidad Autónoma de Bucaramanga (UNAB) |
| **Unidad** | Centro de Competencias Digitales (CCD) |
| **Ciudad** | Bucaramanga, Colombia |
| **Modelo base** | DCGAN — Radford et al., 2015 |
| **Dataset** | Rostros de futbolistas FIFA (Google Drive) |
| **Framework** | TensorFlow 2.x / Keras |
| **Interfaz** | Streamlit Community Cloud |

---

> 🏆 **Taller práctico de Deep Learning Generativo**  
> *"De ruido aleatorio a tarjeta coleccionable en 100 épocas"*  
> CCD · UNAB · FIFA World Cup 2026
