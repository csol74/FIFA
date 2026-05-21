"""
🃏 CCD·UNAB — Generador de Tarjeta Panini FIFA World Cup 2026
Powered by DCGAN entrenada con rostros de futbolistas FIFA.

Dependencias (requirements.txt):
    streamlit
    tensorflow
    pillow
    numpy
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import os

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Panini FIFA 2026 · CCD·UNAB",
    page_icon="🃏",
    layout="centered",
)

# ─── Constantes ──────────────────────────────────────────────────────────────
# Ambos archivos están en la raíz del repositorio de GitHub.
# Streamlit Cloud clona el repo completo, así que los encuentra directamente.
Z_DIM        = 100
MODEL_PATH   = "generador_fifa_ccd.keras"
STICKER_PATH = "sticker.jpg"

# ─── Coordenadas calibradas del sticker (1197 × 1600 px) ─────────────────────
# Zona morada = donde va la foto del jugador
FOTO_X1, FOTO_Y1 = 395, 418      # esquina superior-izquierda de la zona foto
FOTO_X2, FOTO_Y2 = 868, 968      # esquina inferior-derecha
FOTO_W = FOTO_X2 - FOTO_X1       # 473 px
FOTO_H = FOTO_Y2 - FOTO_Y1       # 550 px

# Barras naranjas de texto
BARRA1_CY = 1350    # centro vertical barra naranja superior (nombre)
BARRA2_CY = 1468    # centro vertical barra naranja inferior (datos)
BARRA_X1  = 155     # inicio horizontal barra
BARRA_X2  = 940     # fin horizontal barra

# ─── Carga del modelo con caché ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando generador DCGAN…")
def cargar_modelo():
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return modelo

# ─── Carga del sticker de fondo ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando sticker…")
def cargar_sticker():
    return Image.open(STICKER_PATH).convert("RGB")

# ─── Generación de imagen por semilla ────────────────────────────────────────
def generar_imagen_gan(modelo, semilla: int) -> Image.Image:
    """Genera una cara de futbolista desde una semilla entera."""
    tf.random.set_seed(semilla)
    z = tf.random.normal([1, Z_DIM], seed=semilla)
    tensor = modelo(z, training=False)
    img_np = ((tensor[0].numpy() + 1.0) / 2.0)
    img_np = np.clip(img_np, 0, 1)
    return Image.fromarray((img_np * 255).astype(np.uint8))

# ─── Función que superpone la foto sobre el sticker ──────────────────────────
def componer_sticker(
    sticker_base: Image.Image,
    foto_jugador: Image.Image,
    nombre: str,
    datos: str,
    brillo_foto: float = 1.1,
) -> Image.Image:
    """
    Fusiona la foto del jugador sobre el sticker Panini real y
    escribe nombre y datos sobre las barras naranjas.
    """
    card = sticker_base.copy()
    sw, sh = card.size   # 1197 × 1600

    # ── 1. Preparar la foto: redimensionar y mejorar ──────────────────────────
    foto = foto_jugador.convert("RGB")
    foto = foto.resize((FOTO_W, FOTO_H), Image.LANCZOS)
    foto = ImageEnhance.Brightness(foto).enhance(brillo_foto)
    foto = ImageEnhance.Color(foto).enhance(1.25)
    foto = ImageEnhance.Sharpness(foto).enhance(1.4)

    # Suavizar bordes de la foto (evita el corte abrupto)
    mask = Image.new("L", (FOTO_W, FOTO_H), 255)
    md = ImageDraw.Draw(mask)
    radio = 18
    md.rectangle([0, 0, FOTO_W - 1, radio], fill=0)
    md.rectangle([0, 0, radio, FOTO_H - 1], fill=0)
    md.rectangle([FOTO_W - radio, 0, FOTO_W - 1, FOTO_H - 1], fill=0)
    md.rectangle([0, FOTO_H - radio, FOTO_W - 1, FOTO_H - 1], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))

    # Pegar foto sobre el sticker en la zona calibrada
    card.paste(foto, (FOTO_X1, FOTO_Y1), mask)

    # ── 2. Escribir textos sobre las barras naranjas ──────────────────────────
    draw = ImageDraw.Draw(card)

    # Intentar fuente bold; fallback a default
    try:
        fuente_nm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        fuente_dt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
    except Exception:
        fuente_nm = ImageFont.load_default()
        fuente_dt = ImageFont.load_default()

    # Nombre en barra superior (centrado)
    texto_nm = nombre.upper()
    bb = draw.textbbox((0, 0), texto_nm, font=fuente_nm)
    tw = bb[2] - bb[0]
    tx = (sw - tw) // 2
    ty = BARRA1_CY - (bb[3] - bb[1]) // 2 - 4
    # Sombra
    draw.text((tx + 2, ty + 2), texto_nm, font=fuente_nm, fill=(60, 20, 0))
    draw.text((tx, ty), texto_nm, font=fuente_nm, fill=(255, 255, 255))

    # Datos en barra inferior (centrado)
    texto_dt = datos.upper()
    bb2 = draw.textbbox((0, 0), texto_dt, font=fuente_dt)
    tw2 = bb2[2] - bb2[0]
    tx2 = (sw - tw2) // 2
    ty2 = BARRA2_CY - (bb2[3] - bb2[1]) // 2 - 4
    draw.text((tx2 + 2, ty2 + 2), texto_dt, font=fuente_dt, fill=(60, 20, 0))
    draw.text((tx2, ty2), texto_dt, font=fuente_dt, fill=(255, 248, 180))

    return card


# ═══════════════════════════════════════════════════════════════════════════════
#   INTERFAZ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🃏 Generador de Tarjeta Panini FIFA 2026")
st.caption("Centro de Competencias Digitales · UNAB · Powered by DCGAN")

st.markdown("""
> Elige si quieres que tu tarjeta muestre un **rostro generado por IA** (DCGAN)
> o una **foto tuya propia**. Luego personaliza los textos y descarga tu sticker.
""")

# ── Carga de recursos ─────────────────────────────────────────────────────────
sticker_base = cargar_sticker()
if sticker_base is None:
    st.error("⚠️ No se encontró `sticker.jpg` en el repositorio. "
             "Asegúrate de subir el archivo al repo de GitHub.")
    st.stop()

# ── Sección 1: Fuente de la imagen del jugador ───────────────────────────────
st.markdown("---")
st.subheader("📸 Paso 1 — Elige la fuente de la imagen")

modo_imagen = st.radio(
    "¿Qué imagen quieres usar en tu sticker?",
    options=["🤖 Rostro generado por IA (DCGAN)", "🖼️ Mi propia foto"],
    horizontal=True,
)

foto_jugador = None   # imagen PIL final que irá al sticker

# ── Modo A: GAN ───────────────────────────────────────────────────────────────
if modo_imagen == "🤖 Rostro generado por IA (DCGAN)":

    try:
        modelo = cargar_modelo()
        st.success("✅ Generador DCGAN cargado")
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {e}")
        st.info("Configura `MODEL_URL` con el enlace de tu modelo en Google Drive.")
        st.stop()

    st.markdown("**Mueve el slider para explorar distintos rostros generados por la red neuronal.**")
    semilla = st.slider("🎲 Semilla del generador", 0, 2000, 42, step=1)

    col_gan1, col_gan2 = st.columns([1, 2])
    with col_gan1:
        img_gan = generar_imagen_gan(modelo, semilla)
        # Preview 2x
        st.image(img_gan.resize((192, 192), Image.NEAREST),
                 caption=f"Semilla {semilla}", width=192)
    with col_gan2:
        st.info(
            "💡 **¿Cómo funciona?**\n\n"
            "Cada semilla produce un vector de ruido diferente en el espacio latente "
            "(ℝ¹⁰⁰). El Generador convierte ese vector en una imagen de 64×64 px. "
            "La misma semilla siempre da la misma cara."
        )

    foto_jugador = img_gan

# ── Modo B: Foto propia ───────────────────────────────────────────────────────
else:
    st.markdown("**Sube una foto de tu cara o tómala con la cámara.**")

    metodo_foto = st.radio(
        "Método:",
        ["📁 Subir imagen", "📷 Cámara web"],
        horizontal=True,
        label_visibility="collapsed",
    )

    foto_raw = None
    if metodo_foto == "📁 Subir imagen":
        archivo = st.file_uploader(
            "Selecciona una imagen (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if archivo:
            foto_raw = Image.open(archivo).convert("RGB")
    else:
        captura = st.camera_input("Toma la foto", label_visibility="collapsed")
        if captura:
            foto_raw = Image.open(captura).convert("RGB")

    if foto_raw is not None:
        # Recorte cuadrado centrado (face crop simple)
        w_f, h_f = foto_raw.size
        lado = min(w_f, h_f)
        left  = (w_f - lado) // 2
        top   = (h_f - lado) // 2
        foto_raw = foto_raw.crop((left, top, left + lado, top + lado))

        col_fp1, col_fp2 = st.columns([1, 2])
        with col_fp1:
            st.image(foto_raw.resize((192, 192), Image.LANCZOS),
                     caption="Tu foto (recortada cuadrada)", width=192)
        with col_fp2:
            st.info(
                "✅ Foto cargada. La imagen se recortará de forma cuadrada "
                "y se ajustará a la zona del sticker automáticamente."
            )
        foto_jugador = foto_raw
    else:
        st.warning("Sube o captura una foto para continuar.")

# ── Sección 2: Datos personales ───────────────────────────────────────────────
st.markdown("---")
st.subheader("✏️ Paso 2 — Personaliza tu tarjeta")

col_f1, col_f2 = st.columns(2)
with col_f1:
    nombre    = st.text_input("Nombre del jugador", value="TU NOMBRE",
                              max_chars=24)
    fecha_nac = st.text_input("Fecha de nacimiento", value="01-01-2000")
    altura    = st.text_input("Altura", value="1,75m")
with col_f2:
    peso      = st.text_input("Peso", value="70 kg")
    club      = st.text_input("Club / Universidad", value="UNAB FC")
    pais      = st.text_input("País (3 letras)", value="COL", max_chars=3)

brillo = st.slider("☀️ Brillo de la foto en el sticker", 0.7, 1.5, 1.1, step=0.05)

# Texto de la barra inferior (datos)
datos_texto = f"{fecha_nac}  ·  {altura}  ·  {peso}  ·  {club}  ·  {pais.upper()}"

# ── Sección 3: Generar y previsualizar ───────────────────────────────────────
st.markdown("---")
st.subheader("🃏 Paso 3 — Tu sticker Panini")

if foto_jugador is None:
    st.info("Completa el Paso 1 para generar el sticker.")
else:
    with st.spinner("Generando tu sticker…"):
        tarjeta = componer_sticker(
            sticker_base  = sticker_base,
            foto_jugador  = foto_jugador,
            nombre        = nombre,
            datos         = datos_texto,
            brillo_foto   = brillo,
        )

    # Mostrar a tamaño razonable en pantalla (reducida al 35%)
    ancho_display = 420
    alto_display  = int(tarjeta.height * ancho_display / tarjeta.width)
    st.image(
        tarjeta.resize((ancho_display, alto_display), Image.LANCZOS),
        caption="Vista previa de tu sticker Panini FIFA 2026",
        use_column_width=False,
    )

    # ── Botón de descarga ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    # Guardar en tamaño real (1197 × 1600) para imprimir
    tarjeta.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)

    nombre_archivo = f"sticker_panini_{nombre.replace(' ', '_')}_{pais.upper()}.png"
    st.download_button(
        label="⬇️ Descargar sticker en alta resolución (300 DPI)",
        data=buf,
        file_name=nombre_archivo,
        mime="image/png",
        use_container_width=True,
    )

# ── Sección educativa ─────────────────────────────────────────────────────────
with st.expander("💡 ¿Cómo funciona la DCGAN?", expanded=False):
    st.markdown("""
    ### Del ruido a la cara

    ```
    Semilla (entero)
        ↓
    tf.random.set_seed(semilla)
    z = tf.random.normal([1, 100])    ← vector en espacio latente ℝ¹⁰⁰
        ↓
    GENERADOR (red neuronal DCGAN)
    z(100) → 4×4×512 → 8×8 → 16×16 → 32×32 → 64×64×3
        ↓
    Imagen RGB 64×64 → redimensionada → sticker Panini
    ```

    | Capa | Salida |
    |------|--------|
    | Dense + Reshape | 4 × 4 × 512 |
    | ConvTranspose + BN + ReLU | 8 × 8 × 256 |
    | ConvTranspose + BN + ReLU | 16 × 16 × 128 |
    | ConvTranspose + BN + ReLU | 32 × 32 × 64 |
    | ConvTranspose + **Tanh** | **64 × 64 × 3** |
    """)

st.markdown("---")
st.caption("🏫 Centro de Competencias Digitales · UNAB · Bucaramanga, Colombia")
