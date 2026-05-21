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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import io
import os

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Panini FIFA 2026 · CCD·UNAB",
    page_icon="🃏",
    layout="wide",
)

# ─── Constantes ──────────────────────────────────────────────────────────────
Z_DIM        = 100
MODEL_PATH   = "generador_fifa_ccd.keras"
STICKER_PATH = "sticker.jpg"

# ─── Coordenadas calibradas del sticker (1197 × 1600 px) ─────────────────────
# Zona foto ampliada ×1.6 respecto al rectángulo morado original
FOTO_X1, FOTO_Y1 = 253, 253
FOTO_X2, FOTO_Y2 = 1009, 1133
FOTO_W = FOTO_X2 - FOTO_X1     # 756 px
FOTO_H = FOTO_Y2 - FOTO_Y1     # 880 px

# Barras naranjas de texto (medidas reales del sticker)
BARRA1_Y1, BARRA1_Y2 = 1295, 1420   # barra superior → NOMBRE
BARRA2_Y1, BARRA2_Y2 = 1430, 1530   # barra inferior → datos

# ─── Carga del modelo con caché ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando generador DCGAN…")
def cargar_modelo():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# ─── Carga del sticker de fondo ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando sticker…")
def cargar_sticker():
    return Image.open(STICKER_PATH).convert("RGB")

# ─── Generación de imagen por semilla ────────────────────────────────────────
def generar_imagen_gan(modelo, semilla: int) -> Image.Image:
    tf.random.set_seed(semilla)
    z = tf.random.normal([1, Z_DIM], seed=semilla)
    tensor = modelo(z, training=False)
    img_np = np.clip((tensor[0].numpy() + 1.0) / 2.0, 0, 1)
    return Image.fromarray((img_np * 255).astype(np.uint8))

# ─── Filtros de fondo aplicables a la foto ───────────────────────────────────
def aplicar_filtro(img: Image.Image, filtro: str) -> Image.Image:
    """Aplica un filtro de fondo/estilo a la imagen del jugador."""
    img = img.convert("RGB")
    if filtro == "🎨 Sin filtro":
        return img
    elif filtro == "🔵 Fondo azul deportivo":
        # Separa fondo oscuro y lo tinta azul intenso
        enhanced = ImageEnhance.Color(img).enhance(1.5)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
        overlay = Image.new("RGB", img.size, (10, 40, 120))
        # Mezcla suave: zonas oscuras del original se ven azules
        arr = np.array(enhanced).astype(float)
        ov  = np.array(overlay).astype(float)
        luminance = (arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114) / 255
        alpha = np.clip(1.0 - luminance, 0, 1)[:,:, np.newaxis] * 0.55
        blended = arr * (1 - alpha) + ov * alpha
        return Image.fromarray(blended.clip(0,255).astype(np.uint8))
    elif filtro == "🟡 Alto contraste FIFA":
        img2 = ImageEnhance.Contrast(img).enhance(1.8)
        img2 = ImageEnhance.Color(img2).enhance(1.6)
        img2 = ImageEnhance.Sharpness(img2).enhance(2.0)
        return img2
    elif filtro == "⚫ Blanco y negro":
        gris = ImageOps.grayscale(img)
        gris = ImageEnhance.Contrast(gris).enhance(1.4)
        return gris.convert("RGB")
    elif filtro == "🟠 Sepia vintage":
        gris = ImageOps.grayscale(img)
        arr  = np.array(gris).astype(float)
        r = np.clip(arr * 1.1,  0, 255)
        g = np.clip(arr * 0.87, 0, 255)
        b = np.clip(arr * 0.65, 0, 255)
        sepia = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(sepia)
    elif filtro == "🌟 Neón FIFA":
        arr  = np.array(img).astype(float)
        # Boost de canal verde y azul, bajamos rojo → look FIFA teal/cyan
        arr[:,:,0] = np.clip(arr[:,:,0] * 0.5, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * 1.3, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * 1.5, 0, 255)
        img2 = Image.fromarray(arr.astype(np.uint8))
        img2 = ImageEnhance.Contrast(img2).enhance(1.4)
        return img2
    elif filtro == "🔴 Fondo rojo Colombia":
        arr = np.array(img).astype(float)
        ov  = np.array(Image.new("RGB", img.size, (180, 10, 20))).astype(float)
        luminance = (arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114) / 255
        alpha = np.clip(1.0 - luminance, 0, 1)[:,:, np.newaxis] * 0.6
        blended = arr * (1 - alpha) + ov * alpha
        img2 = Image.fromarray(blended.clip(0,255).astype(np.uint8))
        return ImageEnhance.Contrast(img2).enhance(1.2)
    return img

# ─── Composición del sticker ─────────────────────────────────────────────────
def componer_sticker(
    sticker_base : Image.Image,
    foto_jugador : Image.Image,
    nombre       : str,
    fecha_nac    : str,
    altura       : str,
    peso         : str,
    club         : str,
    pais         : str,
    brillo       : float,
    filtro       : str,
) -> Image.Image:

    card = sticker_base.copy()
    sw, sh = card.size   # 1197 × 1600

    # ── 1. Aplicar filtro y ajustar foto ─────────────────────────────────────
    foto = aplicar_filtro(foto_jugador.convert("RGB"), filtro)
    foto = foto.resize((FOTO_W, FOTO_H), Image.LANCZOS)
    foto = ImageEnhance.Brightness(foto).enhance(brillo)
    foto = ImageEnhance.Color(foto).enhance(1.2)
    foto = ImageEnhance.Sharpness(foto).enhance(1.3)

    # ── 2. Máscara de bordes suavizados ──────────────────────────────────────
    mask = Image.new("L", (FOTO_W, FOTO_H), 255)
    md   = ImageDraw.Draw(mask)
    r    = 22   # radio de esquinas
    # Bordes superior, izquierdo, derecho, inferior → negros para fade
    for i in range(r):
        t = int(255 * (i / r) ** 2)   # gradiente cuadrático
        inv = 255 - t
        md.rectangle([0,         i,         FOTO_W,     i+1     ], fill=t)
        md.rectangle([0,         FOTO_H-i-1, FOTO_W,    FOTO_H-i], fill=t)
        md.rectangle([i,         0,         i+1,        FOTO_H  ], fill=t)
        md.rectangle([FOTO_W-i-1, 0,        FOTO_W-i,  FOTO_H  ], fill=t)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=14))

    card.paste(foto, (FOTO_X1, FOTO_Y1), mask)

    # ── 3. Cargar fuentes ─────────────────────────────────────────────────────
    def fuente(size, bold=True):
        nombre_f = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(
                f"/usr/share/fonts/truetype/dejavu/{nombre_f}", size)
        except Exception:
            return ImageFont.load_default()

    fn_nombre = fuente(200, bold=True)
    fn_dato   = fuente(150, bold=True)
    fn_label  = fuente(10, bold=False)

    draw = ImageDraw.Draw(card)

    # ── 4. Helper: texto centrado con sombra ─────────────────────────────────
    def texto_centrado(draw, texto, cy, font, color_texto, color_sombra=(40,15,0)):
        bb  = draw.textbbox((0, 0), texto, font=font)
        tw  = bb[2] - bb[0]
        th  = bb[3] - bb[1]
        tx  = (sw - tw) // 2
        ty  = cy - th // 2
        draw.text((tx+2, ty+2), texto, font=font, fill=color_sombra)
        draw.text((tx,   ty  ), texto, font=font, fill=color_texto)

    # ── 5. BARRA 1 → NOMBRE ──────────────────────────────────────────────────
    cy1 = (BARRA1_Y1 + BARRA1_Y2) // 2
    texto_centrado(draw, nombre.upper(), cy1, fn_nombre,
                   color_texto=(255, 255, 255))

    # ── 6. BARRA 2 → DATOS en dos líneas ─────────────────────────────────────
    # La barra 2 tiene ~100px de alto; usamos fuente pequeña para dos líneas
    # Línea A: fecha · altura · peso
    # Línea B: club · país
    barra2_h   = BARRA2_Y2 - BARRA2_Y1   # ≈100 px
    margen_v   = 6

    linea_a = f"{fecha_nac}   {altura}   {peso}"
    linea_b = f"{club.upper()}   ·   {pais.upper()}"

    # Ajustar tamaño para que ambas quepan
    fn_d = fuente(30, bold=True)

    bb_a = draw.textbbox((0,0), linea_a, font=fn_d)
    bb_b = draw.textbbox((0,0), linea_b, font=fn_d)
    th_a = bb_a[3] - bb_a[1]
    th_b = bb_b[3] - bb_b[1]
    total_h = th_a + margen_v + th_b

    # Centrar verticalmente el bloque de dos líneas en la barra
    bloque_top = BARRA2_Y1 + (barra2_h - total_h) // 2

    # Línea A
    tw_a = bb_a[2] - bb_a[0]
    tx_a = (sw - tw_a) // 2
    draw.text((tx_a+2, bloque_top+2),   linea_a, font=fn_d, fill=(40,15,0))
    draw.text((tx_a,   bloque_top),     linea_a, font=fn_d, fill=(255, 248, 180))

    # Línea B
    tw_b = bb_b[2] - bb_b[0]
    tx_b = (sw - tw_b) // 2
    ty_b = bloque_top + th_a + margen_v
    draw.text((tx_b+2, ty_b+2), linea_b, font=fn_d, fill=(40,15,0))
    draw.text((tx_b,   ty_b  ), linea_b, font=fn_d, fill=(255, 220, 80))

    return card


# ═══════════════════════════════════════════════════════════════════════════════
#   INTERFAZ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🃏 Generador de Tarjeta Panini FIFA 2026")
st.caption("Centro de Competencias Digitales · UNAB · Powered by DCGAN")

sticker_base = cargar_sticker()

# ── Paso 1: Fuente de imagen ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📸 Paso 1 — Imagen del jugador")

modo_imagen = st.radio(
    "¿Qué imagen usarás?",
    ["🤖 Rostro generado por IA (DCGAN)", "🖼️ Mi propia foto"],
    horizontal=True,
)

foto_jugador = None

if modo_imagen == "🤖 Rostro generado por IA (DCGAN)":
    modelo = cargar_modelo()
    st.success("✅ Generador DCGAN listo")

    semilla = st.slider("🎲 Semilla — cada número produce un rostro diferente",
                        0, 2000, 42, step=1)

    img_gan = generar_imagen_gan(modelo, semilla)

    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        # Mostrar preview más grande (×4)
        st.image(img_gan.resize((256, 256), Image.NEAREST),
                 caption=f"Semilla {semilla}", width=256)
    with col_g2:
        st.info(
            "💡 **Espacio latente:** cada semilla genera un vector de ruido z ∈ ℝ¹⁰⁰. "
            "El Generador transforma ese ruido en una imagen 64×64 px. "
            "La misma semilla → siempre la misma cara."
        )
    foto_jugador = img_gan

else:
    metodo = st.radio("Método:", ["📁 Subir imagen", "📷 Cámara web"],
                      horizontal=True, label_visibility="collapsed")
    foto_raw = None
    if metodo == "📁 Subir imagen":
        arch = st.file_uploader("Sube tu foto (JPG, PNG)",
                                type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
        if arch:
            foto_raw = Image.open(arch).convert("RGB")
    else:
        cap = st.camera_input("Toma la foto", label_visibility="collapsed")
        if cap:
            foto_raw = Image.open(cap).convert("RGB")

    if foto_raw is not None:
        w_f, h_f = foto_raw.size
        lado = min(w_f, h_f)
        foto_raw = foto_raw.crop(((w_f-lado)//2, (h_f-lado)//2,
                                   (w_f+lado)//2, (h_f+lado)//2))
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            st.image(foto_raw.resize((256, 256), Image.LANCZOS),
                     caption="Tu foto", width=256)
        with col_p2:
            st.info("✅ Foto lista. Se ajustará automáticamente a la zona del sticker.")
        foto_jugador = foto_raw
    else:
        st.warning("Sube o captura una foto para continuar.")

# ── Paso 2: Filtro y ajustes ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎨 Paso 2 — Filtro y ajustes de imagen")

filtro = st.selectbox(
    "Filtro de imagen",
    [
        "🎨 Sin filtro",
        "🔵 Fondo azul deportivo",
        "🟡 Alto contraste FIFA",
        "⚫ Blanco y negro",
        "🟠 Sepia vintage",
        "🌟 Neón FIFA",
        "🔴 Fondo rojo Colombia",
    ]
)

brillo = st.slider("☀️ Brillo", 0.7, 1.6, 1.1, step=0.05)

# ── Paso 3: Datos del jugador ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("✏️ Paso 3 — Datos del jugador")

col_a, col_b, col_c = st.columns(3)
with col_a:
    nombre    = st.text_input("👤 Nombre del jugador",    value="TU NOMBRE",  max_chars=22)
    fecha_nac = st.text_input("📅 Fecha de nacimiento",   value="01-01-2000")
with col_b:
    altura    = st.text_input("📏 Altura",                value="1,75 m")
    peso      = st.text_input("⚖️ Peso",                  value="70 kg")
with col_c:
    club      = st.text_input("🏟️ Club / Universidad",    value="UNAB FC")
    pais      = st.text_input("🌍 País (3 letras)",        value="COL", max_chars=3)

# ── Paso 4: Generar sticker ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🃏 Paso 4 — Tu sticker Panini")

if foto_jugador is None:
    st.info("↑ Completa el Paso 1 para generar el sticker.")
else:
    with st.spinner("Generando tu sticker…"):
        tarjeta = componer_sticker(
            sticker_base = sticker_base,
            foto_jugador = foto_jugador,
            nombre       = nombre,
            fecha_nac    = fecha_nac,
            altura       = altura,
            peso         = peso,
            club         = club,
            pais         = pais,
            brillo       = brillo,
            filtro       = filtro,
        )

    # Mostrar centrado y grande (60% del ancho original = 718px)
    ancho_display = 680
    alto_display  = int(tarjeta.height * ancho_display / tarjeta.width)

    col_left, col_mid, col_right = st.columns([1, 4, 1])
    with col_mid:
        st.image(
            tarjeta.resize((ancho_display, alto_display), Image.LANCZOS),
            caption="Vista previa — descarga en alta resolución abajo",
            use_column_width=False,
        )

    # Descarga en resolución original
    buf = io.BytesIO()
    tarjeta.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)

    nombre_archivo = f"sticker_{nombre.replace(' ','_')}_{pais.upper()}.png"
    st.download_button(
        label     = "⬇️ Descargar sticker en alta resolución (300 DPI — 1197×1600 px)",
        data      = buf,
        file_name = nombre_archivo,
        mime      = "image/png",
        use_container_width = True,
    )

# ── Info educativa ────────────────────────────────────────────────────────────
with st.expander("💡 ¿Cómo funciona la DCGAN?", expanded=False):
    st.markdown("""
    ### Del número al rostro

    ```
    Semilla (ej: 42)
        ↓  tf.random.set_seed(42)
    z = vector aleatorio ∈ ℝ¹⁰⁰   ← espacio latente
        ↓
    GENERADOR DCGAN
    z(100) → Dense → 4×4×512 → 8×8 → 16×16 → 32×32 → 64×64×3
        ↓
    Imagen RGB 64×64 → upscale a zona sticker → Panini
    ```

    | Capa | Salida | Activación |
    |---|---|---|
    | Dense + Reshape | 4 × 4 × 512 | ReLU |
    | ConvTranspose + BN | 8 × 8 × 256 | ReLU |
    | ConvTranspose + BN | 16 × 16 × 128 | ReLU |
    | ConvTranspose + BN | 32 × 32 × 64 | ReLU |
    | ConvTranspose | **64 × 64 × 3** | **Tanh** |
    """)

st.markdown("---")
st.caption("🏫 Centro de Competencias Digitales · UNAB · Bucaramanga, Colombia")
