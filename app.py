# ============================================================
# CABECERA
# ============================================================
# Alumno: Diego Vallejo Mijallo
# URL Streamlit Cloud: https://diego-vallejo-mda13-bc5.streamlit.app
# URL GitHub: https://github.com/Mijayo/ISDI_BC5_MD13.git

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Ruta del dataset
BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / 'streaming_history.json'

# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente de análisis de datos de Spotify. Tu tarea es responder preguntas sobre los hábitos de escucha del usuario generando código Python con Plotly.

## DATOS DISPONIBLES

Tienes acceso a un DataFrame de pandas llamado `df` con los siguientes datos:
- Rango de fechas: {fecha_min} hasta {fecha_max}
- Plataformas presentes: {plataformas}
- Valores posibles de reason_start: {reason_start_values}
- Valores posibles de reason_end: {reason_end_values}
- Total de sesiones de escucha: {total_sesiones}
- Artistas únicos en el historial: {artistas_unicos}

### Columnas del DataFrame

| Columna             | Tipo    | Descripción |
|---------------------|---------|-------------|
| ts                  | datetime (UTC) | Timestamp exacto de la reproducción |
| hora                | int     | Hora del día (0–23) |
| dia_semana          | str     | Día en inglés: 'Monday', 'Tuesday'... |
| mes                 | int     | Mes del año (1–12) |
| año                 | int     | Año de la reproducción |
| franja              | str     | 'Madrugada' (0-6h), 'Mañana' (6-12h), 'Tarde' (12-18h), 'Noche' (18-21h), 'Noche tardía' (21-24h) |
| es_finde            | bool    | True si es sábado o domingo |
| track               | str     | Nombre de la canción |
| artista             | str     | Nombre del artista |
| album               | str     | Nombre del álbum |
| ms_played           | int     | Milisegundos reproducidos |
| minutos_escuchados  | float   | Minutos reproducidos |
| skipped             | bool    | True si el usuario saltó la canción |
| escucha_completa    | bool    | True si la canción terminó de forma natural (reason_end == 'trackdone') |
| shuffle             | bool    | True si el modo aleatorio estaba activado |
| platform            | str     | Plataforma original (texto libre) |
| plataforma          | str     | Plataforma simplificada: 'ios', 'android', 'desktop' o 'web' |
| reason_start        | str     | Motivo por el que empezó la canción |
| reason_end          | str     | Motivo por el que terminó la canción |
| hora_local          | int     | Hora real en España (Europe/Madrid), usa esta en lugar de `hora` para análisis temporales |
| temporada           | str     | 'Invierno', 'Primavera', 'Verano' u 'Otoño' |
| sesion_id           | int     | Identificador de sesión de escucha (pausa > 30 min = nueva sesión) |
| duracion_sesion_min | float   | Minutos totales de la sesión a la que pertenece cada reproducción |
| primera_vez_artista | bool    | True si es la primera reproducción de ese artista en todo el historial |

## INSTRUCCIONES

1. Analiza la pregunta del usuario sobre sus datos de Spotify.
2. Si la pregunta es sobre los datos, genera código Python con Plotly que cree una figura llamada `fig`.
3. Si la pregunta NO tiene relación con los datos de Spotify, responde con tipo "fuera_de_alcance".

### Reglas para el código generado
- Usa SOLO las variables disponibles: `df`, `pd`, `px`, `go`. NO importes ni uses ninguna otra librería ni función (no uses `make_subplots`, no uses `import`)
- Si la pregunta tiene varias partes, responde con UN solo gráfico que capture la idea principal
- La figura SIEMPRE debe guardarse en una variable llamada `fig`
- Usa `px` (plotly.express) para gráficos simples y `go` (plotly.graph_objects) para los complejos
- Añade siempre un título descriptivo al gráfico
- No uses `fig.show()` — la app ya se encarga de renderizarlo
### Patrones pandas correctos

Para contar reproducciones por artista:
```python
top = df.groupby("artista")["track"].count().reset_index()
top.columns = ["artista", "reproducciones"]
```

Para calcular una tasa (ej. tasa de skip):
```python
tasa = df.groupby("dia_semana")["skipped"].mean().reset_index()
tasa.columns = ["dia_semana", "tasa_skip"]
```

Para un heatmap por dos dimensiones:
```python
pivot = df.groupby(["dia_semana", "hora_local"])["skipped"].mean().reset_index()
pivot.columns = ["dia_semana", "hora_local", "tasa_skip"]
fig = px.density_heatmap(pivot, x="hora_local", y="dia_semana", z="tasa_skip")
```

IMPORTANTE: después de un groupby + reset_index(), comprueba siempre que los nombres de columna coinciden con los que usas en px.

## FORMATO DE RESPUESTA

Responde SIEMPRE con un JSON válido, sin texto adicional, sin markdown, sin backticks:

Si la pregunta es sobre los datos:
{{"tipo": "grafico", "codigo": "fig = px.bar(...)", "interpretacion": "Texto breve explicando el resultado"}}

Si la pregunta está fuera de alcance:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Texto explicando que solo puedes analizar datos de Spotify"}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json(DATASET)
    # DEBUG CARGA DATOS 
    # print(f'{df.sample(10)}')

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    # 1. Timestamp → datetime UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Columnas derivadas de tiempo
    df["hora"]       = df["ts"].dt.hour
    df["dia_semana"] = df["ts"].dt.day_name()   # 'Monday', 'Tuesday'...
    df["mes"]        = df["ts"].dt.month
    df["año"]        = df["ts"].dt.year

    # 3. Milisegundos → minutos
    df["minutos_escuchados"] = df["ms_played"] / 60_000

    # 4. Renombrar columnas largas
    df = df.rename(columns={
        "master_metadata_track_name":          "track",
        "master_metadata_album_artist_name":   "artista",
        "master_metadata_album_album_name":    "album",
    })

    # 5. Filtrar registros sin canción (podcasts, etc.)
    df = df.dropna(subset=["track", "artista"])

    # 6. skipped → booleano limpio
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # 7. Escucha completa (reason_end == "trackdone")
    df["escucha_completa"] = df["reason_end"] == "trackdone"

    # 8. Franja horaria del día
    df["franja"] = pd.cut(
        df["hora"],
        bins=[0, 6, 12, 18, 21, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche", "Noche tardía"],
        right=False
    ).astype(str)

    # 9. Fin de semana vs. entre semana
    df["es_finde"] = df["ts"].dt.dayofweek >= 5

    # 10. Plataforma simplificada
    df["plataforma"] = df["platform"].str.lower().str.extract(r"(ios|android|desktop|web)", expand=False)

    # 11. Hora local (España, Europe/Madrid)
    df["ts_local"]   = df["ts"].dt.tz_convert("Europe/Madrid")
    df["hora_local"] = df["ts_local"].dt.hour

    # 12. Temporada del año
    def get_temporada(mes):
        if mes in [12, 1, 2]: return "Invierno"
        if mes in [3, 4, 5]:  return "Primavera"
        if mes in [6, 7, 8]:  return "Verano"
        return "Otoño"
    df["temporada"] = df["mes"].apply(get_temporada)

    # 13. Sesión de escucha (pausa > 30 min = nueva sesión)
    df = df.sort_values("ts").reset_index(drop=True)
    gap = df["ts"].diff() > pd.Timedelta(minutes=30)
    df["sesion_id"] = gap.cumsum()

    # 14. Duración total de cada sesión (minutos)
    df["duracion_sesion_min"] = df.groupby("sesion_id")["minutos_escuchados"].transform("sum")

    # 15. Primera vez que aparece un artista en el historial
    df["primera_vez_artista"] = ~df.duplicated(subset=["artista"], keep="first")

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min        = df["ts"].min()
    fecha_max        = df["ts"].max()
    plataformas      = df["plataforma"].dropna().unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values   = df["reason_end"].unique().tolist()
    total_sesiones   = df["sesion_id"].nunique()
    artistas_unicos  = df["artista"].nunique()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        total_sesiones=total_sesiones,
        artistas_unicos=artistas_unicos,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                # st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")
                st.error(f"Error: {e}")
                if 'parsed' in locals():
                    st.code(parsed.get("codigo", ""), language="python")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La app sigue una arquitectura modular: cada función tiene una
#    responsabilidad única y aislada. El flujo arranca en "App principal".
#    El LLM recibe dos inputs vía get_response(): el system prompt con
#    la descripción del dataset (columnas, rangos, valores posibles) y
#    la pregunta del usuario. No recibe los datos directamente porque
#    14.983 registros no caben en el contexto y harían cada llamada
#    lenta y costosa. En su lugar, devuelve un JSON con dos campos:
#    "codigo" (Python como string) e "interpretacion" (texto).
#    execute_chart() ejecuta ese código con exec() en un entorno
#    controlado donde solo existen df, pd, px y go, y devuelve
#    una figura Plotly que Streamlit renderiza en pantalla.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    Le damos informacion del dataset en el system prompt sin enviarle 
#    el total de registros pasandole la descripcion de las columnas. Si no 
#    pasamos los campos de fechas (fecha_min, fecha_max) y los quitamos de 
#    la descripcion del SYSTEM_PROMPT, no podriamos hacer preguntas tipo 
#    "¿cuánto llevo escuchando música este año?" porque el LLM no sabe el 
#    rango del dataset y podría filtrar por fechas incorrectas. 
#    Si quitásemos la tabla de columnas: cualquier pregunta sobre artistas 
#    o canciones, porque el LLM no sabría que las columnas se llaman "artista" 
#    y "track" e inventaría nombres como "artist_name" haciendo que el código falle.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    El usuario escribe una pregunta en el chat. Streamlit llama a
#    build_prompt() que inyecta los datos reales del dataset en el
#    system prompt (fechas, plataformas, sesiones...). get_response()
#    envía system prompt + pregunta a GPT-4.1-mini vía API de OpenAI.
#    El modelo devuelve un JSON con "tipo", "codigo" e "interpretacion".
#    parse_response() limpia y convierte ese string a diccionario Python.
#    Si tipo es "grafico", execute_chart() ejecuta el código con exec()
#    en un entorno aislado (df, pd, px, go) y obtiene una figura Plotly.
#    Streamlit renderiza el gráfico, la interpretación y el código.
#    Si tipo es "fuera_de_alcance", se muestra solo el texto.