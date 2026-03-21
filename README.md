# BC5 — Spotify Analytics Assistant

**Alumno:** Diego Vallejo Mijallo
**Programa:** MDA13 — ISDI
**App desplegada:** https://diego-vallejo-mda13-bc5.streamlit.app

---

## Descripción

Aplicación web que permite analizar hábitos de escucha de Spotify mediante lenguaje natural. El usuario escribe una pregunta en un chat y la app responde con un gráfico interactivo generado automáticamente por un LLM.

## Arquitectura

La app sigue una arquitectura **text-to-code**: el LLM no recibe los datos directamente (14.983 registros no caben en el contexto), sino una descripción estructurada del dataset. A partir de la pregunta del usuario, genera código Python con Plotly que se ejecuta localmente para producir la visualización.

```
Pregunta del usuario
       ↓
build_prompt()  →  system prompt con metadatos del dataset
       ↓
get_response()  →  GPT-4.1-mini (OpenAI API)
       ↓
parse_response()  →  JSON { tipo, codigo, interpretacion }
       ↓
execute_chart()  →  exec() en entorno controlado (df, pd, px, go)
       ↓
Gráfico Plotly renderizado en Streamlit
```

## Dataset

~15.000 registros sintéticos con estructura inspirada en la exportación de datos de Spotify. El DataFrame incluye columnas derivadas creadas en `load_data()`: franja horaria, temporada, sesión de escucha, hora local (Europe/Madrid), primera vez que aparece un artista, etc.

## Puesta en marcha

1. Clona el repositorio
2. Crea un entorno virtual y actívalo
3. Instala dependencias: `pip install -r requirements.txt`
4. Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y añade tu API key de OpenAI y una contraseña
5. Ejecuta: `streamlit run app.py`

## Archivos

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación completa: preparación de datos, system prompt, llamada a la API y UI |
| `streaming_history.json` | Dataset del caso (~15.000 registros) |
| `requirements.txt` | Dependencias |
| `.streamlit/secrets.toml.example` | Plantilla para API key y contraseña |
