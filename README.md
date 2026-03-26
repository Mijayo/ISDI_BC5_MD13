# BC5 — Spotify Analytics Assistant

**Alumno:** Diego Vallejo Mijallo
**Programa:** MDA13 — ISDI
**Aplicación desplegada:** [https://diego-vallejo-mda13-bc5.streamlit.app](https://diego-vallejo-mda13-bc5.streamlit.app)
**Repositorio:** [https://github.com/Mijayo/ISDI_BC5_MD13](https://github.com/Mijayo/ISDI_BC5_MD13)

---

## Resumen

Este proyecto implementa un asistente conversacional de análisis de datos que permite explorar hábitos de escucha de Spotify mediante lenguaje natural. La arquitectura adoptada —denominada *text-to-code*— delega en un modelo de lenguaje de gran escala (LLM) la responsabilidad de traducir preguntas formuladas en lenguaje natural a código Python ejecutable, evitando la transmisión directa del dataset al modelo. El resultado es una interfaz de chat que genera visualizaciones interactivas en respuesta a consultas arbitrarias sobre el historial de reproducción.

---

## 1. Contexto y motivación

El análisis exploratorio de datos requiere habitualmente conocimientos de programación que constituyen una barrera de entrada para usuarios no técnicos. Los LLMs modernos ofrecen la capacidad de actuar como intermediarios entre la intención del usuario y el código necesario para satisfacerla, siempre que dispongan de una descripción precisa del contexto de datos.

El presente caso de negocio explora esta capacidad utilizando como dominio de aplicación el historial de escucha de Spotify: un dataset tabular con múltiples dimensiones temporales, comportamentales y de contenido que admite una gran variedad de preguntas analíticas.

---

## 2. Arquitectura del sistema

La aplicación sigue una arquitectura modular de cinco capas con flujo unidireccional:

```
Pregunta del usuario (lenguaje natural)
         │
         ▼
 build_prompt()
 Construye el system prompt inyectando metadatos
 dinámicos del dataset (fechas, plataformas, sesiones...)
         │
         ▼
 get_response()
 Envía [system prompt + pregunta] al modelo GPT-4.1-mini
 vía la API de OpenAI (temperature = 0.2)
         │
         ▼
 parse_response()
 Deserializa el JSON devuelto por el LLM:
 { "tipo", "codigo", "interpretacion" }
         │
         ▼
 execute_chart()
 Ejecuta el código Python generado mediante exec()
 en un entorno aislado: { df, pd, px, go }
         │
         ▼
 Gráfico Plotly renderizado en Streamlit
```

### 2.1 Principio de separación de responsabilidades

Cada función del sistema tiene una única responsabilidad bien delimitada:

| Función | Responsabilidad |
|---|---|
| `load_data()` | Ingesta, limpieza y enriquecimiento del dataset |
| `build_prompt()` | Parametrización dinámica del system prompt |
| `get_response()` | Comunicación con la API del LLM |
| `parse_response()` | Deserialización y saneamiento de la respuesta |
| `execute_chart()` | Ejecución controlada del código generado |

### 2.2 Justificación del patrón *text-to-code*

El dataset contiene aproximadamente 15.000 registros. Transmitirlos directamente al LLM en cada petición sería técnica y económicamente inviable por dos razones:

1. **Limitación de contexto:** la ventana de contexto de los modelos actuales no puede alojar datasets de este tamaño en producción sin un coste elevado por token.
2. **Latencia y coste:** cada llamada consumiría un volumen de tokens proporcional al dataset, incrementando tanto el tiempo de respuesta como el coste operativo de manera lineal.

En su lugar, el LLM recibe una descripción estructurada del esquema: nombres de columnas, tipos de datos, rangos de valores y estadísticas de alto nivel. Con esta información es capaz de generar código pandas/Plotly correcto sin necesitar acceso a los datos subyacentes.

---

## 3. Dataset y preprocesamiento

### 3.1 Fuente de datos

El dataset (`streaming_history.json`) contiene ~15.000 registros con estructura inspirada en la exportación oficial de datos de Spotify. Cada registro representa una reproducción individual e incluye metadatos de canción, artista, álbum, plataforma, duración reproducida y motivos de inicio y fin de la pista.

### 3.2 Pipeline de transformación (`load_data`)

La función `load_data()` se ejecuta una única vez al iniciar la aplicación (memoizada con `@st.cache_data`) y realiza las siguientes transformaciones:

| Paso | Transformación | Justificación |
|------|---------------|---------------|
| 1 | `ts` → `datetime` UTC | Habilita operaciones temporales sobre pandas |
| 2 | Columnas derivadas: `hora`, `dia_semana`, `mes`, `año` | Granularidad temporal para análisis |
| 3 | `ms_played` → `minutos_escuchados` | Unidad más legible e intuitiva |
| 4 | Renombrado de columnas largas (`master_metadata_*`) | Simplifica el código generado por el LLM |
| 5 | Filtrado de registros sin canción (podcasts, anuncios) | Homogeneidad del dataset |
| 6 | `skipped` → booleano limpio | Normalización de valores nulos |
| 7 | `escucha_completa` ← `reason_end == "trackdone"` | Variable derivada binaria |
| 8 | `franja` horaria (5 tramos) | Segmentación temporal del día |
| 9 | `es_finde` (booleano) | Distinción laborable/festivo |
| 10 | `plataforma` simplificada (`ios`, `android`, `desktop`, `web`) | Normalización de texto libre |
| 11 | `hora_local` (Europe/Madrid) | Corrección de zona horaria para análisis temporales |
| 12 | `temporada` del año | Variable categórica estacional |
| 13-14 | `sesion_id` + `duracion_sesion_min` | Segmentación de sesiones (pausa > 30 min = nueva sesión) |
| 15 | `primera_vez_artista` | Identificación de descubrimientos musicales |

El resultado es un DataFrame de **24 columnas** con semántica clara y consistente, diseñado específicamente para ser utilizado por código generado automáticamente.

---

## 4. Diseño del system prompt

### 4.1 Estructura y contenido

El system prompt es el componente central de la arquitectura *text-to-code*. Su función es proporcionar al LLM el contexto necesario para generar código correcto sin acceso a los datos. Contiene:

- **Metadatos dinámicos del dataset:** rango de fechas, plataformas presentes, valores posibles de variables categóricas, número de sesiones y artistas únicos. Estos valores se inyectan en tiempo de ejecución mediante `build_prompt()`.
- **Esquema completo del DataFrame:** tabla con nombre de columna, tipo de dato y descripción semántica de las 24 columnas disponibles.
- **Reglas de generación de código:** restricciones sobre las bibliotecas utilizables (`df`, `pd`, `px`, `go`), convenciones de nomenclatura y patrones pandas recomendados para operaciones frecuentes.
- **Especificación del formato de respuesta:** el modelo debe devolver exclusivamente un objeto JSON con las claves `tipo`, `codigo` e `interpretacion`, sin texto adicional ni bloques markdown.
- **Manejo de preguntas fuera de alcance:** instrucción explícita para responder con `tipo: "fuera_de_alcance"` ante preguntas no relacionadas con el dataset.

### 4.2 Importancia de los metadatos dinámicos

La parametrización dinámica del prompt es crítica para la corrección del código generado. Dos ejemplos ilustrativos:

- **Sin `fecha_min`/`fecha_max`:** una pregunta como *"¿cuánto he escuchado este año?"* llevaría al modelo a filtrar por un año incorrecto, pues desconoce el rango temporal del dataset.
- **Sin la tabla de columnas:** cualquier pregunta sobre artistas o canciones provocaría un error de ejecución, ya que el modelo inventaría nombres de columna (`artist_name`, `song_title`) distintos de los reales (`artista`, `track`).

### 4.3 Temperatura y determinismo

La llamada a la API utiliza `temperature=0.2`, valor que reduce la variabilidad en la generación de código. Un valor elevado produciría código más creativo pero menos fiable sintácticamente; un valor de 0 maximizaría el determinismo a expensas de la capacidad de adaptación a preguntas inusuales.

---

## 5. Flujo de ejecución completo

El siguiente diagrama describe el flujo paso a paso desde que el usuario formula una pregunta hasta que visualiza el resultado:

```
1. El usuario escribe una pregunta en el widget st.chat_input()

2. build_prompt() inyecta en SYSTEM_PROMPT los valores reales del
   dataset: fechas, plataformas, sesiones, artistas únicos.

3. get_response() construye la petición con dos mensajes:
   - role: "system" → system prompt parametrizado
   - role: "user"   → pregunta del usuario
   y la envía a GPT-4.1-mini con temperature=0.2.

4. El LLM devuelve un string con un objeto JSON:
   {
     "tipo": "grafico" | "fuera_de_alcance",
     "codigo": "fig = px.bar(...)",
     "interpretacion": "El artista más escuchado es..."
   }

5. parse_response() sanitiza el string (elimina posibles bloques
   de markdown) y lo deserializa a diccionario Python con json.loads().

6a. Si tipo == "fuera_de_alcance":
    → Se muestra únicamente el texto de interpretación.

6b. Si tipo == "grafico":
    → execute_chart() invoca exec(codigo, {}, local_vars)
      donde local_vars = {df, pd, px, go}.
    → Se extrae la variable `fig` del entorno de ejecución.
    → Streamlit renderiza: gráfico Plotly + interpretación + código fuente.
```

---

## 6. Seguridad y limitaciones

### 6.1 Entorno de ejecución controlado

El código generado por el LLM se ejecuta mediante `exec()` en un espacio de nombres aislado que expone únicamente cuatro objetos: `df`, `pd`, `px` y `go`. El system prompt instruye explícitamente al modelo para no utilizar sentencias `import` ni funciones externas, reduciendo la superficie de ataque de la ejecución dinámica.

### 6.2 Limitaciones conocidas

- **Preguntas multipart complejas:** el sistema genera un único gráfico por pregunta; consultas que requieren múltiples visualizaciones deben descomponerse manualmente.
- **Historial de conversación:** la implementación actual es *stateless* —cada pregunta se procesa de forma independiente, sin memoria de preguntas anteriores.
- **Calidad del código generado:** en casos extremos el LLM puede generar código sintácticamente correcto pero semánticamente incorrecto (e.g., agrupaciones erróneas). El sistema muestra el código generado en la interfaz para facilitar la verificación por parte del usuario.

---

## 7. Puesta en marcha local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Mijayo/ISDI_BC5_MD13.git
cd ISDI_BC5_MD13

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar secrets.toml y añadir OPENAI_API_KEY y PASSWORD

# 5. Ejecutar la aplicación
streamlit run app.py
```

---

## 8. Dependencias

| Paquete | Versión | Función |
|---|---|---|
| `streamlit` | 1.55.0 | Framework de interfaz web |
| `openai` | 2.28.0 | Cliente para la API de OpenAI |
| `pandas` | 2.2.3 | Manipulación de datos tabulares |
| `plotly` | 6.6.0 | Generación de gráficos interactivos |

---

## 9. Estructura del repositorio

```
BC5/
├── app.py                          # Aplicación principal (pipeline completo + UI)
├── streaming_history.json          # Dataset (~15.000 registros)
├── requirements.txt                # Dependencias fijadas
├── .streamlit/
│   └── secrets.toml.example        # Plantilla de configuración
└── README.md                       # Este documento
```

---

## 10. Reflexión técnica

### 10.1 Arquitectura text-to-code

La aplicación implementa una arquitectura en la que el LLM actúa como compilador de intención: recibe una descripción del contexto de datos y una consulta en lenguaje natural, y devuelve código ejecutable. Este enfoque desacopla completamente el modelo del dato, eliminando la necesidad de transmitir el dataset en cada petición y permitiendo que la aplicación opere de forma eficiente con datasets de cualquier tamaño, siempre que el esquema quepa en el contexto del modelo.

### 10.2 El system prompt como contrato de interfaz

El system prompt actúa como un contrato formal entre la aplicación y el LLM: especifica qué variables están disponibles, qué nombres tienen, qué valores admiten y en qué formato debe estructurarse la respuesta. La correcta especificación de este contrato es condición necesaria para la fiabilidad del sistema. La parametrización dinámica mediante `build_prompt()` garantiza que el contrato refleje siempre el estado real del dataset.

### 10.3 Flujo de ejecución

El flujo completo —desde la pregunta hasta el gráfico— traversa cinco transformaciones bien definidas: parametrización del contexto, llamada al modelo, deserialización, ejecución controlada y renderizado. La separación de estas etapas en funciones independientes facilita el diagnóstico de fallos: cualquier excepción puede atribuirse inequívocamente a una etapa específica del pipeline.
