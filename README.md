# Filtro ATS de CVs

App local que recibe CVs (PDF y DOCX) y una Job Description, y devuelve un ranking de
candidatos con score, skills encontradas y faltantes, años de experiencia y veredicto.
Todo corre en tu máquina: no usa APIs pagas ni envía CVs a ningún servicio.

## Requisitos

- Python 3.10 o superior.
- ~1,5 GB libres para las dependencias (PyTorch CPU) y ~90 MB para el modelo.
- Internet **solo la primera vez**, para descargar el modelo `all-MiniLM-L6-v2` de Hugging Face.

## Instalación

Windows (PowerShell):

```powershell
cd ats_filter
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd ats_filter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` apunta al índice de PyTorch CPU: así en Linux no se bajan varios GB de
> librerías CUDA que la app no usa. Si se quiere usar GPU, quitar esa línea.

## Uso

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. En la barra lateral:

1. Pegar la **Job Description** completa.
2. Escribir las **skills requeridas** separadas por coma (`python, sql, power bi, azure`).
   Se normalizan solas: `PowerBI`, `Power-BI` y `power bi` son lo mismo.
3. Elegir el origen de los CVs: **Subir archivos** (selección múltiple) o **Carpeta local**
   (ruta a una carpeta, con la opción de incluir subcarpetas).
4. Presionar **Analizar CVs**.

El resultado muestra:

- Métricas: mejor score, cuántos pasan, score promedio y archivos no procesados.
- Tabla ordenada por score: Archivo, Email, Score %, Skills Encontradas, Skills Faltantes,
  Años Exp y Veredicto.
- Botones para descargar en **CSV** y en **Excel**. El Excel trae colores por veredicto,
  teléfono, LinkedIn, el desglose del score y dos hojas más: archivos con error y parámetros usados.
- **Ficha del candidato:** el cálculo del score con sus números, el ajuste al puesto, las skills
  requeridas ✓/✗, los datos que se pudieron leer (contacto, experiencia, secciones) y una lectura
  en palabras de por qué salió ese resultado. La guía *Cómo se lee el score* resume la fórmula.
- Lista de archivos que no se pudieron leer y por qué.

La primera carga del modelo tarda unos segundos (y más si tiene que descargarlo); después
queda en memoria y cada CV se procesa en fracciones de segundo en CPU.

### Modo "Afinidad por área"

Para ver a qué tipo de puesto apunta un CV sin tener una Job Description: elegir el modo
**Afinidad por área**, subir un CV y presionar **Comparar áreas**. El CV se compara contra 14
perfiles genéricos (Backend, Frontend, Mobile, Data Engineering, Data Analytics/BI, Data
Science, DevOps/Cloud, QA, UX/UI, Marketing Digital, Social Media, Gestión de Proyectos,
Comercial, Administración y Finanzas) y sale una tabla ordenada por afinidad.

Afinidad = 70% ajuste al perfil del área + 30% skills típicas del área (con 4 encontradas ya
cuenta completa). Es orientativo: para una búsqueda concreta, usar el modo Ranking con la JD
real. Los perfiles están en `AREA_PROFILES` (`ats_parser.py`) y se pueden editar o ampliar.

## Cómo se calcula el score

**Ajuste al puesto (score semántico, 0-100).** El CV y la JD se convierten en embeddings con
`all-MiniLM-L6-v2` y se comparan con cosine similarity (scikit-learn). Como el modelo
trunca a 256 tokens, ambos textos se parten en ventanas de 120 palabras con solapamiento:
para cada fragmento de la JD se busca el fragmento más parecido del CV y se promedian esos
máximos. El coseno resultante se escala linealmente de `[0.20, 0.80]` a `[0, 100]`.

**Cobertura de skills (0-100).** Porcentaje de skills requeridas que aparecen en el CV.

**Score final** = `peso × semántico + (1 − peso) × cobertura`. El peso por defecto es 70% y
se ajusta en *Opciones avanzadas*. Sin skills requeridas, el score es 100% semántico.

**Veredicto:** `PASA` > 65 · `REVISAR` 45-65 · `NO PASA` < 45.

## Qué extrae de cada CV

| Dato | Cómo |
|---|---|
| Email | Regex; también lo toma de los links `mailto:` del PDF/DOCX. |
| Teléfono | Regex; prioriza números con etiqueta (Tel, Cel, WhatsApp) y descarta DNI/CUIT, fechas y rangos de años. |
| LinkedIn | Regex sobre el texto **y** sobre los hipervínculos (muchos CVs dicen solo "LinkedIn" con el link escondido). |
| Skills | Catálogo predefinido de 118 skills (datos, BI, cloud, desarrollo, QA, diseño, marketing) con alias (`t-sql` → sql, `ms fabric` → fabric, `pyspark` → spark + python). Las skills requeridas que no están en el catálogo se buscan literalmente. |
| Años de experiencia | Menciones explícitas ("5 años de experiencia", "5+ years", "más de 5 años", "with 7 years") y suma de rangos de fechas de la sección Experiencia ("Ene 2019 - Actualidad"), sin contar doble los períodos solapados ni la sección Educación. |

En DOCX se leen párrafos, tablas (incluidas las anidadas), encabezados/pies y cuadros de texto.

## Archivos que no se procesan

No frenan el lote: aparecen en *Archivos no procesados* con el motivo.

- **PDF escaneado** (imagen sin capa de texto): hay que pasarlo por OCR o pedir el CV en
  formato digital. Si solo algunas páginas son imagen, el CV se procesa igual y se avisa
  qué páginas se ignoraron.
- PDF protegido con contraseña, archivos corruptos o vacíos.
- `.doc` (Word 97-2003): convertir a `.docx` o PDF.

## Uso por línea de comandos

Para procesar una carpeta sin interfaz (por ejemplo, en una tarea programada):

```bash
python ats_parser.py ./cvs --jd jd.txt --skills "python, sql, power bi" --out ranking.xlsx
```

Opciones: `--weight 0.7` (peso semántico), `--recursive` (subcarpetas), `--out` con `.csv` o `.xlsx`.
`--jd` acepta el texto directamente o la ruta a un `.txt`/`.md`.

También se puede usar como librería:

```python
import ats_parser as ats

text = ats.extract_text("cv.pdf")
parsed = ats.parse_cv(text)                      # email, phone, linkedin, skills
years = ats.extract_experience_years(text)
score = ats.score_cv(text, job_description)       # 0-100
missing = ats.get_missing_skills(parsed.skills, ["python", "sql", "azure"])
```

## Personalización

Todo está en constantes al principio de `ats_parser.py`:

- **Umbrales del veredicto:** `PASS_THRESHOLD`, `REVIEW_THRESHOLD`.
- **Calibración del score:** `SIMILARITY_FLOOR` / `SIMILARITY_CEILING`. Conviene validarla
  con 5-10 CVs cuyo veredicto ya conozcas: si buenos candidatos caen en REVISAR, bajar
  `SIMILARITY_CEILING`; si pasan CVs flojos, subir `SIMILARITY_FLOOR`.
- **Catálogo de skills:** `SKILL_CATALOG` (nombre canónico → alias regex sobre texto en
  minúsculas y sin tildes) e `IMPLIED_SKILLS` (una skill implica otras).
- **Modelo:** variable de entorno `ATS_MODEL_NAME`. `all-MiniLM-L6-v2` está entrenado en
  inglés; funciona razonablemente con CVs en español, pero si casi todos los CVs están en
  español se puede probar `paraphrase-multilingual-MiniLM-L12-v2` (recalibrar luego).

  ```powershell
  $env:ATS_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"; streamlit run app.py
  ```

## Privacidad y modo offline

Los CVs se procesan en memoria y no se guardan en disco (salvo los exports que descargues).
Después de la primera descarga, el modelo se carga desde la caché local de Hugging Face sin
tocar la red. Para garantizar que nunca haya conexión, definir `HF_HUB_OFFLINE=1`.

## Publicar en Streamlit Community Cloud

1. Subir esta carpeta a un repo de GitHub (con `app.py` y `requirements.txt` en la raíz).
2. En [share.streamlit.io](https://share.streamlit.io) → **Create app** → elegir el repo,
   branch `main` y archivo `app.py`. En *Advanced settings*, Python 3.12.
3. La primera instalación tarda unos minutos (PyTorch CPU); el modelo se descarga en el
   primer análisis.

En la nube la opción **Carpeta local** se oculta sola: solo se pueden subir archivos.

**Contraseña de acceso.** Community Cloud permite una sola app privada por cuenta, así que la
app trae su propia pantalla de contraseña. En *Settings → Secrets* de la app cargar:

```toml
APP_PASSWORD = "la-clave-elegida"
```

Sin ese secret la app queda abierta (es lo que pasa en uso local). Localmente también se
puede activar con `.streamlit/secrets.toml` (ignorado por git) o la variable `ATS_APP_PASSWORD`.

> **Privacidad:** publicada, la app ya no es local: los CVs se procesan en servidores de
> Streamlit, en memoria. No se guardan y cada sesión es independiente, así que nadie ve los
> CVs que sube otra persona.

## Estructura

```
ats_filter/
├── app.py            # Interfaz Streamlit (inputs, estado, tabla, métricas, descargas)
├── ats_parser.py     # Lógica: extracción, parsing, scoring, ranking, exportación y CLI
├── requirements.txt
└── README.md
```

## Limitaciones conocidas

- El score mide similitud de contenido, no calidad del candidato: usarlo para priorizar la
  lectura, no para descartar sin revisar. Los casos en REVISAR merecen una mirada humana.
- CVs muy "de diseño" en PDF (columnas, gráficos de nivel de skills) pueden extraerse con
  el texto desordenado; el score semántico tolera eso, los rangos de fechas no siempre.
- Los años de experiencia son una estimación: un rango solo con años ("2018 - 2020") se
  cuenta como la diferencia de años.
- No hace OCR. Para CVs escaneados se puede preprocesar con `ocrmypdf` u otra herramienta.
