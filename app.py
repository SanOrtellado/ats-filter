"""Interfaz Streamlit del filtro ATS.

Ejecutar con::

    streamlit run app.py

Toda la lógica de extracción, parsing y scoring vive en ``ats_parser.py``;
este módulo solo maneja inputs, estado de la sesión y presentación.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import ats_parser as ats

logger = logging.getLogger("ats_filter.app")

st.set_page_config(page_title="Filtro ATS", page_icon="📄", layout="wide")

VERDICT_STYLES: dict[str, str] = {
    ats.VERDICT_PASS: "background-color: #C6EFCE; color: #006100; font-weight: 600",
    ats.VERDICT_REVIEW: "background-color: #FFEB9C; color: #9C5700; font-weight: 600",
    ats.VERDICT_FAIL: "background-color: #FFC7CE; color: #9C0006; font-weight: 600",
}
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIN_JD_WORDS = 30

# Streamlit Community Cloud monta el repo en /mount/src. Ahí "Carpeta local" sería el disco
# del servidor, no el del usuario: se oculta (también se puede apagar con la variable de entorno).
FOLDER_MODE_ENABLED = not Path("/mount/src").exists() and os.getenv("ATS_DISABLE_FOLDER_MODE") != "1"


# ---------------------------------------------------------------------------
# Recursos cacheados
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando modelo de embeddings (la primera vez lo descarga, ~90 MB)…")
def load_model(model_name: str):
    """Modelo compartido entre sesiones y reruns: se carga una sola vez por proceso."""
    return ats.get_model(model_name)


@st.cache_data(show_spinner=False, max_entries=2000)
def extract_cached(data: bytes, file_name: str) -> ats.Extraction:
    """Extracción cacheada por contenido: re-analizar con otra JD no vuelve a leer los archivos."""
    return ats.extract_from_bytes(data, file_name)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def unique_name(name: str, taken: set[str]) -> str:
    """Evita que dos archivos con el mismo nombre se pisen en la tabla."""
    candidate, counter = name, 2
    while candidate in taken:
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate, counter = f"{stem} ({counter}){suffix}", counter + 1
    taken.add(candidate)
    return candidate


def files_from_upload(uploaded: list) -> list[tuple[str, bytes]]:
    taken: set[str] = set()
    return [(unique_name(f.name, taken), f.getvalue()) for f in uploaded]


def files_from_folder(folder: str, recursive: bool) -> list[tuple[str, bytes]]:
    root = Path(folder.strip().strip('"')).expanduser()
    return [(str(path.relative_to(root)), path.read_bytes()) for path in ats.iter_cv_files(root, recursive)]


def file_fingerprint(source: str, uploaded: list, folder: str, recursive: bool) -> list[tuple[str, int]]:
    """Nombre y tamaño de cada archivo, sin leer el contenido (para detectar cambios)."""
    if source == "upload":
        return [(f.name, f.size) for f in uploaded or []]
    try:
        root = Path(folder.strip().strip('"')).expanduser()
        return [(str(p), p.stat().st_size) for p in ats.iter_cv_files(root, recursive)]
    except (OSError, NotADirectoryError):
        return []


def inputs_signature(*parts: object) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------


def style_verdict(value: str) -> str:
    return VERDICT_STYLES.get(value, "")


def render_metrics(ranking: pd.DataFrame, failures: pd.DataFrame) -> None:
    passed = int((ranking["Veredicto"] == ats.VERDICT_PASS).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mejor score", f"{ranking['Score %'].max():.1f}%")
    col2.metric(f"Pasan (> {ats.PASS_THRESHOLD:.0f}%)", f"{passed} de {len(ranking)}")
    col3.metric("Score promedio", f"{ranking['Score %'].mean():.1f}%")
    col4.metric("Archivos no procesados", len(failures))


def render_ranking(ranking: pd.DataFrame) -> None:
    st.dataframe(
        ranking[ats.DISPLAY_COLUMNS].style.map(style_verdict, subset=["Veredicto"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Archivo": st.column_config.TextColumn("Archivo", width="medium"),
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Score %": st.column_config.ProgressColumn(
                "Score %", min_value=0, max_value=100, format="%.1f%%", width="small"
            ),
            "Skills Encontradas": st.column_config.TextColumn("Skills Encontradas", width="large"),
            "Skills Faltantes": st.column_config.TextColumn("Skills Faltantes", width="medium"),
            "Años Exp": st.column_config.NumberColumn("Años Exp", format="%.1f", width="small"),
            "Veredicto": st.column_config.TextColumn("Veredicto", width="small"),
        },
    )


def render_detail(ranking: pd.DataFrame) -> None:
    with st.expander("🔎 Detalle por candidato"):
        name = st.selectbox("Candidato", ranking["Archivo"].tolist(), label_visibility="collapsed")
        row = ranking.loc[ranking["Archivo"] == name].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Score final", f"{row['Score %']:.1f}%")
        col2.metric("Score semántico", f"{row['Score Semántico %']:.1f}%")
        coverage = row["Cobertura Skills %"]
        col3.metric("Cobertura de skills", "—" if pd.isna(coverage) else f"{coverage:.0f}%")
        st.markdown(
            f"**Email:** {row['Email'] or '—'}  \n"
            f"**Teléfono:** {row['Teléfono'] or '—'}  \n"
            f"**LinkedIn:** {row['LinkedIn'] or '—'}  \n"
            f"**Años de experiencia:** {'—' if pd.isna(row['Años Exp']) else row['Años Exp']}"
        )
        if row["Advertencias"]:
            st.warning(row["Advertencias"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuración")
    job_description = st.text_area(
        "Job Description",
        height=260,
        placeholder="Pegar acá la descripción completa de la búsqueda: responsabilidades, requisitos, stack…",
    )
    skills_raw = st.text_input(
        "Skills requeridas (separadas por coma)",
        placeholder="python, sql, power bi, azure, etl",
    )

    source = "upload"
    if FOLDER_MODE_ENABLED:
        source_label = st.radio("Origen de los CVs", ["Subir archivos", "Carpeta local"], horizontal=True)
        source = "upload" if source_label == "Subir archivos" else "folder"
    uploaded: list = []
    folder, recursive = "", False
    if source == "upload":
        uploaded = st.file_uploader("CVs (PDF o DOCX)", type=["pdf", "docx"], accept_multiple_files=True) or []
    else:
        folder = st.text_input("Ruta de la carpeta", placeholder=r"C:\CVs\analista_datos")
        recursive = st.checkbox("Incluir subcarpetas")

    with st.expander("Opciones avanzadas"):
        weight_pct = st.slider(
            "Peso del score semántico (%)",
            min_value=0,
            max_value=100,
            value=int(ats.DEFAULT_SEMANTIC_WEIGHT * 100),
            step=5,
            help="Score final = peso × similitud semántica + (100 − peso) × % de skills requeridas encontradas. "
            "Sin skills requeridas, el score es 100% semántico.",
        )

    analyze_clicked = st.button("Analizar CVs", type="primary", width="stretch")

st.title("📄 Filtro ATS de CVs")
st.caption(
    f"Ranking de candidatos contra la Job Description · modelo `{ats.DEFAULT_MODEL_NAME}` · "
    "sin APIs externas: los CVs se procesan en memoria y no se guardan"
)

required_skills = ats.parse_skill_list(skills_raw)
signature = inputs_signature(
    job_description.strip(),
    required_skills,
    weight_pct,
    source,
    folder,
    recursive,
    file_fingerprint(source, uploaded, folder, recursive),
)

if analyze_clicked:
    problems: list[str] = []
    if not job_description.strip():
        problems.append("Falta la Job Description.")
    files: list[tuple[str, bytes]] = []
    if source == "upload":
        files = files_from_upload(uploaded)
    elif not folder.strip():
        problems.append("Falta la ruta de la carpeta.")
    else:
        try:
            files = files_from_folder(folder, recursive)
        except (OSError, NotADirectoryError) as exc:
            problems.append(f"No se pudo leer la carpeta: {exc}")
    if not files and not problems:
        problems.append("No hay CVs en PDF o DOCX para analizar.")

    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()
    if len(job_description.split()) < MIN_JD_WORDS:
        st.warning(f"La Job Description tiene menos de {MIN_JD_WORDS} palabras: el score semántico será menos confiable.")

    try:
        model = load_model(ats.DEFAULT_MODEL_NAME)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    progress = st.progress(0.0, text="Iniciando…")

    def report_progress(done: int, total: int, current: str) -> None:
        label = f"Analizando {done + 1}/{total}: {current}" if current else "Listo"
        progress.progress(done / total if total else 1.0, text=label)

    with st.spinner(f"Analizando {len(files)} CV(s)…"):
        results, failures = ats.analyze_files(
            files,
            job_description,
            required_skills,
            model=model,
            semantic_weight=weight_pct / 100,
            extract=extract_cached,
            on_progress=report_progress,
        )
    progress.empty()

    ranking = ats.results_to_dataframe(results)
    failures_df = ats.failures_to_dataframe(failures)
    parameters = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Modelo": ats.DEFAULT_MODEL_NAME,
        "Skills requeridas": ", ".join(required_skills) or "(ninguna)",
        "Peso semántico %": weight_pct,
        "Umbral PASA": f"> {ats.PASS_THRESHOLD:.0f}",
        "Umbral REVISAR": f"{ats.REVIEW_THRESHOLD:.0f} - {ats.PASS_THRESHOLD:.0f}",
        "CVs analizados": len(results),
        "Archivos no procesados": len(failures),
        "Job Description": job_description.strip(),
    }
    st.session_state["analysis"] = {
        "signature": signature,
        "ranking": ranking,
        "failures": failures_df,
        "csv": ats.to_csv_bytes(ranking),
        "xlsx": ats.to_excel_bytes(ranking, failures_df, parameters),
        "stamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }

analysis = st.session_state.get("analysis")
if not analysis:
    st.info(
        "**Cómo usarlo:** pegar la Job Description en la barra lateral, listar las skills requeridas, "
        f"subir los CVs{' (o indicar una carpeta local)' if FOLDER_MODE_ENABLED else ''} "
        "y presionar **Analizar CVs**.\n\n"
        f"Veredicto: **PASA** > {ats.PASS_THRESHOLD:.0f}% · **REVISAR** {ats.REVIEW_THRESHOLD:.0f}–"
        f"{ats.PASS_THRESHOLD:.0f}% · **NO PASA** < {ats.REVIEW_THRESHOLD:.0f}%."
    )
    st.stop()

if analysis["signature"] != signature:
    st.info("Los parámetros o los archivos cambiaron desde el último análisis: presionar **Analizar CVs** para actualizar.")

ranking: pd.DataFrame = analysis["ranking"]
failures_df: pd.DataFrame = analysis["failures"]

if ranking.empty:
    st.error("Ningún CV pudo procesarse. Revisar el detalle de los archivos abajo.")
else:
    render_metrics(ranking, failures_df)
    render_ranking(ranking)

    col_csv, col_xlsx, _ = st.columns([1, 1, 3])
    col_csv.download_button(
        "⬇️ Descargar CSV",
        data=analysis["csv"],
        file_name=f"ranking_ats_{analysis['stamp']}.csv",
        mime="text/csv",
        width="stretch",
    )
    col_xlsx.download_button(
        "⬇️ Descargar Excel",
        data=analysis["xlsx"],
        file_name=f"ranking_ats_{analysis['stamp']}.xlsx",
        mime=EXCEL_MIME,
        width="stretch",
    )

    with_warnings = int((ranking["Advertencias"] != "").sum())
    if with_warnings:
        st.warning(f"{with_warnings} CV(s) con advertencias (por ejemplo, páginas escaneadas). Ver el detalle por candidato.")
    render_detail(ranking)

if not failures_df.empty:
    with st.expander(f"⚠️ Archivos no procesados ({len(failures_df)})", expanded=ranking.empty):
        st.dataframe(failures_df, hide_index=True, width="stretch")
