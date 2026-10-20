"""Interfaz Streamlit del filtro ATS.

Ejecutar con::

    streamlit run app.py

Toda la lógica de extracción, parsing y scoring vive en ``ats_parser.py``;
este módulo solo maneja inputs, estado de la sesión y presentación.
"""

from __future__ import annotations

import hashlib
import hmac
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
VERDICT_BADGE_COLORS: dict[str, str] = {
    ats.VERDICT_PASS: "green",
    ats.VERDICT_REVIEW: "orange",
    ats.VERDICT_FAIL: "red",
}
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

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


def render_ranking(ranking: pd.DataFrame, weight_pct: int) -> None:
    st.dataframe(
        ranking[ats.DISPLAY_COLUMNS].style.map(style_verdict, subset=["Veredicto"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Archivo": st.column_config.TextColumn("Archivo", width="medium"),
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Score %": st.column_config.ProgressColumn(
                "Score %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="small",
                help=f"{weight_pct}% ajuste al puesto + {100 - weight_pct}% skills requeridas. "
                "El desglose está en la ficha del candidato.",
            ),
            "Skills Encontradas": st.column_config.TextColumn(
                "Skills Encontradas", width="large", help="Primero las requeridas que aparecen; después otras detectadas."
            ),
            "Skills Faltantes": st.column_config.TextColumn(
                "Skills Faltantes", width="medium", help="Skills requeridas que no aparecen en el CV."
            ),
            "Años Exp": st.column_config.NumberColumn(
                "Años Exp", format="%.1f", width="small", help="Estimado por menciones explícitas o rangos de fechas."
            ),
            "Veredicto": st.column_config.TextColumn(
                "Veredicto",
                width="small",
                help=f"PASA > {ats.PASS_THRESHOLD:.0f} · REVISAR {ats.REVIEW_THRESHOLD:.0f}–{ats.PASS_THRESHOLD:.0f} "
                f"· NO PASA < {ats.REVIEW_THRESHOLD:.0f}",
            ),
        },
    )


def render_score_guide(weight_pct: int) -> None:
    with st.expander("ℹ️ Cómo se lee el score"):
        st.markdown(
            f"**Score final** = {weight_pct}% *ajuste al puesto* + {100 - weight_pct}% *skills requeridas* "
            "(sin skills requeridas, es 100% ajuste al puesto).\n\n"
            "- **🎯 Ajuste al puesto:** cuánto se parece el contenido del CV a la Job Description. Lo calcula un "
            "modelo de lenguaje, así que reconoce sinónimos y contexto, no solo palabras exactas.\n"
            "- **🧩 Skills requeridas:** porcentaje de las skills cargadas que aparecen en el CV.\n"
            "- **📄 Datos legibles:** qué datos pudo leer el filtro (contacto, experiencia, secciones). No suma al "
            "score, pero si faltan, el CV puede estar mal estructurado o ser difícil de leer para un ATS.\n\n"
            f"**Veredicto:** :green-badge[PASA] > {ats.PASS_THRESHOLD:.0f} · :orange-badge[REVISAR] "
            f"{ats.REVIEW_THRESHOLD:.0f}–{ats.PASS_THRESHOLD:.0f} · :red-badge[NO PASA] < {ats.REVIEW_THRESHOLD:.0f}\n\n"
            "El score sirve para **ordenar y priorizar la lectura**, no es una nota absoluta del CV: un score bajo "
            "no significa que el candidato no sirva, y los REVISAR merecen una mirada humana."
        )


def badge(text: str, color: str) -> str:
    """Badge de Markdown de Streamlit; quita corchetes que romperían la sintaxis."""
    return f":{color}-badge[{text.replace('[', '(').replace(']', ')')}]"


def render_candidate_card(
    result: ats.CandidateResult, required: list[str], jd_words: int, weight_pct: int
) -> None:
    with st.container(border=True):
        score_col, verdict_col = st.columns([1, 3])
        score_col.metric("Score final", f"{result.score:.1f}%")
        with verdict_col:
            st.markdown(
                f"{badge(result.verdict, VERDICT_BADGE_COLORS[result.verdict])} "
                f"{ats.VERDICT_MESSAGES[result.verdict]}"
            )
            if result.skill_coverage is None:
                st.caption("Sin skills requeridas: el score final es el ajuste al puesto.")
            else:
                st.caption(
                    f"Cálculo: {weight_pct}% × {result.semantic_score:.1f} (ajuste al puesto) + "
                    f"{100 - weight_pct}% × {result.skill_coverage:.0f} (skills requeridas) = {result.score:.1f}"
                )

        fit_col, skills_col, data_col = st.columns(3)
        with fit_col:
            st.metric(
                "🎯 Ajuste al puesto",
                f"{result.semantic_score:.1f}%",
                help="Similitud de contenido entre el CV y la Job Description (modelo de lenguaje local).",
            )
            st.caption(f"Nivel {ats.fit_label(result.semantic_score).lower()} · pesa {weight_pct}% del score")
        with skills_col:
            coverage = result.skill_coverage
            found_required = [skill for skill in required if skill not in result.missing_skills]
            st.metric(
                "🧩 Skills requeridas",
                "—" if coverage is None else f"{len(found_required)} de {len(required)}",
                help="Skills cargadas en la barra lateral que aparecen en el CV.",
            )
            if required:
                st.markdown(
                    " ".join(
                        [badge(f"✓ {s}", "green") for s in found_required]
                        + [badge(f"✗ {s}", "red") for s in result.missing_skills]
                    )
                )
            others = [skill for skill in result.found_skills if skill not in required]
            if others:
                st.caption("Otras detectadas: " + ", ".join(others))
        with data_col:
            checks = ats.data_checks(result)
            st.metric(
                "📄 Datos legibles",
                f"{sum(ok for _, ok in checks)} de {len(checks)}",
                help="Qué pudo leer el filtro del CV. No suma al score.",
            )
            st.markdown("  \n".join(f"{'✅' if ok else '❌'} {label}" for label, ok in checks))

        st.markdown("**Por qué este resultado**")
        st.markdown("\n".join(f"- {note}" for note in ats.explain_result(result, required, jd_words)))

        years = "—" if result.experience_years is None else f"{result.experience_years:.1f} (estimado)"
        st.caption(
            f"Email: {result.email or '—'} · Teléfono: {result.phone or '—'} · "
            f"LinkedIn: {result.linkedin or '—'} · Años de experiencia: {years}"
        )


# ---------------------------------------------------------------------------
# Acceso
# ---------------------------------------------------------------------------


def configured_password() -> str | None:
    """Contraseña de acceso: secret ``APP_PASSWORD`` (Streamlit Cloud) o variable ``ATS_APP_PASSWORD``.

    Sin ninguna de las dos, la app queda abierta (uso local).
    """
    try:
        password = st.secrets.get("APP_PASSWORD")
    except Exception:  # sin secrets.toml, Streamlit lanza al leer st.secrets
        password = None
    return str(password) if password else os.getenv("ATS_APP_PASSWORD") or None


def require_password() -> None:
    """Muestra el ingreso y corta la ejecución hasta que se ingrese la contraseña correcta."""
    expected = configured_password()
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("📄 Filtro ATS de CVs")
    with st.form("login"):
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar", type="primary")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Contraseña incorrecta.")
    st.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

require_password()

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
            help="Score final = peso × ajuste al puesto + (100 − peso) × % de skills requeridas encontradas. "
            "Sin skills requeridas, el score es 100% ajuste al puesto.",
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
    if len(job_description.split()) < ats.MIN_JD_WORDS:
        st.warning(
            f"La Job Description tiene menos de {ats.MIN_JD_WORDS} palabras: el ajuste al puesto será menos confiable."
        )

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
        "results": {result.file_name: result for result in results},
        "required": required_skills,
        "jd_words": len(job_description.split()),
        "weight_pct": weight_pct,
    }

analysis = st.session_state.get("analysis")
if analysis and "results" not in analysis:  # análisis guardado por una versión anterior de la app
    analysis = None
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
    render_score_guide(analysis["weight_pct"])
    render_ranking(ranking, analysis["weight_pct"])

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
        st.warning(f"{with_warnings} CV(s) con advertencias (por ejemplo, páginas escaneadas). Ver la ficha del candidato.")

    st.subheader("🔎 Ficha del candidato")
    name = st.selectbox("Candidato", ranking["Archivo"].tolist(), label_visibility="collapsed")
    render_candidate_card(analysis["results"][name], analysis["required"], analysis["jd_words"], analysis["weight_pct"])

if not failures_df.empty:
    with st.expander(f"⚠️ Archivos no procesados ({len(failures_df)})", expanded=ranking.empty):
        st.dataframe(failures_df, hide_index=True, width="stretch")
