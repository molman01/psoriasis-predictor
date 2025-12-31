{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import io\
\
from sklearn.model_selection import train_test_split\
from sklearn.pipeline import Pipeline\
from sklearn.compose import ColumnTransformer\
from sklearn.impute import SimpleImputer\
from sklearn.preprocessing import StandardScaler\
from sklearn.metrics import (\
    roc_auc_score, accuracy_score, confusion_matrix, classification_report,\
    f1_score\
)\
from sklearn.linear_model import LogisticRegression\
\
st.set_page_config(page_title="Psoriasis \'96 Predictor PASI75/PASI90", layout="centered")\
\
st.title("Calculadora predictiva de respuesta (PASI75 / PASI90)")\
st.caption("Sube tu Excel, elige PASI75 o PASI90, la app entrena autom\'e1ticamente y podr\'e1s predecir pacientes (individual o en lote).")\
st.warning("\uc0\u9888 \u65039  Uso orientativo / apoyo. No sustituye el juicio cl\'ednico.")\
\
FEATURES = ["HOMBRE", "MUJER", "EDAD", "IMC", "PASI INICIAL ADA", "ARTRITIS", "NPREV"]\
\
TARGETS = \{\
    "PASI75": \{"ok": "PASIOK75", "ko": "PASIKO75"\},\
    "PASI90": \{"ok": "PASIOK90", "ko": "PASIKO90"\},\
\}\
\
WARN_MIN_PER_CLASS = 15\
BLOCK_MIN_PER_CLASS = 5\
\
\
def clean_binary_series(s: pd.Series) -> pd.Series:\
    s2 = pd.to_numeric(s, errors="coerce")\
    if s2.isna().mean() > 0.5:\
        mapped = s.astype(str).str.strip().str.lower().map(\{\
            "1": 1, "0": 0, "true": 1, "false": 0, "s\'ed": 1, "si": 1, "no": 0\
        \})\
        s2 = pd.to_numeric(mapped, errors="coerce")\
    return s2\
\
\
def class_summary(df: pd.DataFrame, target_ok: str, target_ko: str) -> dict:\
    out = \{\
        "has_ok": target_ok in df.columns,\
        "has_ko": target_ko in df.columns,\
        "n_labeled": 0,\
        "n_ok": 0,\
        "n_ko_from_ok": 0,\
        "pct_ok": np.nan,\
        "pct_ko": np.nan,\
        "incons_both_1": None,\
        "incons_both_0": None,\
    \}\
\
    if target_ok not in df.columns:\
        return out\
\
    ok = clean_binary_series(df[target_ok])\
    labeled = ok.dropna()\
    out["n_labeled"] = int(labeled.shape[0])\
    out["n_ok"] = int((labeled == 1).sum())\
    out["n_ko_from_ok"] = int((labeled == 0).sum())\
\
    if out["n_labeled"] > 0:\
        out["pct_ok"] = out["n_ok"] / out["n_labeled"]\
        out["pct_ko"] = out["n_ko_from_ok"] / out["n_labeled"]\
\
    if target_ko in df.columns:\
        ko = clean_binary_series(df[target_ko])\
        both_defined = (~ok.isna()) & (~ko.isna())\
        out["incons_both_1"] = int(((ok == 1) & (ko == 1) & both_defined).sum())\
        out["incons_both_0"] = int(((ok == 0) & (ko == 0) & both_defined).sum())\
\
    return out\
\
\
def warn_or_block_for_small_classes(n_ok: int, n_ko: int, objective: str) -> tuple[bool, list[str]]:\
    msgs = []\
    block = False\
\
    if n_ok == 0 or n_ko == 0:\
        block = True\
        msgs.append(\
            f"\uc0\u10060  \{objective\}: una de las clases est\'e1 vac\'eda (OK=\{n_ok\}, NO=\{n_ko\}). "\
            "No se puede entrenar un modelo binario."\
        )\
        return block, msgs\
\
    if min(n_ok, n_ko) < BLOCK_MIN_PER_CLASS:\
        block = True\
        msgs.append(\
            f"\uc0\u10060  \{objective\}: hay muy pocos casos en una clase (OK=\{n_ok\}, NO=\{n_ko\}). "\
            f"Por seguridad, el entrenamiento se bloquea si una clase tiene < \{BLOCK_MIN_PER_CLASS\}."\
        )\
        return block, msgs\
\
    if min(n_ok, n_ko) < WARN_MIN_PER_CLASS:\
        msgs.append(\
            f"\uc0\u9888 \u65039  \{objective\}: posible inestabilidad por clases peque\'f1as (OK=\{n_ok\}, NO=\{n_ko\}). "\
            f"Recomendaci\'f3n: idealmente \uc0\u8805  \{WARN_MIN_PER_CLASS\} casos por clase."\
        )\
\
    total = n_ok + n_ko\
    frac = n_ok / total if total > 0 else np.nan\
    if not np.isnan(frac) and (frac < 0.15 or frac > 0.85):\
        msgs.append(\
            f"\uc0\u9888 \u65039  \{objective\}: desbalance fuerte (proporci\'f3n OK=\{frac*100:.1f\}%). "\
            "Las m\'e9tricas y el umbral pueden ser sensibles."\
        )\
\
    return block, msgs\
\
\
def load_and_validate_training(df: pd.DataFrame, target_ok: str, target_ko: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:\
    issues = []\
    missing = [c for c in FEATURES + [target_ok] if c not in df.columns]\
    if missing:\
        issues.append(f"Faltan columnas necesarias: \{missing\}")\
\
    has_ko = target_ko in df.columns\
    if issues:\
        return df, pd.Series(dtype=float), issues\
\
    X = df[FEATURES].copy()\
    y = clean_binary_series(df[target_ok].copy())\
\
    for b in ["HOMBRE", "MUJER", "ARTRITIS"]:\
        X[b] = clean_binary_series(X[b])\
\
    inconsistent_sex = ((X["HOMBRE"] == 1) & (X["MUJER"] == 1)).sum()\
    missing_sex = ((X["HOMBRE"].fillna(0) == 0) & (X["MUJER"].fillna(0) == 0)).sum()\
    if inconsistent_sex > 0:\
        issues.append(f"Hay \{inconsistent_sex\} filas con HOMBRE=1 y MUJER=1 (inconsistente).")\
    if missing_sex > 0:\
        issues.append(f"Hay \{missing_sex\} filas con HOMBRE=0 y MUJER=0 (sexo no indicado).")\
\
    if has_ko:\
        ko = clean_binary_series(df[target_ko])\
        both_1 = ((y == 1) & (ko == 1)).sum()\
        both_0 = ((y == 0) & (ko == 0)).sum()\
        if both_1 > 0:\
            issues.append(f"Hay \{both_1\} filas con \{target_ok\}=1 y \{target_ko\}=1 (inconsistente).")\
        if both_0 > 0:\
            issues.append(f"Hay \{both_0\} filas con \{target_ok\}=0 y \{target_ko\}=0 (inconsistente).")\
\
    y_valid = y.dropna()\
    uniq = set(y_valid.unique().tolist())\
    if not uniq.issubset(\{0, 1\}):\
        issues.append(f"\{target_ok\} tiene valores fuera de 0/1: \{sorted(list(uniq))\}")\
\
    return X, y, issues\
\
\
def compute_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:\
    thresholds = np.unique(np.clip(y_prob, 0, 1))\
    if thresholds.size < 5:\
        thresholds = np.linspace(0.05, 0.95, 19)\
\
    best_youden = \{"thr": 0.5, "youden": -np.inf, "sens": np.nan, "spec": np.nan\}\
    best_f1 = \{"thr": 0.5, "f1": -np.inf\}\
\
    for t in thresholds:\
        pred = (y_prob >= t).astype(int)\
\
        tp = int(((pred == 1) & (y_true == 1)).sum())\
        tn = int(((pred == 0) & (y_true == 0)).sum())\
        fp = int(((pred == 1) & (y_true == 0)).sum())\
        fn = int(((pred == 0) & (y_true == 1)).sum())\
\
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan\
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan\
        youden = (sens + spec - 1) if (not np.isnan(sens) and not np.isnan(spec)) else -np.inf\
\
        if youden > best_youden["youden"]:\
            best_youden = \{"thr": float(t), "youden": float(youden), "sens": float(sens), "spec": float(spec)\}\
\
        f1 = f1_score(y_true, pred, zero_division=0)\
        if f1 > best_f1["f1"]:\
            best_f1 = \{"thr": float(t), "f1": float(f1)\}\
\
    return \{"youden": best_youden, "f1": best_f1\}\
\
\
@st.cache_data\
def train_model_from_df(df: pd.DataFrame, target_ok: str, target_ko: str, test_size: float, random_state: int):\
    X, y, issues = load_and_validate_training(df, target_ok, target_ko)\
    if issues:\
        return None, None, None, None, None, issues\
\
    mask = ~y.isna()\
    X = X.loc[mask].copy()\
    y = y.loc[mask].astype(int).copy()\
\
    numeric_features = ["EDAD", "IMC", "PASI INICIAL ADA", "NPREV"]\
    binary_features = ["HOMBRE", "MUJER", "ARTRITIS"]\
\
    numeric_transformer = Pipeline(steps=[\
        ("imputer", SimpleImputer(strategy="median")),\
        ("scaler", StandardScaler()),\
    ])\
\
    binary_transformer = Pipeline(steps=[\
        ("imputer", SimpleImputer(strategy="most_frequent")),\
    ])\
\
    preprocessor = ColumnTransformer(\
        transformers=[\
            ("num", numeric_transformer, numeric_features),\
            ("bin", binary_transformer, binary_features),\
        ],\
        remainder="drop"\
    )\
\
    clf = LogisticRegression(max_iter=2000)\
    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])\
\
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=test_size, random_state=random_state, stratify=y\
    )\
\
    model.fit(X_train, y_train)\
\
    proba_test = model.predict_proba(X_test)[:, 1]\
    pred_test = (proba_test >= 0.5).astype(int)\
\
    auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan\
    acc = accuracy_score(y_test, pred_test)\
    cm = confusion_matrix(y_test, pred_test)\
    report = classification_report(y_test, pred_test, output_dict=False)\
    thr = compute_optimal_thresholds(np.array(y_test), np.array(proba_test))\
\
    return model, (X_train, X_test, y_train, y_test), (auc, acc), cm, thr, [report]\
\
\
def validate_batch_inputs(df_new: pd.DataFrame) -> list[str]:\
    issues = []\
    missing = [c for c in FEATURES if c not in df_new.columns]\
    if missing:\
        issues.append(f"Faltan columnas de entrada en el Excel de pacientes nuevos: \{missing\}")\
        return issues\
\
    for b in ["HOMBRE", "MUJER", "ARTRITIS"]:\
        df_new[b] = clean_binary_series(df_new[b])\
\
    return issues\
\
\
def to_excel_bytes(df_out: pd.DataFrame, sheet_name: str = "predicciones") -> bytes:\
    output = io.BytesIO()\
    with pd.ExcelWriter(output, engine="openpyxl") as writer:\
        df_out.to_excel(writer, index=False, sheet_name=sheet_name)\
    return output.getvalue()\
\
\
def template_patients_excel_bytes() -> bytes:\
    """\
    Plantilla vac\'eda con las columnas correctas para 'pacientes nuevos'.\
    """\
    df_template = pd.DataFrame(columns=FEATURES)\
    return to_excel_bytes(df_template, sheet_name="plantilla")\
\
\
# =========================\
# UI\
# =========================\
\
st.subheader("1) Excel de entrenamiento \uc0\u8594  Entrenamiento autom\'e1tico")\
uploaded = st.file_uploader("Sube tu archivo .xlsx (con salidas PASI75/PASI90)", type=["xlsx"])\
objective = st.selectbox("Objetivo a predecir", ["PASI75", "PASI90"])\
\
col1, col2 = st.columns(2)\
with col1:\
    test_size = st.slider("Tama\'f1o de test", 0.1, 0.4, 0.25, 0.05)\
with col2:\
    random_state = st.number_input("Semilla (reproducibilidad)", min_value=0, max_value=9999, value=42, step=1)\
\
if uploaded is None:\
    st.info("Sube tu Excel de entrenamiento para entrenar el modelo autom\'e1ticamente.")\
    st.stop()\
\
try:\
    df = pd.read_excel(uploaded)\
except Exception as e:\
    st.error("No pude leer el Excel. Aseg\'farate de que es un .xlsx v\'e1lido.")\
    st.exception(e)\
    st.stop()\
\
st.write("Vista previa del archivo de entrenamiento:")\
st.dataframe(df.head(10))\
\
# Resumen de clases + alertas\
st.subheader("Resumen autom\'e1tico de clases en tu Excel")\
summaries = []\
all_msgs = []\
block_for = set()\
\
for obj in ["PASI75", "PASI90"]:\
    t_ok = TARGETS[obj]["ok"]\
    t_ko = TARGETS[obj]["ko"]\
    s = class_summary(df, t_ok, t_ko)\
\
    if s["has_ok"]:\
        block, msgs = warn_or_block_for_small_classes(s["n_ok"], s["n_ko_from_ok"], obj)\
        all_msgs.extend(msgs)\
        if block:\
            block_for.add(obj)\
\
    summaries.append(\{\
        "Objetivo": obj,\
        "Columna OK": t_ok if s["has_ok"] else "NO EXISTE",\
        "Columna KO": t_ko if s["has_ko"] else "NO EXISTE",\
        "N con etiqueta (OK no NaN)": s["n_labeled"] if s["has_ok"] else 0,\
        "Respondedores (OK=1)": s["n_ok"] if s["has_ok"] else 0,\
        "No respondedores (OK=0)": s["n_ko_from_ok"] if s["has_ok"] else 0,\
        "% OK": f'\{s["pct_ok"]*100:.1f\}%' if s["has_ok"] and s["n_labeled"] > 0 else "\'97",\
        "Inconsist. OK=1 y KO=1": s["incons_both_1"] if s["has_ko"] else "\'97",\
        "Inconsist. OK=0 y KO=0": s["incons_both_0"] if s["has_ko"] else "\'97",\
    \})\
\
st.dataframe(pd.DataFrame(summaries), use_container_width=True)\
\
if all_msgs:\
    st.subheader("Avisos autom\'e1ticos")\
    for m in all_msgs:\
        if m.startswith("\uc0\u10060 "):\
            st.error(m)\
        else:\
            st.warning(m)\
\
target_ok = TARGETS[objective]["ok"]\
target_ko = TARGETS[objective]["ko"]\
\
if objective in block_for:\
    st.stop()\
\
with st.spinner(f"Entrenando modelo para \{objective\} (autom\'e1tico)..."):\
    model, splits, metrics, cm, thr, extra = train_model_from_df(\
        df, target_ok, target_ko, float(test_size), int(random_state)\
    )\
\
if model is None:\
    st.error("No se pudo entrenar por problemas en columnas/valores.")\
    st.write("Revisa estos puntos:")\
    for it in extra:\
        st.code(it)\
    st.stop()\
\
auc, acc = metrics\
st.success(f"Modelo entrenado correctamente \uc0\u9989  (\{objective\})")\
\
m1, m2 = st.columns(2)\
m1.metric("AUC (test)", f"\{auc:.3f\}" if not np.isnan(auc) else "N/A")\
m2.metric("Accuracy (test)", f"\{acc:.3f\}")\
\
st.write("Matriz de confusi\'f3n (test) [ [TN, FP], [FN, TP] ]:")\
st.code(cm)\
\
with st.expander("Informe (classification report)"):\
    st.text(extra[0])\
\
st.subheader("Umbrales recomendados (calculados en el test)")\
colA, colB = st.columns(2)\
colA.metric("Youden \'f3ptimo", f'\{thr["youden"]["thr"]:.2f\}')\
colB.metric("F1 \'f3ptimo", f'\{thr["f1"]["thr"]:.2f\}')\
\
with st.expander("Detalles de los umbrales"):\
    st.write(\
        f'- Youden: thr=\{thr["youden"]["thr"]:.3f\}, '\
        f'Youden=\{thr["youden"]["youden"]:.3f\}, '\
        f'Sens=\{thr["youden"]["sens"]:.3f\}, '\
        f'Esp=\{thr["youden"]["spec"]:.3f\}\\n'\
        f'- F1: thr=\{thr["f1"]["thr"]:.3f\}, F1=\{thr["f1"]["f1"]:.3f\}'\
    )\
\
st.divider()\
st.subheader("2) Predicci\'f3n individual")\
\
with st.form("predict_form"):\
    sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)\
    hombre = 1 if sexo == "Hombre" else 0\
    mujer = 1 if sexo == "Mujer" else 0\
\
    edad = st.number_input("EDAD (a\'f1os)", min_value=0, max_value=120, value=45, step=1)\
    imc = st.number_input("IMC (kg/m\'b2)", min_value=10.0, max_value=80.0, value=27.0, step=0.1)\
    pasi_ini = st.number_input("PASI INICIAL ADA", min_value=0.0, max_value=80.0, value=12.0, step=0.1)\
    artritis = st.selectbox("ARTRITIS", ["No", "S\'ed"])\
    artritis_val = 1 if artritis == "S\'ed" else 0\
    nprev = st.number_input("NPREV (n\'ba biol\'f3gicos previos)", min_value=0, max_value=50, value=0, step=1)\
\
    threshold_mode = st.radio(\
        "Umbral para clasificar",\
        options=["Manual", "Recomendado (Youden)", "Recomendado (F1)"],\
        horizontal=True\
    )\
\
    if threshold_mode == "Manual":\
        threshold = st.slider("Umbral (manual)", 0.05, 0.95, 0.50, 0.01)\
    elif threshold_mode == "Recomendado (Youden)":\
        threshold = float(thr["youden"]["thr"])\
        st.info(f"Usando umbral Youden: \{threshold:.2f\}")\
    else:\
        threshold = float(thr["f1"]["thr"])\
        st.info(f"Usando umbral F1: \{threshold:.2f\}")\
\
    go = st.form_submit_button("Predecir (individual)")\
\
if go:\
    x_patient = pd.DataFrame([\{\
        "HOMBRE": hombre,\
        "MUJER": mujer,\
        "EDAD": float(edad),\
        "IMC": float(imc),\
        "PASI INICIAL ADA": float(pasi_ini),\
        "ARTRITIS": artritis_val,\
        "NPREV": float(nprev),\
    \}])\
\
    prob = float(model.predict_proba(x_patient)[:, 1][0])\
    st.metric(f"Probabilidad de respuesta (\{objective\})", f"\{prob*100:.1f\}%")\
\
    label = "Respondedor" if prob >= threshold else "No respondedor"\
    if label == "Respondedor":\
        st.success(f"Clasificaci\'f3n: **\{label\}** (\uc0\u8805  \{threshold:.2f\})")\
    else:\
        st.error(f"Clasificaci\'f3n: **\{label\}** (< \{threshold:.2f\})")\
\
    with st.expander("Datos usados"):\
        st.dataframe(x_patient)\
\
st.divider()\
st.subheader("3) Predicci\'f3n por lote (Excel de pacientes nuevos)")\
\
# \uc0\u9989  Bot\'f3n plantilla\
template_bytes = template_patients_excel_bytes()\
st.download_button(\
    label="Descargar plantilla Excel de pacientes nuevos",\
    data=template_bytes,\
    file_name="plantilla_pacientes_nuevos.xlsx",\
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"\
)\
\
st.write(\
    "Sube un Excel con **solo las columnas de entrada** (puede incluir otras columnas como ID). "\
    "La app a\'f1adir\'e1 columnas de probabilidad y clasificaci\'f3n y podr\'e1s descargar el resultado."\
)\
\
batch_file = st.file_uploader("Excel de pacientes nuevos (.xlsx)", type=["xlsx"], key="batch")\
\
if batch_file is not None:\
    try:\
        df_new = pd.read_excel(batch_file)\
    except Exception as e:\
        st.error("No pude leer el Excel de pacientes nuevos.")\
        st.exception(e)\
        st.stop()\
\
    st.write("Vista previa (pacientes nuevos):")\
    st.dataframe(df_new.head(10))\
\
    issues = validate_batch_inputs(df_new)\
    if issues:\
        st.error("Problemas con el Excel de pacientes nuevos:")\
        for it in issues:\
            st.write(f"- \{it\}")\
        st.stop()\
\
    st.write("Selecciona el umbral para clasificar en el lote:")\
    batch_threshold_mode = st.radio(\
        "Umbral para lote",\
        options=["Manual", "Recomendado (Youden)", "Recomendado (F1)"],\
        horizontal=True,\
        key="batch_thr_mode"\
    )\
\
    if batch_threshold_mode == "Manual":\
        batch_threshold = st.slider("Umbral (manual) \'96 lote", 0.05, 0.95, 0.50, 0.01, key="batch_thr")\
    elif batch_threshold_mode == "Recomendado (Youden)":\
        batch_threshold = float(thr["youden"]["thr"])\
        st.info(f"Usando umbral Youden: \{batch_threshold:.2f\}")\
    else:\
        batch_threshold = float(thr["f1"]["thr"])\
        st.info(f"Usando umbral F1: \{batch_threshold:.2f\}")\
\
    if st.button("Predecir lote"):\
        df_out = df_new.copy()\
\
        X_new = df_out[FEATURES].copy()\
        for b in ["HOMBRE", "MUJER", "ARTRITIS"]:\
            X_new[b] = clean_binary_series(X_new[b])\
\
        probs = model.predict_proba(X_new)[:, 1]\
        df_out[f"PROB_\{objective\}"] = probs\
        df_out[f"RESP_\{objective\}"] = np.where(df_out[f"PROB_\{objective\}"] >= batch_threshold, "Respondedor", "No respondedor")\
        df_out["UMBRAL_USADO"] = float(batch_threshold)\
\
        st.success("Predicci\'f3n por lote completada \uc0\u9989 ")\
        st.dataframe(df_out.head(20), use_container_width=True)\
\
        excel_bytes = to_excel_bytes(df_out, sheet_name="predicciones")\
        st.download_button(\
            label="Descargar Excel con predicciones",\
            data=excel_bytes,\
            file_name=f"predicciones_\{objective\}.xlsx",\
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"\
        )\
\
st.caption("Si cambias el Excel de entrenamiento, el objetivo o el split, el modelo y los umbrales se recalculan.")\
}