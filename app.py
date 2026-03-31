# ════════════════════════════════════════════════════════════════════════════
# app.py — PVC Film Yellowing Index (YI) Prediction Dashboard
# Streamlit + TabPFN Client (Prior Labs API)
# ════════════════════════════════════════════════════════════════════════════
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ── TabPFN 경로 패치 ──────────────────────────────────────────────────────
TABPFN_TMP = pathlib.Path("/tmp/.tabpfn")
TABPFN_TMP.mkdir(parents=True, exist_ok=True)

from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.client import ServiceClient
import tabpfn_client
from tabpfn_client import constants

# 토큰 파일 경로 패치
UserAuthenticationClient.CACHED_TOKEN_FILE = TABPFN_TMP / "token"

# 모든 Path 속성 일괄 패치
for cls in [ServiceClient, UserAuthenticationClient]:
    for attr in dir(cls):
        try:
            val = getattr(cls, attr)
            if isinstance(val, pathlib.Path) and ".tabpfn" in str(val):
                setattr(cls, attr, TABPFN_TMP / val.name)
        except Exception:
            pass

# constants 패치
constants.CACHE_DIR = TABPFN_TMP

# dataset_uid_cache_manager 패치 (fit 시 캐시 저장 경로)
try:
    mgr = ServiceClient.dataset_uid_cache_manager
    mgr.file_path = str(TABPFN_TMP / "dataset_uid_cache.json")
except Exception:
    pass

# ServiceClient 내부 str 경로 속성 전체 스캔
for attr in dir(ServiceClient):
    try:
        val = getattr(ServiceClient, attr)
        if isinstance(val, str) and ".tabpfn" in val:
            setattr(ServiceClient, attr, str(TABPFN_TMP / pathlib.Path(val).name))
        elif hasattr(val, "file_path") and ".tabpfn" in str(getattr(val, "file_path", "")):
            val.file_path = str(TABPFN_TMP / pathlib.Path(val.file_path).name)
    except Exception:
        pass

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PVC Film YI Simulator",
    page_icon="🧪",
    layout="wide",
)

# ── 토큰 인증 ─────────────────────────────────────────────────────────────
try:
    token = st.secrets["TABPFN_TOKEN"]
except Exception:
    st.error("⚠️ TABPFN_TOKEN not found in Streamlit Secrets.")
    st.stop()

tabpfn_client.set_access_token(token)
from tabpfn_client import TabPFNRegressor

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    "pvc_dp":             ("DP",       "Degree of Polymerization",  500,  1300, 800,  10),
    "plasticizer_phr":    ("Plast.",   "Plasticizer (phr)",          30,    80,  50,  0.5),
    "stabilizer_phr":     ("Stab.",    "Heat Stabilizer (phr)",      1.0,  5.0, 2.5,  0.1),
    "process_temp_c":     ("Temp.",    "Processing Temp. (°C)",     160,   200, 180,  1.0),
    "residence_time_min": ("Time",     "Residence Time (min)",        1,    10,   5,  0.1),
    "uv_absorber_phr":    ("UV Abs.", "UV Absorber (phr)",           0.1,  1.0, 0.5, 0.05),
    "antioxidant_phr":    ("AO",      "Antioxidant (phr)",           0.1,  0.5, 0.2, 0.05),
}
FEATURE_NAMES = list(FEATURE_META.keys())
SIM_RANGES    = {k: (v[2], v[3]) for k, v in FEATURE_META.items()}
YI_LIMIT      = 25.0

# ════════════════════════════════════════════════════════════════════════════
# Synthetic training data
# ════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int = 300, seed: int = 42):
    np.random.seed(seed)
    rng = {k: np.random.uniform(*v, n) for k, v in SIM_RANGES.items()}
    yi = (
          (rng["process_temp_c"] - 160) * 0.90
        + rng["residence_time_min"]      * 3.50
        + rng["plasticizer_phr"]         * 0.10
        - rng["stabilizer_phr"]          * 5.50
        - rng["pvc_dp"]                  / 220
        - rng["uv_absorber_phr"]         * 6.00
        - rng["antioxidant_phr"]         * 8.00
        + np.random.normal(0, 1.5, n)
    )
    return pd.DataFrame(rng), np.clip(yi + 10, 1.0, 70.0)

# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training TabPFN model …")
def load_model(real_data_hash: str = "none") -> TabPFNRegressor:
    X_syn, y_syn = generate_synthetic_data()

    if "real_df" in st.session_state and st.session_state.real_df is not None:
        real_df = st.session_state.real_df
        real_X  = real_df[FEATURE_NAMES]
        real_y  = real_df["target_YI"].values
        X_train = pd.concat([X_syn] + [real_X] * 3, ignore_index=True)
        y_train = np.concatenate([y_syn] + [real_y] * 3)
    else:
        X_train, y_train = X_syn, y_syn

    m = TabPFNRegressor()
    m.fit(X_train, y_train)
    return m

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🧪 PVC Film YI Simulator")
st.sidebar.markdown("Adjust formulation parameters to predict Yellowing Index (YI).")
st.sidebar.divider()
st.sidebar.subheader("Formulation Parameters")

current = {
    feat: st.sidebar.slider(
        label=meta[1],
        min_value=float(meta[2]),
        max_value=float(meta[3]),
        value=float(meta[4]),
        step=float(meta[5]),
    )
    for feat, meta in FEATURE_META.items()
}

st.sidebar.divider()
st.sidebar.subheader("📂 Experimental Data (optional)")
st.sidebar.caption(
    "Upload an Excel/CSV with feature columns + target_YI. "
    "Model will retrain with your data weighted 3x."
)
uploaded_file = st.sidebar.file_uploader("Upload (.xlsx or .csv)", type=["xlsx", "csv"])

real_data_hash = "none"
if uploaded_file is not None:
    try:
        real_df = (pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx")
                   else pd.read_csv(uploaded_file))
        missing = [f for f in FEATURE_NAMES + ["target_YI"] if f not in real_df.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            st.session_state.real_df = real_df
            real_data_hash = str(pd.util.hash_pandas_object(real_df).sum())
            st.sidebar.success(
                f"✅ {len(real_df)} records loaded. "
                f"YI range: [{real_df['target_YI'].min():.1f}, {real_df['target_YI'].max():.1f}]"
            )
            with st.sidebar.expander("Preview data"):
                st.dataframe(real_df.head())
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

with st.sidebar.expander("Expected column names"):
    st.code("\n".join(FEATURE_NAMES + ["target_YI"]))

# ════════════════════════════════════════════════════════════════════════════
# Load model + Predict
# ════════════════════════════════════════════════════════════════════════════
model   = load_model(real_data_hash=real_data_hash)
x_now   = pd.DataFrame([[current[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
yi_pred = float(model.predict(x_now)[0])

if yi_pred < 8:
    grade_color, grade_label = "#27ae60", "Excellent  (YI < 8)"
elif yi_pred < 15:
    grade_color, grade_label = "#2980b9", "Good  (YI 8-15)"
elif yi_pred < YI_LIMIT:
    grade_color, grade_label = "#e67e22", "Marginal  (YI 15-25)"
else:
    grade_color, grade_label = "#c0392b", "Poor  (YI >= 25)"

# ════════════════════════════════════════════════════════════════════════════
# Dashboard header
# ════════════════════════════════════════════════════════════════════════════
st.title("PVC Film Yellowing Index (YI) Prediction Dashboard")
st.caption("Model: TabPFN Regressor (Prior Labs API)  |  Training: physics-based synthetic data")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted YI",    f"{yi_pred:.2f}")
c2.metric("Grade",            grade_label)
c3.metric("YI Limit",         f"{YI_LIMIT:.0f}")
c4.metric("Margin to Limit",  f"{YI_LIMIT - yi_pred:+.2f}")
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Batch inference
# ════════════════════════════════════════════════════════════════════════════
N_SWEEP   = 30
all_rows  = []
sweep_idx = {}
idx       = 0
base      = [current[f] for f in FEATURE_NAMES]

for i, feat in enumerate(FEATURE_NAMES):
    sweep = np.linspace(*SIM_RANGES[feat], N_SWEEP)
    sweep_idx[feat] = (idx, sweep, i)
    for val in sweep:
        row = base.copy(); row[i] = val
        all_rows.append(row)
    idx += N_SWEEP

impact_rows = []
for i, feat in enumerate(FEATURE_NAMES):
    lo, hi = SIM_RANGES[feat]
    row_lo = base.copy(); row_lo[i] = lo
    row_hi = base.copy(); row_hi[i] = hi
    impact_rows.extend([row_lo, row_hi])

X_all    = pd.DataFrame(all_rows + impact_rows, columns=FEATURE_NAMES)
y_all    = model.predict(X_all)
y_sweep  = y_all[:N_SWEEP * len(FEATURE_NAMES)]
y_impact = y_all[N_SWEEP * len(FEATURE_NAMES):]

# ════════════════════════════════════════════════════════════════════════════
# Plot row 1
# ════════════════════════════════════════════════════════════════════════════
col_g, col_p, col_i = st.columns([1, 1.4, 1.2])
labels_full = [FEATURE_META[f][1] for f in FEATURE_NAMES]

with col_g:
    fig_g, ax_g = plt.subplots(figsize=(4, 2.5))
    fig_g.patch.set_facecolor("#f0f2f5")
    ax_g.set_facecolor("#ffffff")
    ax_g.barh([""], [yi_pred], color=grade_color, height=0.45, zorder=3)
    ax_g.barh([""], [70 - yi_pred], left=[yi_pred], color="#dfe6e9", height=0.45, zorder=2)
    ax_g.axvline(YI_LIMIT, color="#7f8c8d", linestyle="--", lw=1.5, label=f"Limit={YI_LIMIT:.0f}")
    ax_g.set_xlim(0, 70)
    ax_g.text(min(yi_pred + 1.5, 58), 0, f"{yi_pred:.1f}",
              va="center", fontsize=16, fontweight="bold", color=grade_color)
    ax_g.set_title("Predicted YI", fontsize=10, fontweight="bold")
    ax_g.legend(fontsize=8, loc="upper right")
    ax_g.set_yticks([])
    ax_g.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_g)

with col_p:
    normalized = [
        (current[f] - FEATURE_META[f][2]) / (FEATURE_META[f][3] - FEATURE_META[f][2])
        for f in FEATURE_NAMES
    ]
    bar_colors = ["#e74c3c" if n > 0.75 else "#27ae60" if n < 0.25 else "#3498db"
                  for n in normalized]
    fig_p, ax_p = plt.subplots(figsize=(5.5, 2.8))
    fig_p.patch.set_facecolor("#f0f2f5")
    ax_p.set_facecolor("#ffffff")
    bars = ax_p.barh(labels_full, normalized, color=bar_colors, height=0.5)
    ax_p.axvline(0.5, color="#95a5a6", linestyle=":", lw=1.2)
    ax_p.set_xlim(0, 1.1)
    ax_p.set_title("Formulation Position in Range", fontsize=10, fontweight="bold")
    ax_p.set_xlabel("Relative position (0=min, 1=max)", fontsize=8)
    for bar, n, feat in zip(bars, normalized, FEATURE_NAMES):
        ax_p.text(min(n + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                  f"{current[feat]:.2g}", va="center", fontsize=8)
    ax_p.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_p)

with col_i:
    deltas = [float(y_impact[j*2+1]) - float(y_impact[j*2]) for j in range(len(FEATURE_NAMES))]
    imp_colors = ["#e74c3c" if d > 0 else "#27ae60" for d in deltas]
    fig_i, ax_i = plt.subplots(figsize=(4.5, 2.8))
    fig_i.patch.set_facecolor("#f0f2f5")
    ax_i.set_facecolor("#ffffff")
    ax_i.barh(labels_full, deltas, color=imp_colors, height=0.5)
    ax_i.axvline(0, color="black", lw=0.8)
    ax_i.set_title("YI Impact  (min to max)", fontsize=10, fontweight="bold")
    ax_i.set_xlabel("Delta YI", fontsize=8)
    ax_i.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Increases YI"),
        mpatches.Patch(color="#27ae60", label="Decreases YI"),
    ], fontsize=7)
    ax_i.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_i)

# ════════════════════════════════════════════════════════════════════════════
# Plot row 2 — Sensitivity curves
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Sensitivity Analysis — YI vs. Each Parameter")
st.caption("Red dashed line = current value.  Gray dotted line = acceptance limit.")

fig_s, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(22, 3.5))
fig_s.patch.set_facecolor("#f0f2f5")

for ax, feat in zip(axes, FEATURE_NAMES):
    start, sweep, fi = sweep_idx[feat]
    yi_vals = np.clip(y_sweep[start:start + N_SWEEP], 0, 75)
    ax.set_facecolor("#ffffff")
    ax.plot(sweep, yi_vals, color="#2980b9", lw=2.0, zorder=3)
    ax.fill_between(sweep, yi_vals, YI_LIMIT,
                    where=[y > YI_LIMIT for y in yi_vals],
                    alpha=0.2, color="#e74c3c")
    ax.axvline(current[feat], color="#e74c3c", linestyle="--", lw=1.5, zorder=4)
    ax.axhline(YI_LIMIT,      color="#95a5a6", linestyle=":",  lw=1.0)
    ax.scatter([current[feat]], [yi_pred], color="#e74c3c", s=40, zorder=5)
    ax.set_ylim(0, 75)
    ax.set_title(FEATURE_META[feat][0], fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    ax.tick_params(labelsize=6)

axes[0].set_ylabel("YI", fontsize=8)
plt.tight_layout()
st.pyplot(fig_s)

# ════════════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📋 Experimental data upload format"):
    st.markdown("Upload an Excel or CSV file with the following columns:")
    st.code(" | ".join(FEATURE_NAMES + ["target_YI"]), language=None)
    st.markdown(
        "When uploaded, the model retrains merging synthetic data "
        "with your records (experimental rows weighted 3x). "
        "Recommended: 5+ records for meaningful influence, 20+ for full replacement."
    )
