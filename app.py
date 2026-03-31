# ════════════════════════════════════════════════════════════════════════════
# app.py — PVC Film Yellowing Index (YI) Prediction Dashboard
# Streamlit + TabPFN Client (Prior Labs API)
# Deploy: GitHub → Streamlit Cloud
# ════════════════════════════════════════════════════════════════════════════

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ← 이 줄 추가 (서버 환경 필수)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PVC Film YI Simulator",
    page_icon="🧪",
    layout="wide",
)

# ── TabPFN token injection (secrets → env var, no interactive login) ─────
# Streamlit Cloud: set TABPFN_TOKEN in app Settings → Secrets
# Local: add to .streamlit/secrets.toml
if "TABPFN_TOKEN" in st.secrets:
    os.environ["TABPFN_ACCESS_TOKEN"] = st.secrets["TABPFN_TOKEN"]
else:
    st.error(
        "⚠️ TABPFN_TOKEN not found in secrets. "
        "Add it to .streamlit/secrets.toml (local) or Streamlit Cloud Secrets."
    )
    st.stop()

from tabpfn_client import TabPFNRegressor
import tabpfn_client

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    # key: (short_label, full_label, min, max, default, step)
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

YI_LIMIT = 25.0  # acceptance threshold

# ════════════════════════════════════════════════════════════════════════════
# Synthetic training data (physics-based DHC + oxidation model)
# Replace / supplement with real experimental data when available
# ════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Simplified yellowing physics:
      YI ~ f(temp, time, plasticizer) - g(stabilizer, DP, UV absorber, antioxidant)

    Replace this function with real lab data when ≥10 records are available.
    """
    np.random.seed(seed)
    rng = {k: np.random.uniform(*v, n) for k, v in SIM_RANGES.items()}
    yi = (
          (rng["process_temp_c"] - 160) * 0.90   # thermal DHC
        + rng["residence_time_min"]      * 3.50   # cumulative heat exposure
        + rng["plasticizer_phr"]         * 0.10   # plasticizer migration
        - rng["stabilizer_phr"]          * 5.50   # stabilizer defense
        - rng["pvc_dp"]                  / 220    # high DP suppression
        - rng["uv_absorber_phr"]         * 6.00   # UV absorption
        - rng["antioxidant_phr"]         * 8.00   # radical inhibition
        + np.random.normal(0, 1.5, n)             # experimental noise
    )
    return pd.DataFrame(rng), np.clip(yi + 10, 1.0, 70.0)


# ════════════════════════════════════════════════════════════════════════════
# Model — cached so Streamlit does not retrain on every slider move
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training TabPFN model …")
def load_model(real_data_hash: str = "none") -> TabPFNRegressor:
    
    # 토큰을 환경변수에서 직접 읽어 로그인 — 대화형 프롬프트 없이 자동 처리
    token = os.environ.get("TABPFN_ACCESS_TOKEN", "")
    if not token:
        st.error("TABPFN_TOKEN not set.")
        st.stop()
    
    tabpfn_client.init(use_server=True, access_token=token)  # ← 토큰 직접 전달
    
    X_syn, y_syn = generate_synthetic_data()

    # ── Experimental data slot ───────────────────────────────────────────
    # When real_df is stored in session state, merge here (3× weight).
    # This block is intentionally left ready — no real data yet.
    if "real_df" in st.session_state and st.session_state.real_df is not None:
        real_df = st.session_state.real_df
        real_X  = real_df[FEATURE_NAMES]
        real_y  = real_df["target_YI"].values
        # Oversample real records to give them stronger influence
        X_train = pd.concat([X_syn] + [real_X] * 3, ignore_index=True)
        y_train = np.concatenate([y_syn] + [real_y] * 3)
    else:
        X_train, y_train = X_syn, y_syn

    m = TabPFNRegressor()
    m.fit(X_train, y_train)
    return m


# ════════════════════════════════════════════════════════════════════════════
# Sidebar — sliders + data upload
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

# ── Experimental data upload (placeholder for future use) ────────────────
st.sidebar.divider()
st.sidebar.subheader("📂 Experimental Data (optional)")
st.sidebar.caption(
    "Upload an Excel/CSV with columns matching the feature names above "
    "plus a `target_YI` column. The model will retrain incorporating your data."
)
uploaded_file = st.sidebar.file_uploader(
    "Upload (.xlsx or .csv)", type=["xlsx", "csv"]
)

real_data_hash = "none"
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            real_df = pd.read_excel(uploaded_file)
        else:
            real_df = pd.read_csv(uploaded_file)

        missing = [f for f in FEATURE_NAMES + ["target_YI"] if f not in real_df.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            st.session_state.real_df = real_df
            real_data_hash = str(pd.util.hash_pandas_object(real_df).sum())
            st.sidebar.success(
                f"✅ {len(real_df)} records loaded. "
                f"YI range: [{real_df['target_YI'].min():.1f}, "
                f"{real_df['target_YI'].max():.1f}]"
            )
            with st.sidebar.expander("Preview data"):
                st.dataframe(real_df.head())
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

# ── Expected column names helper ─────────────────────────────────────────
with st.sidebar.expander("Expected column names"):
    st.code("\n".join(FEATURE_NAMES + ["target_YI"]))

# ════════════════════════════════════════════════════════════════════════════
# Load / retrain model
# ════════════════════════════════════════════════════════════════════════════
model = load_model(real_data_hash=real_data_hash)

# ════════════════════════════════════════════════════════════════════════════
# Prediction
# ════════════════════════════════════════════════════════════════════════════
x_now   = pd.DataFrame([[current[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
yi_pred = float(model.predict(x_now)[0])

if yi_pred < 8:
    grade_color, grade_label = "#27ae60", "Excellent  (YI < 8)"
elif yi_pred < 15:
    grade_color, grade_label = "#2980b9", "Good  (YI 8–15)"
elif yi_pred < YI_LIMIT:
    grade_color, grade_label = "#e67e22", "Marginal  (YI 15–25)"
else:
    grade_color, grade_label = "#c0392b", "Poor  (YI ≥ 25)"

# ════════════════════════════════════════════════════════════════════════════
# Dashboard header
# ════════════════════════════════════════════════════════════════════════════
st.title("PVC Film Yellowing Index (YI) Prediction Dashboard")
st.caption("Model: TabPFN Regressor (Prior Labs API)  |  Training: physics-based synthetic data")

# ── Top KPI row ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted YI",  f"{yi_pred:.2f}")
c2.metric("Grade",          grade_label)
c3.metric("YI Limit",       f"{YI_LIMIT:.0f}")
c4.metric("Margin to Limit", f"{YI_LIMIT - yi_pred:+.2f}")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Batch inference for all plots (single API call)
# ════════════════════════════════════════════════════════════════════════════
N_SWEEP  = 30
all_rows = []
sweep_idx = {}
idx = 0
base = [current[f] for f in FEATURE_NAMES]

for i, feat in enumerate(FEATURE_NAMES):
    sweep = np.linspace(*SIM_RANGES[feat], N_SWEEP)
    sweep_idx[feat] = (idx, sweep, i)
    for val in sweep:
        row    = base.copy()
        row[i] = val
        all_rows.append(row)
    idx += N_SWEEP

# Impact (min→max) rows
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
# Plot row 1 — YI gauge + Formulation position + Impact bar
# ════════════════════════════════════════════════════════════════════════════
col_g, col_p, col_i = st.columns([1, 1.4, 1.2])

# ─ YI Gauge ───────────────────────────────────────────────────────────────
with col_g:
    fig_g, ax_g = plt.subplots(figsize=(4, 2.5))
    fig_g.patch.set_facecolor("#f0f2f5")
    ax_g.set_facecolor("#ffffff")
    ax_g.barh([""], [yi_pred],            color=grade_color, height=0.45, zorder=3)
    ax_g.barh([""], [70 - yi_pred], left=[yi_pred],
              color="#dfe6e9", height=0.45, zorder=2)
    ax_g.axvline(YI_LIMIT, color="#7f8c8d", linestyle="--", lw=1.5, label=f"Limit = {YI_LIMIT:.0f}")
    ax_g.set_xlim(0, 70)
    ax_g.text(min(yi_pred + 1.5, 58), 0, f"{yi_pred:.1f}",
              va="center", fontsize=16, fontweight="bold", color=grade_color)
    ax_g.set_title("Predicted YI", fontsize=10, fontweight="bold")
    ax_g.legend(fontsize=8, loc="upper right")
    ax_g.set_yticks([])
    ax_g.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_g)

# ─ Formulation position ───────────────────────────────────────────────────
with col_p:
    normalized = [
        (current[f] - FEATURE_META[f][2]) / (FEATURE_META[f][3] - FEATURE_META[f][2])
        for f in FEATURE_NAMES
    ]
    bar_colors = ["#e74c3c" if n > 0.75 else "#27ae60" if n < 0.25 else "#3498db"
                  for n in normalized]
    labels_full = [FEATURE_META[f][1] for f in FEATURE_NAMES]

    fig_p, ax_p = plt.subplots(figsize=(5.5, 2.8))
    fig_p.patch.set_facecolor("#f0f2f5")
    ax_p.set_facecolor("#ffffff")
    bars = ax_p.barh(labels_full, normalized, color=bar_colors, height=0.5)
    ax_p.axvline(0.5, color="#95a5a6", linestyle=":", lw=1.2)
    ax_p.set_xlim(0, 1.1)
    ax_p.set_title("Formulation Position in Range", fontsize=10, fontweight="bold")
    ax_p.set_xlabel("Relative position (0 = min, 1 = max)", fontsize=8)
    for bar, n, feat in zip(bars, normalized, FEATURE_NAMES):
        ax_p.text(min(n + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                  f"{current[feat]:.2g}", va="center", fontsize=8)
    ax_p.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_p)

# ─ YI Impact (min → max) ─────────────────────────────────────────────────
with col_i:
    deltas = []
    for j, feat in enumerate(FEATURE_NAMES):
        yi_lo  = float(y_impact[j * 2])
        yi_hi  = float(y_impact[j * 2 + 1])
        deltas.append(yi_hi - yi_lo)

    imp_colors = ["#e74c3c" if d > 0 else "#27ae60" for d in deltas]

    fig_i, ax_i = plt.subplots(figsize=(4.5, 2.8))
    fig_i.patch.set_facecolor("#f0f2f5")
    ax_i.set_facecolor("#ffffff")
    ax_i.barh(labels_full, deltas, color=imp_colors, height=0.5)
    ax_i.axvline(0, color="black", lw=0.8)
    ax_i.set_title("YI Impact  (min → max)", fontsize=10, fontweight="bold")
    ax_i.set_xlabel("ΔYI", fontsize=8)
    pos_p = mpatches.Patch(color="#e74c3c", label="Increases YI")
    neg_p = mpatches.Patch(color="#27ae60", label="Decreases YI")
    ax_i.legend(handles=[pos_p, neg_p], fontsize=7)
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
    ax.axvline(current[feat], color="#e74c3c",  linestyle="--", lw=1.5, zorder=4)
    ax.axhline(YI_LIMIT,      color="#95a5a6",  linestyle=":",  lw=1.0)
    ax.scatter([current[feat]], [yi_pred], color="#e74c3c", s=40, zorder=5)
    ax.set_ylim(0, 75)
    ax.set_title(FEATURE_META[feat][0], fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    ax.tick_params(labelsize=6)

axes[0].set_ylabel("YI", fontsize=8)
plt.tight_layout()
st.pyplot(fig_s)

# ════════════════════════════════════════════════════════════════════════════
# Footer — column reference
# ════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📋 Experimental data upload format"):
    st.markdown("Upload an Excel or CSV file with the following columns:")
    col_str = " | ".join(FEATURE_NAMES + ["target_YI"])
    st.code(col_str, language=None)
    st.markdown(
        "When uploaded, the model retrains by merging synthetic baseline data "
        "with your records (experimental rows weighted 3x). "
        "Recommended: 5+ records for meaningful influence, 20+ for full replacement."
    )
