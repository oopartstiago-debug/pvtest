# ════════════════════════════════════════════════════════════════════════════
# app.py — PVC 필름 황변지수(YI) 예측 대시보드
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
# Streamlit Cloud는 패키지 내부 폴더에 쓰기 권한이 없으므로
# /tmp 경로로 토큰 및 캐시 파일 위치를 강제 변경합니다.
TABPFN_TMP = pathlib.Path("/tmp/.tabpfn")
TABPFN_TMP.mkdir(parents=True, exist_ok=True)

from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.client import ServiceClient
import tabpfn_client
from tabpfn_client import constants

UserAuthenticationClient.CACHED_TOKEN_FILE = TABPFN_TMP / "token"

for cls in [ServiceClient, UserAuthenticationClient]:
    for attr in dir(cls):
        try:
            val = getattr(cls, attr)
            if isinstance(val, pathlib.Path) and ".tabpfn" in str(val):
                setattr(cls, attr, TABPFN_TMP / val.name)
        except Exception:
            pass

constants.CACHE_DIR = TABPFN_TMP

try:
    mgr = ServiceClient.dataset_uid_cache_manager
    mgr.file_path = str(TABPFN_TMP / "dataset_uid_cache.json")
except Exception:
    pass

for attr in dir(ServiceClient):
    try:
        val = getattr(ServiceClient, attr)
        if isinstance(val, str) and ".tabpfn" in val:
            setattr(ServiceClient, attr, str(TABPFN_TMP / pathlib.Path(val).name))
        elif hasattr(val, "file_path") and ".tabpfn" in str(getattr(val, "file_path", "")):
            val.file_path = str(TABPFN_TMP / pathlib.Path(val.file_path).name)
    except Exception:
        pass

# ── 페이지 설정 ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PVC 필름 황변 예측 시스템",
    page_icon="🧪",
    layout="wide",
)

# ── TabPFN 토큰 인증 ──────────────────────────────────────────────────────
# Streamlit Cloud Secrets에 TABPFN_TOKEN을 등록해야 합니다.
# 로컬에서는 .streamlit/secrets.toml에 추가하세요.
try:
    token = st.secrets["TABPFN_TOKEN"]
except Exception:
    st.error("⚠️ TABPFN_TOKEN이 Streamlit Secrets에 없습니다. 설정을 확인하세요.")
    st.stop()

tabpfn_client.set_access_token(token)
from tabpfn_client import TabPFNRegressor

# ════════════════════════════════════════════════════════════════════════════
# 처방 변수 정의
# (변수명, 축약 라벨, 전체 라벨, 최솟값, 최댓값, 기본값, 슬라이더 단계)
# ════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    "pvc_dp": (
        "DP", "중합도 (Degree of Polymerization)",
        500, 1300, 800, 10,
        # 중합도가 높을수록 분자 사슬 얽힘이 증가해 탈염화수소(DHC) 반응 전파를 억제합니다.
    ),
    "plasticizer_phr": (
        "Plast.", "가소제 함량 (phr)",
        30, 80, 50, 0.5,
        # 가소제가 많을수록 분자 이동성이 증가해 산화 황변이 가속될 수 있습니다.
    ),
    "stabilizer_phr": (
        "Stab.", "열안정제 함량 (phr)",
        1.0, 5.0, 2.5, 0.1,
        # 열안정제는 DHC 반응을 직접 억제하는 핵심 방어 성분입니다.
    ),
    "process_temp_c": (
        "Temp.", "가공 온도 (°C)",
        160, 200, 180, 1.0,
        # 고온 가공 시 DHC 반응 속도가 지수적으로 증가해 YI가 급격히 상승합니다.
    ),
    "residence_time_min": (
        "Time", "체류 시간 (min)",
        1, 10, 5, 0.1,
        # 가공기 내 체류 시간이 길수록 열 노출이 누적되어 YI가 상승합니다.
    ),
    "uv_absorber_phr": (
        "UV Abs.", "UV흡수제 함량 (phr)",
        0.1, 1.0, 0.5, 0.05,
        # 광산화 개시 반응을 차단해 야외 노출 환경에서의 황변을 억제합니다.
    ),
    "antioxidant_phr": (
        "AO", "산화방지제 함량 (phr)",
        0.1, 0.5, 0.2, 0.05,
        # 자유 라디칼 연쇄 반응을 억제해 산화 열화에 의한 황변을 방어합니다.
    ),
}
FEATURE_NAMES = list(FEATURE_META.keys())
SIM_RANGES    = {k: (v[2], v[3]) for k, v in FEATURE_META.items()}

# 황변지수 허용 한계 (이 값 이하를 목표로 합니다)
YI_LIMIT = 25.0

# ════════════════════════════════════════════════════════════════════════════
# 가상 학습 데이터 생성 (물리 기반 황변 모델)
# ════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int = 300, seed: int = 42):
    """
    실제 실험 데이터가 없을 때 사용하는 물리 기반 시뮬레이션 데이터.
    PVC 황변의 핵심 반응(DHC + 산화 열화)을 수식으로 표현했습니다.

    실제 실험 데이터가 10건 이상 확보되면 이 함수를 대체하세요.
    """
    np.random.seed(seed)
    rng = {k: np.random.uniform(*v, n) for k, v in SIM_RANGES.items()}
    yi = (
          (rng["process_temp_c"] - 160) * 0.90   # 고온 가공에 의한 DHC 반응
        + rng["residence_time_min"]      * 3.50   # 누적 열 노출
        + rng["plasticizer_phr"]         * 0.10   # 가소제 마이그레이션 → 산화 가속
        - rng["stabilizer_phr"]          * 5.50   # 열안정제 방어 효과
        - rng["pvc_dp"]                  / 220    # 고중합도에 의한 DHC 전파 억제
        - rng["uv_absorber_phr"]         * 6.00   # 광산화 차단
        - rng["antioxidant_phr"]         * 8.00   # 라디칼 연쇄 반응 억제
        + np.random.normal(0, 1.5, n)             # 실험 오차 모사
    )
    return pd.DataFrame(rng), np.clip(yi + 10, 1.0, 70.0)

# ════════════════════════════════════════════════════════════════════════════
# TabPFN 모델 학습
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="TabPFN 모델 학습 중 … (최초 1회만 실행됩니다)")
def load_model(real_data_hash: str = "none") -> TabPFNRegressor:
    """
    real_data_hash: 실험 데이터 업로드 시 캐시를 갱신하기 위한 해시값.
    실험 데이터가 없으면 가상 데이터 300건으로 학습합니다.
    실험 데이터가 있으면 가상 데이터 + 실험 데이터(3배 가중)로 혼합 학습합니다.
    """
    X_syn, y_syn = generate_synthetic_data()

    if "real_df" in st.session_state and st.session_state.real_df is not None:
        real_df = st.session_state.real_df
        real_X  = real_df[FEATURE_NAMES]
        real_y  = real_df["target_YI"].values
        # 실험 데이터가 적을 때 3배 복제로 가중치를 높입니다
        X_train = pd.concat([X_syn] + [real_X] * 3, ignore_index=True)
        y_train = np.concatenate([y_syn] + [real_y] * 3)
    else:
        X_train, y_train = X_syn, y_syn

    m = TabPFNRegressor()
    m.fit(X_train, y_train)
    return m

# ── 단일 처방 YI 예측 헬퍼 ───────────────────────────────────────────────
def predict_yi(model, vals: dict) -> float:
    x = pd.DataFrame([[vals[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    return float(model.predict(x)[0])

# ════════════════════════════════════════════════════════════════════════════
# 사이드바 — 슬라이더 + 실험 데이터 업로드
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🧪 PVC 필름 황변 예측 시스템")
st.sidebar.markdown("처방 변수를 조절하여 황변지수(YI)를 실시간으로 예측합니다.")
st.sidebar.divider()
st.sidebar.subheader("처방 변수 설정")

current = {}
for feat, meta in FEATURE_META.items():
    current[feat] = st.sidebar.slider(
        label=meta[1],
        min_value=float(meta[2]),
        max_value=float(meta[3]),
        value=float(meta[4]),
        step=float(meta[5]),
        help=meta[6] if len(meta) > 6 else "",
    )

st.sidebar.divider()
st.sidebar.subheader("📂 실험 데이터 업로드 (선택)")
st.sidebar.caption(
    "실제 실험 결과가 있으면 업로드하세요. "
    "모델이 자동으로 재학습되어 예측 정확도가 높아집니다."
)
uploaded_file = st.sidebar.file_uploader(
    "파일 선택 (.xlsx 또는 .csv)", type=["xlsx", "csv"]
)

real_data_hash = "none"
if uploaded_file is not None:
    try:
        real_df = (pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx")
                   else pd.read_csv(uploaded_file))
        missing = [f for f in FEATURE_NAMES + ["target_YI"] if f not in real_df.columns]
        if missing:
            st.sidebar.error(f"누락된 컬럼: {missing}")
        else:
            st.session_state.real_df = real_df
            real_data_hash = str(pd.util.hash_pandas_object(real_df).sum())
            st.sidebar.success(
                f"✅ {len(real_df)}건 로드 완료. "
                f"YI 범위: [{real_df['target_YI'].min():.1f}, {real_df['target_YI'].max():.1f}]"
            )
            with st.sidebar.expander("데이터 미리보기"):
                st.dataframe(real_df.head())
    except Exception as e:
        st.sidebar.error(f"파일 읽기 실패: {e}")

with st.sidebar.expander("업로드 파일 컬럼명 확인"):
    st.caption("아래 컬럼명이 정확히 포함되어야 합니다.")
    st.code("\n".join(FEATURE_NAMES + ["target_YI"]))

# ════════════════════════════════════════════════════════════════════════════
# 모델 로드 및 예측
# ════════════════════════════════════════════════════════════════════════════
model   = load_model(real_data_hash=real_data_hash)
yi_pred = predict_yi(model, current)

# YI 등급 판정
if yi_pred < 8:
    grade_color, grade_label = "#27ae60", "우수  (YI < 8)"
elif yi_pred < 15:
    grade_color, grade_label = "#2980b9", "양호  (YI 8~15)"
elif yi_pred < YI_LIMIT:
    grade_color, grade_label = "#e67e22", "주의  (YI 15~25)"
else:
    grade_color, grade_label = "#c0392b", "불량  (YI ≥ 25)"

# ════════════════════════════════════════════════════════════════════════════
# 대시보드 헤더
# ════════════════════════════════════════════════════════════════════════════
st.title("PVC 필름 황변지수(YI) 예측 대시보드")
st.caption("모델: TabPFN Regressor (Prior Labs API)  |  학습 데이터: 물리 기반 가상 데이터 300건")

# 상단 핵심 지표 4개
c1, c2, c3, c4 = st.columns(4)
c1.metric("예측 황변지수 (YI)", f"{yi_pred:.2f}",
          help="현재 처방 조건에서 TabPFN이 예측한 황변지수입니다.")
c2.metric("등급", grade_label,
          help="YI 수준에 따른 품질 등급입니다.")
c3.metric("허용 한계 (YI)", f"{YI_LIMIT:.0f}",
          help="이 값 이하를 목표로 처방을 설계해야 합니다.")
c4.metric("한계까지 여유", f"{YI_LIMIT - yi_pred:+.2f}",
          help="양수면 한계 이내, 음수면 한계 초과 상태입니다.",
          delta=f"{YI_LIMIT - yi_pred:+.2f}", delta_color="normal")
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# 배치 추론 (그래프용 데이터 일괄 계산)
# ════════════════════════════════════════════════════════════════════════════
N_SWEEP   = 30
all_rows  = []
sweep_idx = {}
idx       = 0
base      = [current[f] for f in FEATURE_NAMES]

# 민감도 분석용: 변수 하나씩 범위 전체를 스위프
for i, feat in enumerate(FEATURE_NAMES):
    sweep = np.linspace(*SIM_RANGES[feat], N_SWEEP)
    sweep_idx[feat] = (idx, sweep, i)
    for val in sweep:
        row = base.copy(); row[i] = val
        all_rows.append(row)
    idx += N_SWEEP

# 영향도 분석용: 각 변수의 최솟값/최댓값에서 YI 계산
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
# 그래프 행 1: YI 게이지 / 처방 포지션 / 변수 영향도
# ════════════════════════════════════════════════════════════════════════════
col_g, col_p, col_i = st.columns([1, 1.4, 1.2])
labels_full = [FEATURE_META[f][1] for f in FEATURE_NAMES]

# ─ YI 게이지 바 ──────────────────────────────────────────────────────────
with col_g:
    st.markdown("**예측 YI 게이지**")
    st.caption("현재 처방의 YI 예측값과 허용 한계의 관계를 보여줍니다.")
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

# ─ 처방 포지션 ───────────────────────────────────────────────────────────
with col_p:
    st.markdown("**처방 포지션**")
    st.caption("각 변수가 허용 범위 내 어느 위치에 있는지 보여줍니다. 빨강=상한 근접, 초록=하한 근접.")
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

# ─ 변수별 YI 영향도 ──────────────────────────────────────────────────────
with col_i:
    st.markdown("**변수별 YI 영향도**")
    st.caption("각 변수를 최솟값→최댓값으로 바꿨을 때 YI 변화량입니다. 빨강=YI 증가, 초록=YI 감소.")
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
# 그래프 행 2: 민감도 분석 곡선
# ════════════════════════════════════════════════════════════════════════════
st.subheader("민감도 분석 — 변수별 YI 변화 곡선")
st.caption("각 변수를 범위 전체로 변화시켰을 때 YI가 어떻게 달라지는지 보여줍니다. 빨간 점선=현재값, 회색 점선=허용 한계.")

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

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ① 개선 우선순위
# ════════════════════════════════════════════════════════════════════════════
st.subheader("① 개선 우선순위 — 어떤 변수를 먼저 바꿔야 하나")
st.caption(
    "현재 처방 기준으로 각 변수를 최적 방향으로 한 번씩 바꿨을 때 "
    "YI 개선 효과가 큰 순서로 정렬됩니다."
)

priority_rows = []
for i, feat in enumerate(FEATURE_NAMES):
    lo, hi      = SIM_RANGES[feat]
    yi_lo       = float(y_impact[i * 2])
    yi_hi       = float(y_impact[i * 2 + 1])
    best_yi     = min(yi_lo, yi_hi)
    best_val    = lo if yi_lo < yi_hi else hi
    direction   = "최솟값으로 낮추기" if yi_lo < yi_hi else "최댓값으로 높이기"
    improvement = yi_pred - best_yi
    priority_rows.append({
        "변수":         FEATURE_META[feat][1],
        "현재값":       round(current[feat], 3),
        "권장값":       round(best_val, 3),
        "조정 방향":    direction,
        "변경 후 YI":   round(best_yi, 2),
        "YI 개선량":    round(improvement, 2),
    })

priority_df = pd.DataFrame(priority_rows).sort_values("YI 개선량", ascending=False)
priority_df.index = range(1, len(priority_df) + 1)

def highlight_improvement(row):
    color = "#d4edda" if row["YI 개선량"] > 3 else \
            "#fff3cd" if row["YI 개선량"] > 0 else "#f8d7da"
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    priority_df.style.apply(highlight_improvement, axis=1)
                     .format({"YI 개선량": "{:+.2f}", "변경 후 YI": "{:.2f}"}),
    use_container_width=True,
)
st.caption("초록=효과 큼 (3 이상) / 노랑=효과 보통 / 빨강=효과 없음 또는 역효과")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ② 최적 처방 자동 추천
# ════════════════════════════════════════════════════════════════════════════
st.subheader("② 최적 처방 자동 추천")
st.caption(
    "처방 공간을 무작위로 500건 탐색하여 YI를 가장 낮추는 처방 조합을 찾습니다. "
    "현재 처방과의 차이를 함께 표시합니다."
)

with st.spinner("최적 처방 탐색 중 …"):
    np.random.seed(0)
    N_SEARCH = 500
    search_rows = {
        feat: np.random.uniform(*SIM_RANGES[feat], N_SEARCH)
        for feat in FEATURE_NAMES
    }
    X_search = pd.DataFrame(search_rows)
    y_search = model.predict(X_search)

    best_idx = int(np.argmin(y_search))
    best_row = X_search.iloc[best_idx]
    best_yi  = float(y_search[best_idx])

    opt_rows = []
    for feat in FEATURE_NAMES:
        cur_val = current[feat]
        opt_val = round(float(best_row[feat]), 3)
        delta   = opt_val - cur_val
        opt_rows.append({
            "변수":     FEATURE_META[feat][1],
            "현재값":   round(cur_val, 3),
            "권장값":   opt_val,
            "변화량":   f"{delta:+.3g}",
            "조정 여부": "→ 조정 필요" if abs(delta) > FEATURE_META[feat][5] else "유지",
        })

    opt_df = pd.DataFrame(opt_rows)

col_opt1, col_opt2 = st.columns([2, 1])

with col_opt1:
    def highlight_change(row):
        color = "#d4edda" if row["조정 여부"] == "→ 조정 필요" else "#f8f9fa"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        opt_df.style.apply(highlight_change, axis=1),
        use_container_width=True,
    )

with col_opt2:
    st.metric("현재 처방 YI",  f"{yi_pred:.2f}")
    st.metric("최적 처방 YI",  f"{best_yi:.2f}",
              delta=f"{best_yi - yi_pred:+.2f}", delta_color="inverse")
    if best_yi < YI_LIMIT:
        st.success(f"✅ 최적 처방으로 YI {best_yi:.1f} 달성 가능 — 허용 한계 이내입니다.")
    else:
        st.warning(f"⚠️ 최적 처방에서도 YI {best_yi:.1f} — 허용 한계를 초과합니다.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ③ 목표 YI 달성 가능 범위
# ════════════════════════════════════════════════════════════════════════════
st.subheader("③ 목표 YI 달성 가능 범위")
st.caption(
    "목표 YI를 설정하면 그 값 이하를 달성할 수 있는 각 변수의 범위를 계산합니다. "
    "단, 다른 변수는 현재값으로 고정한 상태에서 해당 변수 단독의 기여를 분석합니다."
)

target_yi = st.slider(
    "목표 YI 설정",
    min_value=1.0,
    max_value=float(YI_LIMIT),
    value=min(15.0, float(YI_LIMIT)),
    step=0.5,
    help="이 값 이하가 되는 처방 범위를 변수별로 계산합니다."
)

with st.spinner("달성 가능 범위 계산 중 …"):
    feasible_rows = []
    for i, feat in enumerate(FEATURE_NAMES):
        start, sweep, fi = sweep_idx[feat]
        yi_vals = y_sweep[start:start + N_SWEEP]
        feasible_vals = sweep[yi_vals <= target_yi]

        if len(feasible_vals) == 0:
            feasible_rows.append({
                "변수":         FEATURE_META[feat][1],
                "달성 가능 범위": "이 변수 단독으로는 달성 불가",
                "최솟값":       "-",
                "최댓값":       "-",
                "현재값":       round(current[feat], 3),
                "현재값 상태":  "❌ 달성 불가",
            })
        else:
            fmin   = round(float(feasible_vals.min()), 3)
            fmax   = round(float(feasible_vals.max()), 3)
            within = fmin <= current[feat] <= fmax
            feasible_rows.append({
                "변수":         FEATURE_META[feat][1],
                "달성 가능 범위": f"{fmin}  ~  {fmax}",
                "최솟값":       fmin,
                "최댓값":       fmax,
                "현재값":       round(current[feat], 3),
                "현재값 상태":  "✅ 범위 내" if within else "⚠️ 조정 필요",
            })

    feasible_df = pd.DataFrame(feasible_rows)

    def highlight_feasible(row):
        if "✅" in row["현재값 상태"]:
            return ["background-color: #d4edda"] * len(row)
        elif "❌" in row["현재값 상태"]:
            return ["background-color: #f8d7da"] * len(row)
        else:
            return ["background-color: #fff3cd"] * len(row)

    st.dataframe(
        feasible_df[["변수", "달성 가능 범위", "현재값", "현재값 상태"]]
        .style.apply(highlight_feasible, axis=1),
        use_container_width=True,
    )
    st.caption("✅ 현재값이 이미 목표 범위 내  /  ⚠️ 조정 필요  /  ❌ 해당 변수 단독으로는 달성 불가")

# ════════════════════════════════════════════════════════════════════════════
# 하단 — 실험 데이터 업로드 형식 안내
# ════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📋 실험 데이터 업로드 형식 안내"):
    st.markdown("아래 컬럼명을 포함한 Excel 또는 CSV 파일을 업로드하세요.")
    st.code(" | ".join(FEATURE_NAMES + ["target_YI"]), language=None)
    st.markdown(
        "업로드 시 가상 데이터와 실험 데이터를 혼합해 재학습합니다 (실험 데이터 3배 가중). "
        "5건 이상이면 유의미한 영향, 20건 이상이면 가상 데이터 없이 단독 학습을 권장합니다."
    )
