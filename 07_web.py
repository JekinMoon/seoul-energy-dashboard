import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import joblib
import json
import datetime
import streamlit.components.v1 as components
from sklearn.metrics import r2_score

# 1. 페이지 기본 설정
st.set_page_config(layout="wide", page_title="서울시 전력 수요 관제 시스템", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, '..', '6_models', filename),
        os.path.join(BASE_DIR, '..', '1_data', 'processed', filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

model = joblib.load(find_file('xgb_power_model_v2.pkl'))
with open(find_file('features_v2.json'), 'r', encoding='utf-8') as f:
        features = json.load(f)

# ==========================================
# 2. CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

    footer { visibility: hidden !important; }
    html, body, [class*="css"], p, div, span { font-size: 18px !important; }

    .block-container {
        padding-top: 3rem !important; padding-bottom: 1rem;
        padding-left: 1rem !important; padding-right: 2rem !important;
    }
    [data-testid="collapsedControl"] { z-index: 999 !important; position: fixed !important; }

    .section-label {
        font-size: 0.85rem !important; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #7a8499; margin: 10px 0px 15px 0px;
        display: flex; align-items: center; gap: 10px;
    }
    .section-label::after { content:''; flex:1; height:1px; background:#E2E8F0; }

    .kpi-row { display: flex; align-items: stretch; gap: 1rem; margin-bottom: 0; }

    .kpi-main {
        background: #1B3A6B; border-radius: 12px; padding: 20px 22px; color: white;
        min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;
        box-sizing: border-box; flex: 1;
    }
    .kpi-main-label { font-size: 0.9rem !important; opacity: 0.8; }
    .kpi-main-value { font-size: 2.8rem !important; font-weight: 900; font-family: 'DM Mono', monospace; line-height: 1; }
    .kpi-badge {
        display: inline-flex; align-items: center; gap: 6px; border-radius: 20px; padding: 6px 14px;
        font-size: 0.85rem !important; font-weight: 700; width: fit-content;
    }
    .kpi-card {
        background: white; border-radius: 12px; padding: 20px 20px; border: 1px solid #E2E8F0;
        min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;
        box-sizing: border-box; flex: 1; height: 100%;
    }
    .kpi-card-label { font-size: 0.85rem !important; color: #7a8499; font-weight: 700; }
    .kpi-card-value { font-size: 2rem !important; font-weight: 900; font-family: 'DM Mono', monospace; line-height: 1.1; }
    .kpi-card-badge {
        display: inline-flex; align-items: center; border-radius: 20px; padding: 5px 12px;
        font-size: 0.8rem !important; font-weight: 700; width: fit-content;
    }
    .badge-red    { background:#FDEDEC; color:#C0392B; }
    .badge-green  { background:#EAFAF1; color:#27AE60; }
    .badge-orange { background:#FEF9E7; color:#E67E22; }
    .badge-blue   { background:#EBF3FB; color:#1B3A6B; }

    .mid-card-title { font-size: 1.15rem !important; color: #1B3A6B; font-weight: 800; margin-bottom: 10px; }

    /* 2탭 설비 카드 높이 통일 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
    }
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 430px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    .weather-kpi-card {
        border-radius: 10px; padding: 12px 8px; text-align: center; border: 1.5px solid; flex: 1;
        display: flex; flex-direction: column; justify-content: center;
    }
    .weather-kpi-label { font-size: 0.9rem !important; font-weight: 700; margin-bottom: 4px; }
    .weather-kpi-value { font-size: 2.1rem !important; font-weight: 900; font-family: 'DM Mono', monospace; line-height: 1; }

    .msg-box {
        border-radius: 8px; padding: 14px 16px; font-size: 1.05rem !important;
        font-weight: 700; text-align: center; margin-top: 15px;
    }

    .insight-bar {
        background: #1B3A6B; border-radius: 12px; padding: 24px 32px; margin: 0 0 24px 0;
        display: grid; grid-template-columns: 1fr 1px 1fr 1px 1fr; align-items: start;
    }
    .insight-item { padding: 0 24px; display: flex; align-items: flex-start; gap: 14px; }
    .insight-divider { background: rgba(255,255,255,0.15); align-self: stretch; }
    .insight-icon {
        width: 42px; height: 42px; min-width: 42px; background: rgba(255,255,255,0.1);
        border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    }
    .insight-title { font-size: 1rem !important; font-weight: 700; color: white; margin-bottom: 6px; }
    .insight-desc  { font-size: 0.9rem !important; color: rgba(255,255,255,0.7); line-height: 1.6; }

    .scenario-card {
        background: white; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0;
        display: flex; flex-direction: column; gap: 8px;
    }
    .scenario-card-title { font-size: 1rem !important; font-weight: 800; color: #1B3A6B; }
    .scenario-card-value { font-size: 1.8rem !important; font-weight: 900; font-family: 'DM Mono', monospace; }
    .scenario-card-sub   { font-size: 0.85rem !important; color: #7a8499; }

    .fact-badge {
        display: inline-block; background: #EBF3FB; color: #1B3A6B;
        border-radius: 6px; padding: 3px 10px; font-size: 0.78rem !important;
        font-weight: 700; margin-top: 4px;
    }

    @media (max-width: 768px) {
        .insight-bar { grid-template-columns: 1fr !important; padding: 16px !important; }
        .insight-divider { display: none !important; }
        .insight-item { padding: 12px 0 !important; border-bottom: 1px solid rgba(255,255,255,0.15); }
        .insight-item:last-child { border-bottom: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 데이터 로드 & 통계 자동 계산
# ==========================================
@st.cache_data
def load_all_data():
    # [수정] 하드코딩된 절대 경로 제거
    df     = pd.read_csv(find_file('df_refined_v2.csv'))
    df_res = pd.read_csv(find_file('df_predicted_v2.csv'))
    df['datetime']     = pd.to_datetime(df['datetime'])
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    return df, df_res

@st.cache_data
def calc_scenario_stats(df):
    """데이터에서 직접 시나리오 수치 산출"""
    weekday_avg = df[df['is_holiday'] == 0]['전력사용량(MWh)'].mean()
    holiday_avg = df[df['is_holiday'] == 1]['전력사용량(MWh)'].mean()
    dr_effect   = weekday_avg - holiday_avg

    peak_avg    = df[df['hour'].between(14, 17)]['전력사용량(MWh)'].mean()
    offpeak_avg = df[df['hour'].between(2,  5)]['전력사용량(MWh)'].mean()
    ess_max     = peak_avg - offpeak_avg

    temp_slopes = {}
    bins   = [(-20, 5), (5, 18), (18, 24), (24, 30), (30, 45)]
    labels = ['혹한', '냉온', '쾌적', '더위', '폭염']
    for (lo, hi), label in zip(bins, labels):
        seg = df[(df['기온(°C)'] >= lo) & (df['기온(°C)'] < hi)]
        if len(seg) > 50:
            slope = np.polyfit(seg['기온(°C)'], seg['전력사용량(MWh)'], 1)[0]
            temp_slopes[label] = round(slope, 1)

    sat_avg  = df[df['dayofweek'] == 5]['전력사용량(MWh)'].mean()
    sun_avg  = df[df['dayofweek'] == 6]['전력사용량(MWh)'].mean()
    weekend_effect = weekday_avg - ((sat_avg + sun_avg) / 2)

    return {
        'dr_effect':      round(dr_effect, 1),
        'ess_max':        round(ess_max, 0),
        'temp_slopes':    temp_slopes,
        'weekday_avg':    round(weekday_avg, 1),
        'holiday_avg':    round(holiday_avg, 1),
        'peak_avg':       round(peak_avg, 1),
        'offpeak_avg':    round(offpeak_avg, 1),
        'weekend_effect': round(weekend_effect, 1),
    }

# --- 덕커브(Duck Curve) 반영: 시간대별 동적 탄소 배출 계수 함수 ---
def get_dynamic_carbon_factor(hour):
    if 11 <= hour <= 15:
        return 0.3500, "🟢 저탄소 (태양광 주도)"
    elif 16 <= hour <= 20:
        return 0.5200, "🔴 고탄소 (화력 기동)"
    else:
        return 0.4594, "🟡 일반 (평균 배출)"

df, df_res = load_all_data()
stats = calc_scenario_stats(df)

df_2024   = df_res[df_res['datetime'].dt.year == 2024]
calc_r2   = r2_score(df_2024['전력사용량(MWh)'], df_2024['예측값(MWh)'])
calc_mape = np.mean(np.abs((df_2024['전력사용량(MWh)'] - df_2024['예측값(MWh)']) / df_2024['전력사용량(MWh)'])) * 100
calc_rmse = np.sqrt(np.mean((df_2024['전력사용량(MWh)'] - df_2024['예측값(MWh)'])**2))

# ==========================================
# 4. 사이드바 (에러 완벽 해결 버전)
# ==========================================

# 1. 초기 세션 상태 설정 (최초 실행 시 한 번만)
if 's_temp' not in st.session_state: st.session_state.s_temp = 25.0
if 's_hum' not in st.session_state: st.session_state.s_hum = 50
if 's_wind' not in st.session_state: st.session_state.s_wind = 2.0
if 's_hour' not in st.session_state: st.session_state.s_hour = 14

# 2. 동기화 콜백 함수
def sync_val(src_key, dst_key):
    st.session_state[dst_key] = st.session_state[src_key]

st.sidebar.markdown("### 📋 관제 메뉴")
page = st.sidebar.radio(
    "메뉴를 선택하세요",
    ["⚡ 전력 수요 시뮬레이션", "🧠 운영 시나리오 분석", "🔍 AI 예측 모델 분석", "📈 에너지 전환 전망"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.header("🌡️ 기상 시나리오 설정")

# --- 기온 (Temp) ---
st.sidebar.write("예상 기온 (°C)")
c1, c2 = st.sidebar.columns([7, 3])
with c1:
    # 핵심: value 파라미터 대신 세션 스테이트를 직접 업데이트 후 위젯 호출
    st.session_state['t_slider_key'] = st.session_state.s_temp
    st.slider("t_slider", -10.0, 40.0, key="t_slider_key", 
              on_change=sync_val, args=("t_slider_key", "s_temp"), 
              label_visibility="collapsed")
with c2:
    st.number_input("t_input", -10.0, 40.0, key="s_temp", 
                    on_change=sync_val, args=("s_temp", "t_slider_key"),
                    step=0.5, label_visibility="collapsed")

# --- 습도 (Hum) ---
st.sidebar.write("예상 습도 (%)")
c1, c2 = st.sidebar.columns([7, 3])
with c1:
    st.session_state['h_slider_key'] = st.session_state.s_hum
    st.slider("h_slider", 0, 100, key="h_slider_key", 
              on_change=sync_val, args=("h_slider_key", "s_hum"), 
              label_visibility="collapsed")
with c2:
    st.number_input("h_input", 0, 100, key="s_hum", 
                    on_change=sync_val, args=("s_hum", "h_slider_key"),
                    step=5, label_visibility="collapsed")

# --- 풍속 (Wind) ---
st.sidebar.write("예상 풍속 (m/s)")
c1, c2 = st.sidebar.columns([7, 3])
with c1:
    st.session_state['w_slider_key'] = st.session_state.s_wind
    st.slider("w_slider", 0.0, 15.0, key="w_slider_key", 
              on_change=sync_val, args=("w_slider_key", "s_wind"), 
              label_visibility="collapsed")
with c2:
    st.number_input("w_input", 0.0, 15.0, key="s_wind", 
                    on_change=sync_val, args=("s_wind", "w_slider_key"),
                    step=0.1, label_visibility="collapsed")

# --- 시간 (Hour) ---
st.sidebar.write("예상 시간 (시)")
c1, c2 = st.sidebar.columns([7, 3])
with c1:
    st.session_state['hr_slider_key'] = st.session_state.s_hour
    st.slider("hr_slider", 0, 23, key="hr_slider_key", 
              on_change=sync_val, args=("hr_slider_key", "s_hour"), 
              label_visibility="collapsed")
with c2:
    st.number_input("hr_input", 0, 23, key="s_hour", 
                    on_change=sync_val, args=("s_hour", "hr_slider_key"),
                    step=1, label_visibility="collapsed")

# 최종 변수 할당 (이후 예측 로직에서 사용)
s_temp = st.session_state.s_temp
s_hum = st.session_state.s_hum
s_wind = st.session_state.s_wind
s_hour = st.session_state.s_hour

now          = datetime.datetime.now()
s_month      = now.month
s_sensory    = 13.12 + 0.6215*s_temp - 11.37*(s_wind**0.16) + 0.3965*s_temp*(s_wind**0.16)
s_discomfort = 0.81*s_temp + 0.01*s_hum*(0.99*s_temp - 14.3) + 46.3

# 예측 로직
similar_date  = "-"
predicted_mwh = 5500.0
if not df.empty:
    df['diff'] = (
        np.abs(df['기온(°C)'] - s_temp) +
        np.abs(df['hour']    - s_hour)  * 2 +
        np.abs(df['month']   - s_month) * 0.5
    )
    target_idx   = df['diff'].idxmin()
    input_row    = df.loc[target_idx].copy()
    similar_date = input_row['datetime'].strftime('%Y-%m-%d %H시')

    input_row['hour']         = s_hour
    input_row['month']        = s_month
    # [수정] 관제 시뮬레이션의 기준(베이스라인)을 평일(수요일=2)로 고정하여 주말 효과 이중 차감 방지
    input_row['dayofweek']    = 2 
    input_row['is_holiday']   = 0
    input_row['is_peak_hour'] = 1 if 14 <= s_hour <= 17 else 0
    input_row['기온(°C)']     = s_temp
    input_row['습도(%)']      = s_hum
    input_row['강수량(mm)']   = 0.0
    input_row['CDD']          = max(0, s_temp - 24)
    input_row['HDD']          = max(0, 18 - s_temp)
    input_row['is_heatwave']  = 1 if s_temp >= 33 else 0
    input_row['temp_3h_mean'] = s_temp
    input_row['temp_6h_max']  = max(s_temp, input_row.get('temp_6h_max', s_temp))
    try:
        predicted_mwh = float(model.predict(pd.DataFrame([input_row[features]]))[0])
    except Exception as e:
        st.error(f"예측 오류: {e}")

avg_mwh    = df['전력사용량(MWh)'].mean()
delta_pct  = (predicted_mwh - avg_mwh) / avg_mwh * 100

# 동적 탄소 계수 적용
current_carbon_factor, carbon_status_msg = get_dynamic_carbon_factor(s_hour)
carbon_ton = predicted_mwh * current_carbon_factor

yoy_delta = None
same_cond = df[(df['hour'] == s_hour) & (df['month'] == s_month)]
if len(same_cond) >= 2:
    a23 = same_cond[same_cond['datetime'].dt.year == 2023]['전력사용량(MWh)'].mean()
    a24 = same_cond[same_cond['datetime'].dt.year == 2024]['전력사용량(MWh)'].mean()
    if a23 > 0 and not np.isnan(a24):
        yoy_delta = ((a24 - a23) / a23) * 100

# 경보 로직: 전력 수요 증가율(delta_pct)과 극단적 기온 모두 반영
if delta_pct >= 15 or s_temp >= 33 or s_temp <= -5:
    alert_status, alert_color, alert_emoji = "비상", "#C0392B", "🔴"
elif delta_pct >= 5 or s_temp >= 30 or s_temp <= 0:
    alert_status, alert_color, alert_emoji = "경계", "#E67E22", "🟡"
else:
    alert_status, alert_color, alert_emoji = "정상", "#27AE60", "🟢"

is_peak = 14 <= s_hour <= 17

# ==========================================
# 5. 상단 헤더
# ==========================================
# 헤더 왼쪽 부분 (기존 st.markdown으로 유지)
st.markdown("""
<div style="background:#1B3A6B;border-radius:12px;padding:20px 24px;margin-bottom:0px;
            box-shadow:0 4px 6px rgba(0,0,0,0.1);display:flex;justify-content:space-between;
            align-items:flex-start;flex-wrap:wrap;gap:8px;">
  <div>
    <div style="font-size:1.3rem;font-weight:800;color:white;font-family:sans-serif;">
      🔌 서울시 전력 수요 관제 시스템
    </div>
    <div style="font-size:0.85rem;opacity:0.7;color:white;margin-top:4px;font-family:'DM Mono',monospace;">
      Seoul Energy Demand Simulation Dashboard
    </div>
  </div>
  <div id="live-clock" style="color:rgba(255,255,255,0.8);font-family:'DM Mono',monospace;font-size:0.85rem;">
    🕐 로딩 중...
  </div>
</div>
""", unsafe_allow_html=True)

# 시계만 components.html로 따로 실행
components.html("""
<script>
  function updateClock() {
    const now = new Date();
    const formatted = now.getFullYear() + '-' +
      String(now.getMonth()+1).padStart(2,'0') + '-' +
      String(now.getDate()).padStart(2,'0') + ' ' +
      String(now.getHours()).padStart(2,'0') + ':' +
      String(now.getMinutes()).padStart(2,'0');

    // 부모 프레임의 live-clock 요소에 접근
    const el = window.parent.document.getElementById('live-clock');
    if (el) el.innerText = '🕐 ' + formatted;
  }
  updateClock();
  setInterval(updateClock, 60000);
</script>
""", height=0)

# ==========================================
# 페이지 1: 전력 수요 시뮬레이션
# ==========================================
if page == "⚡ 전력 수요 시뮬레이션":
    st.markdown('<div class="section-label">① 핵심 KPI</div>', unsafe_allow_html=True)

    delta_badge  = "badge-red" if delta_pct > 10 else ("badge-orange" if delta_pct > 0 else "badge-green")
    delta_msg    = "⬆ 수요 급증" if delta_pct > 10 else ("⬆ 수요 증가" if delta_pct > 0 else "⬇ 수요 감소")
    peak_badge   = "badge-red" if is_peak else "badge-green"
    peak_txt     = "🔴 피크 시간대" if is_peak else "🟢 비피크"
    carbon_badge = "badge-red" if carbon_ton > 2500 else ("badge-orange" if carbon_ton > 2000 else "badge-green")
    carbon_msg   = carbon_status_msg  
    yoy_badge    = "badge-red" if (yoy_delta or 0) > 0 else "badge-green"
    yoy_val      = f"{yoy_delta:+.1f}%" if yoy_delta else "-"
    yoy_msg      = f"{'⬆' if (yoy_delta or 0)>0 else '⬇'} 전년 동기 대비" if yoy_delta else "데이터 없음"

    with st.container():
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
        with c1:
            st.markdown(f"""
            <div class="kpi-main">
                <div class="kpi-main-label">⚡ 예측 전력 수요</div>
                <div><div class="kpi-main-value">{predicted_mwh:,.0f} <span style="font-size:1.2rem">MWh</span></div></div>
                <div class="kpi-badge" style="background:rgba(255,255,255,0.15);color:white;">{alert_emoji} {alert_status} — 평균 대비 {delta_pct:+.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">📈 평균 대비 변화율</div>
                <div class="kpi-card-value" style="color:{'#C0392B' if delta_pct>0 else '#27AE60'};">{delta_pct:+.1f}%</div>
                <div class="kpi-card-badge {delta_badge}">{delta_msg}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">⏰ 피크 가능성</div>
                <div class="kpi-card-value" style="color:{'#C0392B' if is_peak else '#27AE60'};">{'HIGH' if is_peak else 'LOW'}</div>
                <div class="kpi-card-badge {peak_badge}">{peak_txt}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">🌿 탄소 배출 추정</div>
                <div class="kpi-card-value" style="white-space:nowrap; color:#111111;">{carbon_ton:,.0f}<span style="font-size:1.2rem;">tCO₂</span></div>
                <div class="kpi-card-badge {carbon_badge}">{carbon_msg}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">📅 전년 동기 대비</div>
                <div class="kpi-card-value" style="color:{'#C0392B' if (yoy_delta or 0)>0 else '#27AE60'};">{yoy_val}</div>
                <div class="kpi-card-badge {yoy_badge}">{yoy_msg}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">② 세부 KPI</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">🕐 시간대별 전력 수요</div>', unsafe_allow_html=True)
            hour_avg = df.groupby('hour')['전력사용량(MWh)'].mean().reset_index()
            colors   = ['#C0392B' if 14 <= h <= 17 else '#1B3A6B' for h in hour_avg['hour']]
            fig_bar  = go.Figure(go.Bar(x=hour_avg['hour'], y=hour_avg['전력사용량(MWh)'], marker_color=colors))
            fig_bar.add_vline(x=s_hour, line_dash="dash", line_color="orange", line_width=2)
            fig_bar.update_layout(height=290, margin=dict(t=4,b=4,l=4,r=4), xaxis_title="시간", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            msg_color = "#C0392B" if is_peak else "#27AE60"
            msg_bg    = "#FDEDEC" if is_peak else "#EAFAF1"
            msg_txt   = "🔴 14~17시 피크 구간 진입" if is_peak else "🟢 비피크 구간 — 안정적 수요"
            st.markdown(f"<div class='msg-box' style='background:{msg_bg};color:{msg_color};'>{msg_txt}</div>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            r, g, b = int(alert_color[1:3],16), int(alert_color[3:5],16), int(alert_color[5:7],16)
            card_bg  = f"rgba({r},{g},{b},0.08)"
            st.markdown('<div class="mid-card-title">🌡️ 기온 영향 — 체감 & 불쾌지수</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=s_temp,
                number={'suffix': "°C", 'font': {'size': 36, 'color': alert_color}},
                gauge={
                    'axis': {'range': [-15,45], 'tickvals': [-15,0,15,30,45], 'tickfont': {'size':12,'color':'#ADB5BD'}},
                    'bar': {'color': alert_color, 'thickness': 0.3},
                    'steps': [
                        {'range': [-15,5],  'color': "rgba(100,181,246,0.2)"},
                        {'range': [5,18],   'color': "rgba(233,236,239,0.2)"},
                        {'range': [18,26],  'color': "rgba(129,199,132,0.2)"},
                        {'range': [26,33],  'color': "rgba(255,213,79,0.2)"},
                        {'range': [33,45],  'color': "rgba(229,115,115,0.2)"},
                    ],
                    'threshold': {'line': {'color': alert_color, 'width': 3}, 'thickness': 0.8, 'value': s_temp}
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(t=10,b=0,l=30,r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # [수정] 여름/겨울/봄가을 계절별 맞춤형 체감 메시지 적용
            if s_temp >= 26:
                if s_discomfort >= 80:
                    di_msg = "😰 매우 불쾌 — 냉방 부하 폭증"
                elif s_discomfort >= 75:
                    di_msg = "😓 불쾌 — 냉방 부하 증가"
                else:
                    di_msg = "🙂 보통 — 여름철 안정 구간"
            elif s_temp <= 10:
                if s_sensory <= -10:
                    di_msg = "🥶 한파 위험 — 난방 부하 폭증"
                elif s_sensory <= -5:
                    di_msg = "🧣 한파 주의 — 난방 부하 증가"
                else:
                    di_msg = "🌬️ 쌀쌀 — 난방 가동 시작"
            else:
                di_msg = "😊 쾌적 — 냉난방 부하 최소 구간"
                
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin:10px 0;height:100px;">
                <div class="weather-kpi-card" style="background:{card_bg};border-color:{alert_color};color:{alert_color};">
                    <div class="weather-kpi-label">체감온도</div>
                    <div class="weather-kpi-value">{s_sensory:.1f}</div>
                </div>
                <div class="weather-kpi-card" style="background:{card_bg};border-color:{alert_color};color:{alert_color};">
                    <div class="weather-kpi-label">불쾌지수</div>
                    <div class="weather-kpi-value">{s_discomfort:.1f}</div>
                </div>
            </div>
            <div class='msg-box' style='background:{card_bg};color:{alert_color};border:1px solid {alert_color}33;'>{di_msg}</div>
            """, unsafe_allow_html=True)

    with col3:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">🕵️ 유사 패턴 탐색</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#EBF3FB;border-radius:10px;padding:14px 16px;margin-bottom:10px;text-align:center;height:80px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-size:0.9rem;color:#7a8499;font-weight:700;">가장 유사한 과거 사례</div>
                <div style="font-size:1.4rem;font-weight:900;color:#1B3A6B;margin-top:4px;">📅 {similar_date}</div>
            </div>
            """, unsafe_allow_html=True)
            sample = df.sample(min(500, len(df)))
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=sample['기온(°C)'], y=sample['전력사용량(MWh)'],
                mode='markers', marker=dict(color='#AED6F1', opacity=0.5, size=5)))
            fig_sc.add_trace(go.Scatter(x=[s_temp], y=[predicted_mwh],
                mode='markers+text', text=[f"{predicted_mwh:,.0f}"], textposition="top center",
                marker=dict(color='#C0392B', size=16, symbol='star'),
                textfont=dict(size=14, color='#C0392B', weight='bold')))
            fig_sc.update_layout(height=200, margin=dict(t=4,b=4,l=4,r=4), showlegend=False)
            st.plotly_chart(fig_sc, use_container_width=True)
            msg_color = "#C0392B" if is_peak else "#27AE60"
            msg_bg    = "#FDEDEC" if is_peak else "#EAFAF1"
            peak_label = '🔴 피크 시간대 예측' if is_peak else '🟢 안정 구간 예측'
            st.markdown(f"<div class='msg-box' style='background:{msg_bg};color:{msg_color};'>{peak_label} — {predicted_mwh:,.0f} MWh</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">③ 인사이트 및 대응 전략</div>', unsafe_allow_html=True)

    peak_t, peak_d = ("⚡ 피크 시간대 수요 증가 예상", "14~17시 전력 수요 집중. 평균 대비 초과 구간 진입.") \
                      if is_peak else ("⚡ 안정적 수요 구간", "현재 시간대는 피크 외 구간으로 전력 수요가 안정적입니다.")
    cool_t, cool_d = ("❄️ 냉방 부하 증가 가능성", f"기온 {s_temp:.1f}°C — EHP 등 냉방설비 집중 가동 예상.") \
                      if s_temp >= 28 else \
                     ("🔥 난방 부하 증가 가능성", f"기온 {s_temp:.1f}°C — 전열기기 등 난방 수요 증가 예상.") \
                      if s_temp <= 5 else \
                     ("✅ 냉난방 부하 안정", f"기온 {s_temp:.1f}°C는 임계점 이내로 부하가 안정적입니다.")
    resp_t, resp_d = ("🔄 전력 수요 분산 운영 필요", "ESS 방전 대기, 대용량 설비 부하 순차 조정 권고.") \
                      if alert_status == "비상" else \
                     ("🔄 수요 분산 모니터링", "공공기관 냉난방기기 순차 운전 검토 권고.") \
                      if alert_status == "경계" else \
                     ("🔄 정상 운영 유지", "안정적 공급 가능. 상시 모니터링 유지.")

    st.markdown(f"""
    <div class="insight-bar">
        <div class="insight-item">
            <div class="insight-icon">⚡</div>
            <div><div class="insight-title">{peak_t}</div><div class="insight-desc">{peak_d}</div></div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div class="insight-icon">❄️</div>
            <div><div class="insight-title">{cool_t}</div><div class="insight-desc">{cool_d}</div></div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div class="insight-icon">🔄</div>
            <div><div class="insight-title">{resp_t}</div><div class="insight-desc">{resp_d}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 페이지 2: 운영 시나리오 분석
# ==========================================
elif page == "🧠 운영 시나리오 분석":

    st.markdown('<div class="section-label">① 현재 예측 수요 기준</div>', unsafe_allow_html=True)

    # 예측값 요약 배너
    st.markdown(f"""
    <div style="background:#1B3A6B;border-radius:12px;padding:18px 28px;margin-bottom:8px;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
                font-family:'Pretendard','Noto Sans KR',sans-serif;width:100%;box-sizing:border-box;">
        <div>
            <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;font-weight:500;letter-spacing:0.05em;">
                현재 기상 기반 예측 수요
            </div>
            <div style="color:white;font-size:2rem;font-weight:800;margin-top:4px;letter-spacing:-0.02em;">
                {predicted_mwh:,.0f} <span style="font-size:1.1rem;font-weight:500;opacity:0.8;">MWh</span>
                <span style="font-size:1rem;opacity:0.65;margin-left:10px;">{alert_emoji} {alert_status}</span>
            </div>
        </div>
        <div style="display:flex;gap:28px;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="color:rgba(255,255,255,0.5);font-size:0.78rem;font-weight:500;">기온</div>
                <div style="color:white;font-size:1.2rem;font-weight:700;margin-top:2px;">{s_temp:.1f}°C</div>
            </div>
            <div style="text-align:center;">
                <div style="color:rgba(255,255,255,0.5);font-size:0.78rem;font-weight:500;">시간</div>
                <div style="color:white;font-size:1.2rem;font-weight:700;margin-top:2px;">{s_hour:02d}시</div>
            </div>
            <div style="text-align:center;">
                <div style="color:rgba(255,255,255,0.5);font-size:0.78rem;font-weight:500;">탄소 계수</div>
                <div style="color:white;font-size:1.2rem;font-weight:700;margin-top:2px;">{current_carbon_factor:.4f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">② 설비 운영 시나리오 조합</div>', unsafe_allow_html=True)
    st.caption("📌 아래 수치는 2023~2024 서울시 실측 데이터에서 산출된 설비 단위 제어 민감도입니다.")

    if s_temp >= 30:
        setpoint_effect = abs(stats['temp_slopes'].get('폭염', 307))
        temp_zone = "폭염 구간 (30°C↑)"
    elif s_temp >= 24:
        setpoint_effect = abs(stats['temp_slopes'].get('더위', 366))
        temp_zone = "더위 구간 (24~30°C)"
    elif s_temp >= 18:
        setpoint_effect = abs(stats['temp_slopes'].get('쾌적', 138))
        temp_zone = "쾌적 구간 (18~24°C) — 효과 미미"
    elif s_temp >= 5:
        setpoint_effect = abs(stats['temp_slopes'].get('냉온', 70))
        temp_zone = "냉온 구간 (5~18°C)"
    else:
        setpoint_effect = abs(stats['temp_slopes'].get('혹한', 16))
        temp_zone = "혹한 구간 (~5°C)"

    ess_auto = round(stats['ess_max'] * 0.2 / 50) * 50

    if 'ess_on'       not in st.session_state: st.session_state['ess_on']       = False
    if 'setpoint_adj' not in st.session_state: st.session_state['setpoint_adj'] = 0
    if 'weekend_on'   not in st.session_state: st.session_state['weekend_on']   = False

    _ess_on       = st.session_state['ess_on']
    _setpoint_adj = st.session_state['setpoint_adj']
    _weekend_on   = st.session_state['weekend_on']

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">🔋 ESS 방전</div>', unsafe_allow_html=True)

            fig_ess = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ess_auto if _ess_on else 0,
                number={"suffix": " MWh", "font": {"size": 28, "color": "#1B3A6B"}},
                gauge={
                    "axis": {"range": [0, int(stats["ess_max"] * 0.4)], "tickfont": {"size": 10, "color": "#ADB5BD"}},
                    "bar":  {"color": "#1B3A6B" if _ess_on else "#E2E8F0", "thickness": 0.3},
                    "steps": [
                        {"range": [0,               int(stats["ess_max"]*0.13)], "color": "rgba(27,58,107,0.08)"},
                        {"range": [int(stats["ess_max"]*0.13), int(stats["ess_max"]*0.4)],  "color": "rgba(27,58,107,0.04)"},
                    ],
                    "threshold": {"line": {"color": "#1B3A6B", "width": 3}, "thickness": 0.8, "value": ess_auto},
                }
            ))
            fig_ess.update_layout(height=200, margin=dict(t=10, b=0, l=20, r=20))
            st.plotly_chart(fig_ess, use_container_width=True)

            st.markdown(f'<div class="fact-badge">📊 피크 {stats["peak_avg"]:,.0f} − 심야 {stats["offpeak_avg"]:,.0f} MWh × 20%</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.toggle("투입", key="ess_on", help=f"자동 계산값 {ess_auto:,} MWh 투입")
            if _ess_on:
                st.markdown(f'<div class="msg-box" style="background:#EBF3FB;color:#1B3A6B;">✅ {ess_auto:,} MWh 감축 중</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="msg-box" style="background:#F8F9FA;color:#7a8499;">대기 중</div>', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">🌡️ 냉난방 Setpoint 제어</div>', unsafe_allow_html=True)

            zone_labels = ["혹한", "냉온", "쾌적", "더위", "폭염"]
            zone_values = [abs(stats["temp_slopes"].get(z, 0)) for z in zone_labels]
            zone_colors = ["#C0392B" if z == temp_zone.split(" ")[0] else "#AED6F1" for z in zone_labels]
            fig_sp = go.Figure(go.Bar(
                x=zone_values, y=zone_labels, orientation="h",
                marker_color=zone_colors,
                text=[f"{v:.0f}" for v in zone_values],
                textposition="outside",
            ))
            fig_sp.update_layout(
                height=182, margin=dict(t=4, b=4, l=10, r=40),
                xaxis=dict(visible=False), showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_sp, use_container_width=True)

            st.markdown(f'<div class="fact-badge">📊 {temp_zone} — 1°C당 {setpoint_effect:.0f} MWh</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            
            is_heating_season = s_temp <= 18
            btn_sign = "-" if is_heating_season else "＋"
            action_name = "난방" if is_heating_season else "냉방"
            
            sp1, sp2, sp3 = st.columns(3)
            with sp1:
                if st.button(f"{btn_sign}1°C", use_container_width=True):
                    st.session_state["setpoint_adj"] = 0 if _setpoint_adj == 1 else 1
                    st.rerun()
            with sp2:
                if st.button(f"{btn_sign}2°C", use_container_width=True):
                    st.session_state["setpoint_adj"] = 0 if _setpoint_adj == 2 else 2
                    st.rerun()
            with sp3:
                if st.button(f"{btn_sign}3°C", use_container_width=True):
                    st.session_state["setpoint_adj"] = 0 if _setpoint_adj == 3 else 3
                    st.rerun()
                    
            setpoint_reduction = _setpoint_adj * setpoint_effect

            if _setpoint_adj > 0:
                st.markdown(f'<div class="msg-box" style="background:#FDEDEC;color:#C0392B;">✅ {btn_sign}{_setpoint_adj}°C 제어 → {setpoint_reduction:,.0f} MWh 감축</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="msg-box" style="background:#F8F9FA;color:#7a8499;">버튼을 눌러 선택</div>', unsafe_allow_html=True)

    with c3:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">📅 주말·공휴일 효과</div>', unsafe_allow_html=True)

            fig_we = go.Figure(go.Bar(
                x=["평일", "공휴일"],
                y=[stats["weekday_avg"], stats["holiday_avg"]],
                marker_color=["#1B3A6B", "#27AE60" if _weekend_on else "#AED6F1"],
                text=[f'{stats["weekday_avg"]:,.0f}', f'{stats["holiday_avg"]:,.0f}'],
                textposition="outside",
            ))
            fig_we.update_layout(
                height=200, margin=dict(t=4, b=4, l=10, r=10),
                yaxis=dict(range=[0, stats["weekday_avg"] * 1.15], visible=False),
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_we, use_container_width=True)

            st.markdown(f'<div class="fact-badge">📊 평일 {stats["weekday_avg"]:,.0f} → 공휴일 {stats["holiday_avg"]:,.0f} MWh 실측</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.toggle("공휴일 패턴 적용", key="weekend_on", help=f"실측 감축 {stats['dr_effect']:,.0f} MWh")
            if _weekend_on:
                weekend_reduction = stats['dr_effect']
                st.markdown(f'<div class="msg-box" style="background:#EAFAF1;color:#27AE60;">✅ {weekend_reduction:,.0f} MWh 자연 감축</div>', unsafe_allow_html=True)
            else:
                weekend_reduction = 0
                st.markdown('<div class="msg-box" style="background:#F8F9FA;color:#7a8499;">평일 패턴 유지 중</div>', unsafe_allow_html=True)

    # ==========================================
    # 결과 계산
    # ==========================================
    ess_discharge   = ess_auto if _ess_on else 0
    total_reduction = ess_discharge + setpoint_reduction + weekend_reduction
    after_mwh       = max(0, predicted_mwh - total_reduction)
    after_carbon    = after_mwh * current_carbon_factor
    carbon_saved    = total_reduction * current_carbon_factor
    reduction_pct   = (total_reduction / predicted_mwh * 100) if predicted_mwh > 0 else 0

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">③ 탄소 및 계통운영 최적화 결과</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:16px;">
        <div class="kpi-card">
            <div class="kpi-card-label">⚡ 예측 수요 (대응 전)</div>
            <div class="kpi-card-value" style="color:#C0392B;">{predicted_mwh:,.0f}</div>
            <div class="kpi-card-badge badge-red">MWh</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-label">🔽 총 감축량</div>
            <div class="kpi-card-value" style="color:#E67E22;">-{total_reduction:,.0f}</div>
            <div class="kpi-card-badge badge-orange">MWh ({reduction_pct:.1f}%↓)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-label">✅ 대응 후 수요</div>
            <div class="kpi-card-value" style="color:#27AE60;">{after_mwh:,.0f}</div>
            <div class="kpi-card-badge badge-green">MWh</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-label">🌿 최적화 탄소 절감</div>
            <div class="kpi-card-value" style="color:#1B3A6B;">-{carbon_saved:,.0f}</div>
            <div class="kpi-card-badge badge-blue">tCO₂ ({carbon_status_msg.split(' ')[1]} 계수 적용)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2, gap="medium")

    with col_chart1:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">📊 설비별 감축 기여도</div>', unsafe_allow_html=True)
            items = {
                'ESS 방전':      ess_discharge,
                'Setpoint 조정': setpoint_reduction,
                '주말/공휴일':   weekend_reduction,
            }
            labels_chart = [k for k, v in items.items() if v > 0]
            values_chart = [v for v in items.values() if v > 0]
            if values_chart:
                fig_pie = go.Figure(go.Pie(
                    labels=labels_chart,
                    values=values_chart,
                    hole=0.5,
                    marker_colors=['#1B3A6B', '#C0392B', '#27AE60'],
                    textinfo='label+percent',
                ))
                fig_pie.update_layout(
                    height=300, margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=False,
                    annotations=[dict(text=f'-{total_reduction:,.0f}<br>MWh', x=0.5, y=0.5,
                                      font_size=14, showarrow=False, font_color='#1B3A6B')]
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("제어 패널에서 설비를 조작하면 감축 기여도가 표시됩니다.")

    with col_chart2:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">⚡ 운영 최적화 비교</div>', unsafe_allow_html=True)
            fig_bar2 = go.Figure()
            fig_bar2.add_trace(go.Bar(
                name='대응 전',
                x=['대응 전'],
                y=[predicted_mwh],
                marker_color='#C0392B',
                text=[f"{predicted_mwh:,.0f} MWh"],
                textposition='outside',
            ))
            fig_bar2.add_trace(go.Bar(
                name='대응 후',
                x=['대응 후'],
                y=[after_mwh],
                marker_color='#1B3A6B',
                text=[f"{after_mwh:,.0f} MWh"],
                textposition='outside',
            ))
            fig_bar2.update_layout(
                height=300, margin=dict(t=30,b=10,l=10,r=10),
                showlegend=False,
                yaxis=dict(range=[0, predicted_mwh * 1.2]),
                bargap=0.4,
            )
            st.plotly_chart(fig_bar2, use_container_width=True)

    # ---------------------------------------------------------
    # 현장 관제실 운영 지시서 (Dispatch Log) 포맷으로 인사이트 출력
    # ---------------------------------------------------------
    dispatch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 조치 내용 텍스트화
    action_texts = []
    if _setpoint_adj > 0: action_texts.append(f"{action_name} Setpoint 조절 ({btn_sign}{_setpoint_adj}°C) → 예상 감축: {setpoint_reduction:,.0f} MWh")
    if _ess_on: action_texts.append(f"ESS 방전 스케줄 투입 → 예상 감축: {ess_discharge:,.0f} MWh")
    if _weekend_on: action_texts.append(f"주말/공휴일 패턴 (자연 감축) 적용 → 예상 감축: {weekend_reduction:,.0f} MWh")
    
    actions_html = "<br>".join([f"&nbsp;&nbsp;&nbsp;&nbsp;{i+1}. {text}" for i, text in enumerate(action_texts)]) if action_texts else "&nbsp;&nbsp;&nbsp;&nbsp;- 시스템 관망 (추가 제어 불필요)"
    
    # 상황 판단
    if total_reduction == 0:
        situation = "정상. 현재 예측 수요 임계점 이내로 설비 개입 없이 베이스라인 유지 중입니다."
        status_color = "#7a8499"
    elif reduction_pct >= 10:
        situation = f"전력 피크 방어 및 예비력 확보 완료. (총 {reduction_pct:.1f}% 수요 감축)"
        status_color = "#27AE60"
    else:
        situation = f"감축률 {reduction_pct:.1f}% 진행 중. 피크 시간대 예비력 추가 확보 방안 검토 바랍니다."
        status_color = "#E67E22"

    st.markdown(f"""
    <div style="background:#111827; color:#E5E7EB; padding: 20px 24px; border-radius: 12px; font-family: 'DM Mono', monospace; font-size: 0.95rem; margin-top: 15px; border-left: 5px solid {status_color};">
        <div style="color:#60A5FA; font-weight:700; font-size: 1.1rem; margin-bottom: 12px;">
            [중앙 관제실 운영 지시서 / Dispatch Log]
        </div>
        <div style="margin-bottom: 6px;"><b>발행 일시:</b> {dispatch_time} (시뮬레이션 타겟: {s_hour:02d}시)</div>
        <div style="margin-bottom: 6px;"><b>계통 상황:</b> {situation}</div>
        <div style="margin-bottom: 6px;"><b>탄소 지수:</b> {current_carbon_factor:.4f} {carbon_status_msg}</div>
        <div style="margin-bottom: 4px;"><b>설비 제어 명령:</b></div>
        <div>{actions_html}</div>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px dashed #374151; color:#9CA3AF;">
            최종 {after_mwh:,.0f} MWh 도달 목표. 탄소 배출량 {carbon_saved:,.0f} tCO₂ 회피 예정.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 페이지 3: AI 예측 모델 분석
# ==========================================
elif page == "🔍 AI 예측 모델 분석":
    df_res['error_rate'] = np.abs((df_res['전력사용량(MWh)'] - df_res['예측값(MWh)']) / df_res['전력사용량(MWh)'] * 100)

    st.markdown('<div class="section-label">① 핵심 성능 지표</div>', unsafe_allow_html=True)
    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">📐 결정계수</div>
                <div class="kpi-card-value" style="color:#111111;">{calc_r2:.4f}</div>
                <div class="kpi-card-badge badge-blue">분산의 {calc_r2*100:.2f}% 설명</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">🎯 평균 오차율</div>
                <div class="kpi-card-value" style="color:#111111;">{calc_mape:.2f}%</div>
                <div class="kpi-card-badge badge-green">✅ 정확도 {100-calc_mape:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">📏 평균 제곱근 오차</div>
                <div class="kpi-card-value" style="color:#111111;">{calc_rmse:.1f}</div>
                <div class="kpi-card-badge badge-blue">RMSE (MWh)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            # [수정] 오타 수정 (검 검증 -> 검증)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-label">🗂️ 검증 데이터</div>
                <div class="kpi-card-value" style="font-size:2rem; color:#111111;">8,784 <span style="font-size:1rem; color:#7a8499">시간</span></div>
                <div class="kpi-card-badge badge-blue">Test: 2024년 기준</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">② 상세 분석</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">🔍 실제값 vs 예측값 상관관계</div>', unsafe_allow_html=True)
            fig_scat = px.scatter(df_res, x='전력사용량(MWh)', y='예측값(MWh)', opacity=0.3,
                                  trendline="ols", trendline_color_override="#C0392B")
            max_v = max(df_res['전력사용량(MWh)'].max(), df_res['예측값(MWh)'].max())
            # [수정] 오타 수정 (매in -> min)
            min_v = min(df_res['전력사용량(MWh)'].min(), df_res['예측값(MWh)'].min())
            fig_scat.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                               line=dict(dash="dash", color="#1B3A6B"), opacity=0.5)
            fig_scat.update_layout(height=360, margin=dict(t=4,b=4,l=4,r=4))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown("<div class='msg-box' style='background:#EBF3FB;color:#1B3A6B;'>💡 점들이 대각선(y=x) 근처에 밀집 — 예측 정확도 매우 높음</div>", unsafe_allow_html=True)

    with col_r:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">📊 오차율(%) 분포 분석</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(df_res, x='error_rate', nbins=50,
                                    labels={'error_rate': '오차율 (%)'},
                                    color_discrete_sequence=['#1B3A6B'])
            fig_hist.update_layout(height=360, margin=dict(t=4,b=4,l=4,r=4), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("<div class='msg-box' style='background:#EBF3FB;color:#1B3A6B;'>💡 오차율 5% 이내 집중 — 폭염 극단값 구간에서 일부 오차 발생</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">③ 인사이트 및 활용 방안</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-bar">
        <div class="insight-item">
            <div class="insight-icon">📊</div>
            <div>
                <div class="insight-title">예측 엔진 신뢰도</div>
                <div class="insight-desc">MAPE {calc_mape:.2f}%, R² {calc_r2:.4f} — 서울시 전력 수요 변동의 {calc_r2*100:.2f}%를 정확히 설명. 실운영 적용 가능 수준.</div>
            </div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div class="insight-icon">⚠️</div>
            <div>
                <div class="insight-title">한계 구간 식별</div>
                <div class="insight-desc">폭염(33°C+) 극단값 구간에서 오차율 상승. 이상 기후 발생 시 수동 개입 모드 전환 권고.</div>
            </div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div class="insight-icon">🔄</div>
            <div>
                <div class="insight-title">운영 활용 방안</div>
                <div class="insight-desc">피크 경보·ESS 충방전 스케줄링·탄소 배출 관리 등 전력 운영 의사결정 지원 도구로 활용 가능.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 페이지 4: 에너지 전환 전망
# ==========================================
elif page == "📈 에너지 전환 전망":

    renew_cost_drop  = 4.0   # 신재생 단가 하락률 (%/년) 고정
    renew_gen_growth = 8.0   # 신재생 발전량 성장률 (%/년) 고정
    nuke_gen_growth  = 1.5   # 원전 발전량 증가율 (%/년) 고정
    
    # ─── ① 단가 추이 + 역전 시점 예측 ──────────────────────────────
    st.markdown('<div class="section-label">① 에너지원별 정산단가(SMP) 추이 및 역전 시점 예측</div>', unsafe_allow_html=True)

    cost_cross_year = None
    last_nuke = 0
    last_renew = 0
    latest_label = ""

    try:
        df_cost = pd.read_csv(find_file('cost_fuel_type.csv'), encoding='cp949')
        df_cost['datetime'] = pd.to_datetime(df_cost['기간'])
        df_cost = df_cost.sort_values('datetime').reset_index(drop=True)

        renew_cols     = ['태양', '풍력', '수력', '해양', '바이오']
        existing_renew = [c for c in renew_cols if c in df_cost.columns]
        df_cost['신재생_평균'] = df_cost[existing_renew].replace(0, np.nan).mean(axis=1)

        last_row     = df_cost.iloc[-1]
        last_date    = last_row['datetime']
        last_nuke    = last_row['원자력']
        last_renew   = last_row['신재생_평균']
        latest_label = last_date.strftime('%Y.%m')

        future_months = pd.date_range(last_date, periods=12 * 20, freq='MS')
        n = np.arange(1, len(future_months) + 1)

        nuke_future  = last_nuke  * (1.015 ** (n / 12))
        renew_future = last_renew * ((1 - renew_cost_drop / 100) ** (n / 12))

        cost_cross_idx  = np.where(renew_future <= nuke_future)[0]
        cost_cross_year = future_months[cost_cross_idx[0]].year if len(cost_cross_idx) > 0 else None

        col_g1, col_g2 = st.columns([7, 3])

        with col_g1:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=df_cost['datetime'], y=df_cost['원자력'],
                name='원자력 (실측)', line=dict(color='#1B3A6B', width=2.5),
                hovertemplate='원자력: %{y:.1f}원/kWh'
            ))
            fig_cost.add_trace(go.Scatter(
                x=df_cost['datetime'], y=df_cost['신재생_평균'],
                name='신재생 평균 (실측)', line=dict(color='#27AE60', width=2.5),
                hovertemplate='신재생: %{y:.1f}원/kWh'
            ))
            fig_cost.add_trace(go.Scatter(
                x=future_months, y=nuke_future,
                name='원자력 (예측)', line=dict(color="#4B8FFC", width=1.5, dash='dot'), opacity=0.55
            ))
            fig_cost.add_trace(go.Scatter(
                x=future_months, y=renew_future,
                name='신재생 (예측)', line=dict(color="#6EEFA4", width=1.5, dash='dot'), opacity=0.55
            ))
            if cost_cross_year:
                cross_date = future_months[cost_cross_idx[0]]
                fig_cost.add_vline(
                    x=cross_date.timestamp() * 1000,
                    line_dash="dash", line_color="#C0392B", line_width=2,
                    annotation_text=f"💥 단가 역전 {cost_cross_year}년",
                    annotation_position="top right",
                    annotation_font_color="#C0392B",
                    annotation_font_size=13
                )
            fig_cost.update_layout(
                height=420, template="plotly_white",
                xaxis_title="연도", yaxis_title="정산단가 (원/kWh)",
                hovermode="x unified",
                xaxis_range=["2010-01-01", "2035-12-31"],
                legend=dict(
                    orientation="v", 
                    yanchor="bottom", y=0.01, 
                    xanchor="right", x=0.99, 
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(100,100,100,0.5)",
                    borderwidth=1.5,
                    font=dict(color="#333333")),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        with col_g2:
            cross_txt = f"<b style='color:#C0392B; font-size:1.8rem;'>{cost_cross_year}년</b>" \
                        if cost_cross_year else "<b style='color:#7a8499;'>역전 없음</b>"
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px; border:1px solid #E2E8F0; height:auto; margin-bottom:15px;">
                <div style="color:#7a8499; font-weight:700; font-size:0.95rem;">📌 현재 단가 ({latest_label})</div>
                <div style="margin:14px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span style="font-weight:600; color:#333;">⚛️ 원자력</span>
                        <b style="color:#1B3A6B; font-size:1.2rem;">{last_nuke:.1f}<span style="font-size:0.85rem;">원/kWh</span></b>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-weight:600; color:#333;">🌿 신재생에너지</span>
                        <b style="color:#27AE60; font-size:1.2rem;">{last_renew:.1f}<span style="font-size:0.85rem;">원/kWh</span></b>
                    </div>
                </div>
                <hr style="border:1px solid #E2E8F0; margin:10px 0;">
                <div style="color:#7a8499; font-weight:700; font-size:0.95rem;">🏁 단가 역전 예측 시점</div>
                <div style="margin-top:10px; text-align:center;">{cross_txt}</div>
                <div style="color:#7a8499; font-size:0.85rem; margin-top:6px; text-align:center;">
                    (신재생 단가 매년 {renew_cost_drop:.0f}%씩 떨어질 경우)
                </div>
            </div>
            """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"단가 데이터 로드 오류: {e}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


    # ─── ② 발전량 비중 역전 전망 ────────────────────────────────────
    st.markdown('<div class="section-label">② 발전량 비중 역전 시점 전망 (제11차 전력수급기본계획 기반)</div>', unsafe_allow_html=True)
 
    # 정부 계획 앵커 (2023 실적 + 2030·2038 계획)
    plan_years    = np.array([2023, 2030, 2038])
    nuke_plan     = np.array([30.7, 31.8, 35.2])
    renew_plan    = np.array([8.4,  18.8, 29.2])
 
    # 2023~2038: 정부 계획 보간
    plan_years_full = np.arange(2023, 2039)
    nuke_plan_full  = np.interp(plan_years_full, plan_years, nuke_plan)
    renew_plan_full = np.interp(plan_years_full, plan_years, renew_plan)
 
    # 2038~2050: 슬라이더 성장률 적용 (순수 예측 구간)
    pred_years = np.arange(2038, 2051)
    n_pred     = np.arange(0, len(pred_years))
 
    nuke_pred  = nuke_plan[-1]  * ((1 + nuke_gen_growth  / 100) ** n_pred)
    renew_pred = renew_plan[-1] * ((1 + renew_gen_growth  / 100) ** n_pred)
    renew_opt  = renew_plan[-1] * ((1 + (renew_gen_growth + 3)           / 100) ** n_pred)
    renew_con  = renew_plan[-1] * ((1 + max(renew_gen_growth - 3, 0.5)   / 100) ** n_pred)
 
    def find_cross(a, b, years):
        idx = np.where(a >= b)[0]
        return int(years[idx[0]]) if len(idx) > 0 else None
 
    gen_cross_year     = find_cross(renew_pred, nuke_pred, pred_years)
    gen_cross_opt_year = find_cross(renew_opt,  nuke_pred, pred_years)
    gen_cross_con_year = find_cross(renew_con,  nuke_pred, pred_years)
 
    fig_gen = go.Figure()
 
    # 시나리오 밴드 (예측 구간만)
    fig_gen.add_trace(go.Scatter(
        x=np.concatenate([pred_years, pred_years[::-1]]),
        y=np.concatenate([renew_opt,  renew_con[::-1]]),
        fill='toself', fillcolor='rgba(39,174,96,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='신재생 시나리오 범위', hoverinfo='skip'
    ))
 
    # 정부 계획 구간 실선 (2023~2038)
    fig_gen.add_trace(go.Scatter(
        x=plan_years_full, y=nuke_plan_full,
        name='원자력 (정부 계획)', line=dict(color='#1B3A6B', width=2.5),
        hovertemplate='원자력 계획: %{y:.1f}%'
    ))
    fig_gen.add_trace(go.Scatter(
        x=plan_years_full, y=renew_plan_full,
        name='신재생 (정부 계획)', line=dict(color='#27AE60', width=2.5),
        hovertemplate='신재생 계획: %{y:.1f}%'
    ))
 
    # 예측 구간 점선 (2038~2050)
    fig_gen.add_trace(go.Scatter(
        x=pred_years, y=nuke_pred,
        name='원자력 (예측)', line=dict(color='#1B3A6B', width=1.5, dash='dot'), opacity=0.6
    ))
    fig_gen.add_trace(go.Scatter(
        x=pred_years, y=renew_pred,
        name=f'신재생 전망 ({renew_gen_growth:.0f}%/년)',
        line=dict(color='#27AE60', width=2.5, dash='dot'),
        hovertemplate='신재생 전망: %{y:.1f}%'
    ))
    fig_gen.add_trace(go.Scatter(
        x=pred_years, y=renew_opt,
        name=f'빠른 성장 시 ({renew_gen_growth + 3:.0f}%/년)',
        line=dict(color='#27AE60', width=1.5, dash='dot'), opacity=0.75
    ))
    fig_gen.add_trace(go.Scatter(
        x=pred_years, y=renew_con,
        name=f'성장 둔화 시 ({max(renew_gen_growth - 3, 0.5):.0f}%/년)',
        line=dict(color='#E67E22', width=1.5, dash='dot'), opacity=0.75
    ))
 
    # 앵커 포인트 (실적/계획치 확인용)
    fig_gen.add_trace(go.Scatter(
        x=plan_years, y=nuke_plan, mode='markers',
        marker=dict(color='#1B3A6B', size=10, symbol='circle'),
        name='원자력 실적/계획치'
    ))
    fig_gen.add_trace(go.Scatter(
        x=plan_years, y=renew_plan, mode='markers',
        marker=dict(color='#27AE60', size=10, symbol='circle'),
        name='신재생 실적/계획치'
    ))
 
    # 역전 시점 수직선 + annotation (y 위치 분산)
    for yr, label, color, y_pos in [
        (gen_cross_opt_year, f'🚀 빠른 성장 {gen_cross_opt_year}년', '#27AE60', 58),
        (gen_cross_year,     f'📊 기준 전망 {gen_cross_year}년',     '#C0392B', 52),
        (gen_cross_con_year, f'🐢 성장 둔화 {gen_cross_con_year}년', '#E67E22', 46),
    ]:
        if yr:
            fig_gen.add_shape(
                type="line", x0=yr, x1=yr, y0=0, y1=65,
                line=dict(color=color, width=2, dash="dash")
            )
            fig_gen.add_annotation(
                x=yr, y=y_pos,
                text=label,
                showarrow=False,
                font=dict(color=color, size=12),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                xanchor="left"
            )
 
    fig_gen.update_layout(
        height=450, template="plotly_white",
        xaxis_title="연도", yaxis_title="발전량 비중 (%)",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1.5,
            font=dict(color="#333333")
        ),
        margin=dict(t=40, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_gen, use_container_width=True)
 
    # 시나리오 설명 카드
    st.markdown("""
    <div style="display:flex; gap:12px; margin-top:-8px; margin-bottom:20px;">
        <div style="flex:1; background:#EAFAF1; border-left:4px solid #27AE60; border-radius:6px; padding:12px 16px;">
            <div style="font-weight:800; color:#27AE60; margin-bottom:4px;">🚀 빠른 성장 시나리오</div>
            <div style="font-size:0.85rem; color:#555;">정부 정책이 예정대로 추진되고 태양광·풍력 보급이 가속화될 경우</div>
        </div>
        <div style="flex:1; background:#FDEDEC; border-left:4px solid #C0392B; border-radius:6px; padding:12px 16px;">
            <div style="font-weight:800; color:#C0392B; margin-bottom:4px;">📊 현재 추세 유지 시나리오</div>
            <div style="font-size:0.85rem; color:#555;">제11차 전력수급기본계획을 기준으로 현재 성장률이 유지될 경우</div>
        </div>
        <div style="flex:1; background:#FEF9E7; border-left:4px solid #E67E22; border-radius:6px; padding:12px 16px;">
            <div style="font-weight:800; color:#E67E22; margin-bottom:4px;">🐢 성장 둔화 시나리오</div>
            <div style="font-size:0.85rem; color:#555;">인허가 지연, 주민 수용성 문제 등 신재생에너지 보급이 지체될 경우</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
 
    # ─── ③ 종합 전망 KPI ────────────────────────────────────────────
    st.markdown('<div class="section-label">③ 종합 전망 — 에너지 전환 타임라인</div>', unsafe_allow_html=True)
 
    cost_txt = f"{cost_cross_year}년" if cost_cross_year else "역전 없음"
    gen_txt  = f"{gen_cross_year}년"     if gen_cross_year     else "2050년 이후"
    opt_txt  = f"{gen_cross_opt_year}년" if gen_cross_opt_year else "2050년 이후"
    con_txt  = f"{gen_cross_con_year}년" if gen_cross_con_year else "2050년 이후"
 
    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    with c_k1:
        st.markdown(f"""
        <div class="kpi-main" style="height:auto; padding:22px;">
            <div class="kpi-main-label">💰 단가 역전 예측</div>
            <div class="kpi-main-value" style="font-size:2.1rem;">{cost_txt}</div>
            <div class="kpi-badge" style="background:rgba(255,255,255,0.15); color:white;">
                매년 {renew_cost_drop:.0f}%씩 저렴해질 경우
            </div>
        </div>""", unsafe_allow_html=True)
    with c_k2:
        st.markdown(f"""
        <div class="kpi-main" style="height:auto; padding:22px;">
            <div class="kpi-main-label">⚡ 발전량 역전 (기준)</div>
            <div class="kpi-main-value" style="font-size:2.1rem;">{gen_txt}</div>
            <div class="kpi-badge" style="background:rgba(255,255,255,0.15); color:white;">
                성장률 {renew_gen_growth:.0f}%/년 기준
            </div>
        </div>""", unsafe_allow_html=True)
    with c_k3:
        st.markdown(f"""
        <div class="kpi-card" style="height:auto; padding:22px;">
            <div class="kpi-card-label">🚀 빠른 성장 시</div>
            <div class="kpi-card-value" style="color:#27AE60;">{opt_txt}</div>
            <div class="kpi-card-badge badge-green">성장 가속 시나리오</div>
        </div>""", unsafe_allow_html=True)
    with c_k4:
        st.markdown(f"""
        <div class="kpi-card" style="height:auto; padding:22px;">
            <div class="kpi-card-label">🐢 성장 둔화 시</div>
            <div class="kpi-card-value" style="color:#E67E22;">{con_txt}</div>
            <div class="kpi-card-badge badge-orange">성장 둔화 시나리오</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
 
    # ─── ④ 인사이트 바 ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="insight-bar">
        <div class="insight-item">
            <div style="font-size:1.5rem;">💰</div>
            <div>
                <div class="insight-title">단가 역전 — {cost_txt}</div>
                <div class="insight-desc">신재생 단가가 원전을 하회하며 보조금 없는 경쟁의 시대가 열립니다.</div>
            </div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div style="font-size:1.5rem;">⚡</div>
            <div>
                <div class="insight-title">골든크로스 — {opt_txt} ~ {con_txt}</div>
                <div class="insight-desc">기준 시나리오 <b>{gen_txt}</b>, 신재생이 원전을 넘어 주력 전원으로 자리잡습니다.</div>
            </div>
        </div>
        <div class="insight-divider"></div>
        <div class="insight-item">
            <div style="font-size:1.5rem;">🔋</div>
            <div>
                <div class="insight-title">전환의 열쇠 — ESS · 유연운전</div>
                <div class="insight-desc">21.5GW ESS 구축과 원전 유연 운전이 뒷받침될 때 전환은 완성됩니다.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)