import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import time
from sklearn.metrics import r2_score
import joblib
import json
import datetime
import streamlit.components.v1 as components

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

# 2. CSS 설정
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

    footer { visibility: hidden !important; }

    html, body, [class*="css"], p, div, span { font-size: 18px !important; }

    .block-container {
        padding-top: 3rem !important; padding-bottom: 1rem;
        padding-left: 1rem !important; padding-right: 2rem !important;
    }

    [data-testid="collapsedControl"] {
        z-index: 999 !important;
        position: fixed !important;
    }

    .section-label {
        font-size: 0.85rem !important; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #7a8499; margin: 10px 0px 15px 0px;
        display: flex; align-items: center; gap: 10px;
    }
    .section-label::after { content:''; flex:1; height:1px; background:#E2E8F0; }

    .kpi-main {
        background: #1B3A6B; border-radius: 12px; padding: 20px 22px; color: white;
        height: 160px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;
    }
    .kpi-main-label { font-size: 0.9rem !important; opacity: 0.8; }
    .kpi-main-value { font-size: 2.8rem !important; font-weight: 900; font-family: 'DM Mono', monospace; line-height: 1; }
    .kpi-badge {
        display: inline-flex; align-items: center; gap: 6px; border-radius: 20px; padding: 6px 14px;
        font-size: 0.85rem !important; font-weight: 700; width: fit-content;
    }
    .kpi-card {
        background: white; border-radius: 12px; padding: 20px 20px; border: 1px solid #E2E8F0;
        height: 160px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;
    }
    .kpi-card-label { font-size: 0.95rem !important; color: #7a8499; font-weight: 700; }
    .kpi-card-value { font-size: 2.2rem !important; font-weight: 900; font-family: 'DM Mono', monospace; }
    .kpi-card-badge {
        display: inline-flex; align-items: center; border-radius: 20px; padding: 6px 14px;
        font-size: 0.85rem !important; font-weight: 700; width: fit-content;
    }
    .badge-red    { background:#FDEDEC; color:#C0392B; }
    .badge-green  { background:#EAFAF1; color:#27AE60; }
    .badge-orange { background:#FEF9E7; color:#E67E22; }
    .badge-blue   { background:#EBF3FB; color:#1B3A6B; }

    .mid-card-title {
        font-size: 1.15rem !important; color: #1B3A6B; font-weight: 800; margin-bottom: 10px;
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

    @media (max-width: 768px) {
        .insight-bar {
            grid-template-columns: 1fr !important;
            padding: 16px !important;
        }
        .insight-divider { display: none !important; }
        .insight-item {
            padding: 12px 0 !important;
            border-bottom: 1px solid rgba(255,255,255,0.15);
        }
        .insight-item:last-child { border-bottom: none !important; }
        .header-inner { flex-direction: column !important; gap: 6px !important; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    df     = pd.read_csv(find_file('df_refined_v2.csv'))
    df_res = pd.read_csv(find_file('df_predicted_v2.csv'))
    df['datetime']     = pd.to_datetime(df['datetime'])
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    return df, df_res

df, df_res = load_all_data()

CARBON_FACTOR = 0.4594
df_2024   = df_res[df_res['datetime'].dt.year == 2024]
calc_r2   = r2_score(df_2024['전력사용량(MWh)'], df_2024['예측값(MWh)'])
calc_mape = np.mean(np.abs((df_2024['전력사용량(MWh)'] - df_2024['예측값(MWh)']) / df_2024['전력사용량(MWh)'])) * 100
calc_rmse = np.sqrt(np.mean((df_2024['전력사용량(MWh)'] - df_2024['예측값(MWh)'])**2))

# 3. 사이드바
st.sidebar.markdown("### 📋 관제 메뉴")
page = st.sidebar.radio("메뉴를 선택하세요", ["⚡ 전력 수요 시뮬레이션", "🔍 AI 예측 모델 분석"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛰️ SYSTEM CONTROL")
st.sidebar.header("🌡️ 기상 시나리오 설정")

st.sidebar.write("예상 기온 (°C)")
c1_temp, c2_temp = st.sidebar.columns([7, 3])
with c1_temp:
    temp_slider = st.slider("temp_slider_label", -10.0, 40.0, 25.0, 0.5, label_visibility="collapsed")
with c2_temp:
    s_temp = st.number_input("temp_input", -10.0, 40.0, value=temp_slider, step=0.5, label_visibility="collapsed")

st.sidebar.write("예상 습도 (%)")
c1_hum, c2_hum = st.sidebar.columns([7, 3])
with c1_hum:
    hum_slider = st.slider("hum_slider_label", 0, 100, 50, 5, label_visibility="collapsed")
with c2_hum:
    s_hum = st.number_input("hum_input", 0, 100, value=hum_slider, step=5, label_visibility="collapsed")

st.sidebar.write("예상 풍속 (m/s)")
c1_wind, c2_wind = st.sidebar.columns([7, 3])
with c1_wind:
    wind_slider = st.slider("wind_slider_label", 0.0, 15.0, 2.0, 0.5, label_visibility="collapsed")
with c2_wind:
    s_wind = st.number_input("wind_input", 0.0, 15.0, value=wind_slider, step=0.5, label_visibility="collapsed")

st.sidebar.write("예상 시간 (시)")
c1_hour, c2_hour = st.sidebar.columns([7, 3])
with c1_hour:
    hour_slider = st.slider("hour_slider_label", 0, 23, 14, 1, label_visibility="collapsed")
with c2_hour:
    s_hour = st.number_input("hour_input", 0, 23, value=hour_slider, step=1, label_visibility="collapsed")

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
    input_row['dayofweek']    = now.weekday()
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
carbon_ton = predicted_mwh * 1000 * CARBON_FACTOR / 1000

yoy_delta = None
same_cond = df[(df['hour'] == s_hour) & (df['month'] == s_month)]
if len(same_cond) >= 2:
    a23 = same_cond[same_cond['datetime'].dt.year == 2023]['전력사용량(MWh)'].mean()
    a24 = same_cond[same_cond['datetime'].dt.year == 2024]['전력사용량(MWh)'].mean()
    if a23 > 0 and not np.isnan(a24):
        yoy_delta = ((a24 - a23) / a23) * 100

if s_temp >= 33 or s_discomfort >= 80:
    alert_status, alert_color, alert_emoji = "비상", "#C0392B", "🔴"
elif s_temp >= 30 or s_discomfort >= 75:
    alert_status, alert_color, alert_emoji = "경계", "#E67E22", "🟡"
else:
    alert_status, alert_color, alert_emoji = "정상", "#27AE60", "🟢"

is_peak = 14 <= s_hour <= 17

# 4. 상단 헤더 (실시간 시계)
now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
components.html("""
<style>
  .header-box {
    background:#1B3A6B; border-radius:12px; padding:20px 24px;
    margin-bottom:0px; box-shadow:0 4px 6px rgba(0,0,0,0.1);
    display:flex; justify-content:space-between; align-items:flex-start;
    flex-wrap:wrap; gap:8px;
  }
  .header-title { font-size:1.3rem; font-weight:800; color:white; font-family:sans-serif; }
  .header-sub { font-size:0.85rem; opacity:0.7; color:white; margin-top:4px; font-family:'DM Mono',monospace; }
  #live-clock { color:white; font-family:'DM Mono',monospace; font-size:0.85rem; opacity:0.8; }
</style>
<div class="header-box">
  <div>
    <div class="header-title">🔌 서울시 전력 수요 관제 시스템</div>
    <div class="header-sub">Seoul Energy Demand Simulation Dashboard</div>
  </div>
  <div id="live-clock">🕐 --:--</div>
</div>
<script>
  function updateClock() {
    const now = new Date();
    const y = now.getFullYear();
    const mo = String(now.getMonth()+1).padStart(2,'0');
    const d = String(now.getDate()).padStart(2,'0');
    const h = String(now.getHours()).padStart(2,'0');
    const mi = String(now.getMinutes()).padStart(2,'0');
    document.getElementById('live-clock').innerText = '🕐 ' + y + '-' + mo + '-' + d + ' ' + h + ':' + mi;
  }
  updateClock();
  setInterval(updateClock, 1000);
</script>
""", height=120)


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
    carbon_msg   = "⚠ 고탄소" if carbon_ton > 2500 else ("⚠ 주의" if carbon_ton > 2000 else "✅ 정상")
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
                <div class="kpi-card-label">🌿 탄소 배출 추정 </div>
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
            di_msg = "😰 매우 불쾌 — 냉방 급증" if s_discomfort>=80 else \
                     ("😓 불쾌 — 냉방 부하 증가" if s_discomfort>=75 else \
                     ("🙂 보통 — 안정 구간" if s_discomfort>=70 else "😊 쾌적 — 냉방 수요 낮음"))
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

    peak_t, peak_d = ("⚡ 피크 시간대 수요 증가 예상", "14~17시 냉방 집중으로 전력 수요 급증. 평균 대비 초과 구간 진입.") \
                      if is_peak else ("⚡ 안정적 수요 구간", "현재 시간대는 피크 외 구간으로 전력 수요가 안정적입니다.")
    cool_t, cool_d = ("❄️ 냉방 부하 증가 가능성", f"기온 {s_temp:.1f}°C — EHP 집중 가동 예상.") \
                      if s_temp >= 28 else \
                     ("🔥 난방 부하 증가 가능성", f"기온 {s_temp:.1f}°C — 난방 수요 증가 예상.") \
                      if s_temp <= 5 else \
                     ("✅ 냉난방 부하 안정", f"기온 {s_temp:.1f}°C는 임계점 이내로 부하가 안정적입니다.")
    resp_t, resp_d = ("🔄 전력 수요 분산 운영 필요", "ESS 방전 대기, 산업용 냉방 부하 순차 조정 권고.") \
                      if alert_status == "비상" else \
                     ("🔄 수요 분산 모니터링", "공공기관 냉방기기 순차 운전 검토 권고.") \
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
# 페이지 2: AI 예측 모델 분석
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
            fig_scat = px.scatter(df_res, x='전력사용량(MWh)', y='예측값(MWh)', opacity=0.3, trendline="ols", trendline_color_override="#C0392B")
            max_v = max(df_res['전력사용량(MWh)'].max(), df_res['예측값(MWh)'].max())
            min_v = min(df_res['전력사용량(MWh)'].min(), df_res['예측값(MWh)'].min())
            fig_scat.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="#1B3A6B"), opacity=0.5)
            fig_scat.update_layout(height=360, margin=dict(t=4,b=4,l=4,r=4))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown("<div class='msg-box' style='background:#EBF3FB;color:#1B3A6B;'>💡 점들이 대각선(y=x) 근처에 밀집 — 예측 정확도 매우 높음</div>", unsafe_allow_html=True)

    with col_r:
        with st.container(border=True):
            st.markdown('<div class="mid-card-title">📊 오차율(%) 분포 분석</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(df_res, x='error_rate', nbins=50, labels={'error_rate': '오차율 (%)'}, color_discrete_sequence=['#1B3A6B'])
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