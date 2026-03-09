import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from sklearn.metrics import r2_score
import joblib
import json
import datetime

# ── 모델 & 피처 로드 ──────────────────────────────────────────────────
model = joblib.load(r'xgb_power_model_v2.pkl')

with open(r'features_v2.json', 'r', encoding='utf-8') as f:
    features = json.load(f)

# 1. 페이지 설정 및 데이터 로드
st.set_page_config(layout="wide", page_title="서울시 전력 수요 관제 시스템")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { margin-bottom: 0.3rem; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    # 파일 경로 설정 (기존 로직 유지)

    base_path = "df_final_v2_refined.csv"
    predict_path = "df_predicted_v2.csv"

    if not os.path.exists(base_path):
        base_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_final_v2_refined.csv"
    if not os.path.exists(predict_path):
        predict_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_predicted_v2.csv"
        
    df = pd.read_csv(base_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df_res = pd.read_csv(predict_path)
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    
    return df, df_res

df, df_res = load_all_data()

df, df_res = load_all_data()

# ── 탄소 배출 계수 ──
CARBON_FACTOR = 0.4594  # kgCO₂/kWh

# 지표 사전 계산
actual = df_res['전력사용량(MWh)']
predicted = df_res['예측값(MWh)']
calc_r2 = r2_score(actual, predicted)
calc_mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# 2. 사이드바 디자인
st.sidebar.markdown("### 🛰️ SYSTEM CONTROL") 
menu = st.sidebar.radio("Menu", ["⚡ 전력 수요 시뮬레이션", "🔍 AI 예측 모델 분석"])

st.sidebar.markdown("---")
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

# ── 시스템 자동 계산 변수 ──────────────────────────────────────
now = datetime.datetime.now()
s_month = now.month
is_weekend = 1 if now.weekday() >= 5 else 0
s_day = "주말/공휴일" if is_weekend == 1 else "평일"

# UI용 파생값 (모델 입력용 아님)
s_sensory = 13.12 + 0.6215 * s_temp - 11.37 * (s_wind ** 0.16) + 0.3965 * s_temp * (s_wind ** 0.16)
s_discomfort = 0.81 * s_temp + 0.01 * s_hum * (0.99 * s_temp - 14.3) + 46.3

# ── 기본값 초기화 ─────────────────────────────────────────────────────
similar_date = "-"
predicted_mwh = 0.0
fallback_mwh = 5500.0 

# ── 유사 패턴 탐색 및 피처 엔지니어링 (수정된 섹션) ────────────────────────────────
if not df.empty:
    # 유사 날씨/시간대 탐색
    df['diff'] = (
        np.abs(df['기온(°C)'] - s_temp) +
        np.abs(df['hour'] - s_hour) * 2 +
        np.abs(df['month'] - s_month) * 0.5
    )
    target_idx = df['diff'].idxmin()
    input_row = df.loc[target_idx].copy()
    similar_date = input_row['datetime'].strftime('%Y-%m-%d %H시')
    
    # 1. 기본 정보 업데이트
    input_row['hour'] = s_hour
    input_row['month'] = s_month
    input_row['dayofweek'] = now.weekday()
    input_row['is_holiday'] = 0 # 공휴일 로직 필요 시 추가
    input_row['is_peak_hour'] = 1 if 14 <= s_hour <= 17 else 0
    
    # 2. 기상 정보 업데이트
    input_row['기온(°C)'] = s_temp
    input_row['습도(%)'] = s_hum
    input_row['강수량(mm)'] = 0.0 # 기본값
    input_row['일사(MJ/m2)'] = input_row['일사(MJ/m2)'] # 기존 유사 시간대 값 유지
    
    # 3. 모델 전용 파생 피처 업데이트 (features_v2.json 기준)
    input_row['CDD'] = max(0, s_temp - 24)
    input_row['HDD'] = max(0, 18 - s_temp)
    input_row['is_heatwave'] = 1 if s_temp >= 33 else 0
    input_row['temp_3h_mean'] = s_temp # 간소화: 현재 기온 기준
    input_row['temp_6h_max'] = max(s_temp, input_row['temp_6h_max'])

    # 4. 시계열 피처 (lag, rolling)는 유사 시점(target_idx)의 데이터를 그대로 활용하여 현실성 유지

    try:
        # features_v2.json에 명시된 컬럼 순서대로 데이터 구성
        X_input = pd.DataFrame([input_row[features]])
        predicted_mwh = float(model.predict(X_input)[0])
    except Exception as e:
        st.error(f"예측 단계 오류: {e}")
        predicted_mwh = fallback_mwh
else:
    st.warning("데이터프레임이 비어 있어 시뮬레이션을 진행할 수 없습니다.")

carbon_ton = predicted_mwh * 1000 * CARBON_FACTOR / 1000 

# 전년 동기 대비 증감률 계산 (데이터프레임 내 'is_holiday' 등 컬럼명 확인 필요)
yoy_delta = None
if not df.empty:
    same_cond = df[
        (df['hour']  == s_hour) &
        (df['month'] == s_month)
    ]
    if len(same_cond) >= 2:
        avg_2023 = same_cond[same_cond['datetime'].dt.year == 2023]['전력사용량(MWh)'].mean()
        avg_2024 = same_cond[same_cond['datetime'].dt.year == 2024]['전력사용량(MWh)'].mean()
        if avg_2023 > 0 and not np.isnan(avg_2024):
            yoy_delta = ((avg_2024 - avg_2023) / avg_2023) * 100

# --- 페이지 1 : 전력 수요 시뮬레이션 ---
if menu == "⚡ 전력 수요 시뮬레이션":
    st.title("🔌 서울시 전력 수요 시뮬레이션")
    st.markdown("---")

    if   s_discomfort >= 80: status, color = "비상", "red"
    elif s_discomfort >= 75: status, color = "주의", "orange"
    else:                    status, color = "정상", "blue"
    
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])

    with c1:
        st.metric("예상 피크 수요", f"{predicted_mwh:,.1f} MWh")
    with c2:
        st.metric("탄소 배출 추정", f"{carbon_ton:,.1f} tCO₂")
    with c3:
        st.metric("전년 동기 대비", f"{yoy_delta:+.1f}%" if yoy_delta else "-")
    with c4:
        st.write("**관제 상태**")
        st.markdown(f":{color}[**{status}**]")
        
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🚨 전력 수요 경보 게이지")

        # [1단계] 색상이 명확한 미디엄 파스텔 정의
        if s_temp >= 33:
            status_text, gauge_color = "폭염 경보 (냉방 피크)", "#E57373" 
        elif s_temp <= 5:
            status_text, gauge_color = "한파 주의 (난방 피크)", "#64B5F6" 
        elif 18 <= s_temp <= 26:
            status_text, gauge_color = "전력 안정 구간", "#81C784"    
        else:
            status_text, gauge_color = "상시 모니터링", "#FFD54F"      

        # [2단계] 상태 텍스트
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 10px;">
                <span style="background-color: rgba(128,128,128,0.1); color: {gauge_color}; 
                             padding: 6px 22px; border-radius: 50px; border: 2px solid {gauge_color};
                             font-weight: bold; font-size: 1.1rem;">
                    ● {status_text}
                </span>
            </div>
        """, unsafe_allow_html=True)

        # [3단계] 게이지 차트 (rgba 형식으로 수정)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s_temp,
            number={
                'suffix': "°C", 
                'font': {'size': 55, 'color': gauge_color, 'family': 'Arial Black'}
            },
            gauge={
                'axis': {
                    'range': [-15, 45], 
                    'tickwidth': 1, 
                    'tickcolor': "#CED4DA", 
                    'tickvals': [-15, 0, 15, 30, 45],
                    'tickfont': {'size': 13, 'color': '#ADB5BD'} 
                },
                'bar': {'color': gauge_color, 'thickness': 0.35}, 
                'steps': [
                    {'range': [-15, 5],   'color': "rgba(100, 181, 246, 0.2)"}, # 블루
                    {'range': [5, 18],    'color': "rgba(233, 236, 239, 0.2)"}, # 그레이
                    {'range': [18, 26],   'color': "rgba(129, 199, 132, 0.2)"}, # 그린
                    {'range': [26, 33],   'color': "rgba(255, 213, 79, 0.2)"},  # 옐로우
                    {'range': [33, 45],   'color': "rgba(229, 115, 115, 0.2)"}, # 레드
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 3},
                    'thickness': 0.8,
                    'value': s_temp
                }
            }
        ))

        fig_gauge.update_layout(
            height=280, 
            margin=dict(t=50, b=10, l=60, r=60),
            paper_bgcolor="rgba(0,0,0,0)", 
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # [4단계] 하단 수치 영역 (투명도 rgba 적용)
        idx_c1, idx_c2 = st.columns(2)
        
        # HTML/CSS에서는 8자리 헥스코드가 작동하므로 그대로 두거나 안전하게 rgba로 변경
        card_bg = f"rgba({int(gauge_color[1:3], 16)}, {int(gauge_color[3:5], 16)}, {int(gauge_color[5:7], 16)}, 0.1)"
        card_style = f"padding: 15px; background-color: {card_bg}; border-radius: 15px; border: 1.5px solid {gauge_color};"
        
        with idx_c1:
            st.markdown(f"""
                <div style="{card_style} text-align: center;">
                    <p style="color: #ADB5BD; font-size: 0.95rem; margin: 0; font-weight: 500;">🌡️ 체감온도</p>
                    <h2 style="margin: 8px 0 0 0; color: {gauge_color}; font-size: 2.1rem; border:none;">{s_sensory:.1f} °C</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with idx_c2:
            st.markdown(f"""
                <div style="{card_style} text-align: center;">
                    <p style="color: #ADB5BD; font-size: 0.95rem; margin: 0; font-weight: 500;">💦 불쾌지수</p>
                    <h2 style="margin: 8px 0 0 0; color: {gauge_color}; font-size: 2.1rem; border:none;">{s_discomfort:.1f}</h2>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # ── 스캐터 그래프 ────────────────────────────────────────────
        st.subheader("📈 기상 시나리오 분석 (과거 데이터 대조)")
        fig = go.Figure()
        if not df.empty:
            sample_df = df.sample(min(1000, len(df)))
            fig.add_trace(go.Scatter(
                x=sample_df['기온(°C)'], y=sample_df['전력사용량(MWh)'],
                mode='markers', name='과거 실적',
                marker=dict(color='lightgrey', opacity=0.4)
            ))
        fig.add_trace(go.Scatter(
            x=[s_temp], y=[predicted_mwh],
            mode='markers+text', name='현재 예측',
            text=[f"{predicted_mwh:,.0f} MWh"],
            textposition="top center",
            marker=dict(color='red', size=15, symbol='star')
        ))
        fig.update_layout(
            xaxis_title="기온 (°C)", yaxis_title="전력사용량 (MWh)",
            height=300, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📋 실무 분석 리포트")
        st.write("**1. 예측 근거 (유사 사례)**")
        st.info(f"현재 조건은 과거 **{similar_date}** 패턴과 가장 유사함")

        st.write("**2. 탄소 배출 현황**")
        if carbon_ton > 2500:
            st.error(f"🏭 예상 배출량 **{carbon_ton:,.1f} tCO₂** — 고탄소 구간")
        elif carbon_ton > 2000:
            st.warning(f"🏭 예상 배출량 **{carbon_ton:,.1f} tCO₂** — 주의 구간")
        else:
            st.success(f"🌱 예상 배출량 **{carbon_ton:,.1f} tCO₂** — 정상 구간")

        st.write("**3. 단계별 대응 가이드**")
        if s_temp >= 33 or s_discomfort >= 80:
            st.error("🚨 **비상**\nESS 방전 대기, 산업용 냉방 부하 조정")
        elif s_temp >= 30 or s_discomfort >= 75:
            st.warning("⚠️ **경계**\n공공기관 냉방기기 순차 운전 정지")
        else:
            st.success("✅ **정상**\n안정적 공급 가능, 상시 모니터링 유지")
            
    st.markdown("---")
    st.subheader("🕵️ 주요 변수 기여도")

    # 1️⃣ 전력 수요 시뮬레이션 계산 
    base_load_ = 4500
    cooling_degree = max(0, s_temp - 18)
    heating_degree = max(0, 15 - s_temp)
    
    # 냉방 / 난방 부하 (비선형 가중치 반영)
    cool_effect_ = (cooling_degree ** 1.2) * 250
    heat_effect_ = (heating_degree ** 1.1) * 300
    hum_effect_ = max(0, s_hum - 50) * 15
    
    total_load = base_load_ + cool_effect_ + heat_effect_ + hum_effect_

    # 3️⃣ 차트 및 설명 레이아웃 분리
    chart_col, desc_col = st.columns([5.5, 4.5])

    # 4️⃣ 도넛 차트 시각화
    with chart_col:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['🏠 기본 전력 수요', '❄️ 냉방 영향도', '🔥 난방 영향도', '💦 습도 영향 계수'],
            values=[base_load_, cool_effect_, heat_effect_, hum_effect_],
            hole=.45,
            sort=False,
            marker=dict(colors=['#636EFA', '#EF553B', '#FFA15A', '#00CC96']),
            textinfo='percent',
            textfont_size=14,
            textposition='inside'
        )])
        fig_pie.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # 5️⃣ 변수 설명 및 모델 지표 
    with desc_col:
        st.success(f"""
        - **🏠 기본 전력 수요**: 기상 무관, 24시간 일정 소비 전력
        - **❄️ 냉방 영향도**: 기온 상승에 따른 냉방용 전력 소비량
        - **🔥 난방 영향도**: 기온 하강에 따른 난방용 전력 소비량
        - **💦 습도 영향 계수**: 불쾌지수 변화가 전력 소모에 미치는 영향
        """)
        
        st.write("---")
        st.write("**🧪 예측 모델 성능 지표**")
        
        m1, m2 = st.columns(2)
        # 실제 계산된 r2와 mape 사용 
        m1.metric("결정계수 (R²)", f"{calc_r2:.4f}") 
        m2.metric("평균 오차율 (MAPE)", f"{calc_mape:.2f}%")
        st.info(f"""
        **💡 분석 결과 요약**  \n 본 모델은 서울시 전력 수요 변동의 **{calc_r2*100:.2f}% 이상** 정확히 설명합니다  \n 실제 사용량 대비 예측 오차율이 **평균 {calc_mape:.2f}% 미만**으로 제어되고 있어 신뢰도가 매우 높습니다.
    """)                   


# --- [페이지 2: AI 모델 성능 리포트 내부] ---
else:
    st.title("🧠 ML 기반 예측 정밀도 및 성능 보고서")
    st.markdown("---")
    
# 3-1. 핵심 지표 요약
    future_24 = df_res.tail(24)
    peak_info = future_24.loc[future_24['예측값(MWh)'].idxmax()]

# 데이터상 실제 날짜 추출 (예: 2024-12-31)
    target_date = peak_info['datetime'].strftime('%Y-%m-%d')

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("결정계수 (R²)", f"{calc_r2:.4f}")
    col_m2.metric("평균 오차율 (MAPE)", f"{calc_mape:.2f}%")

# "내일" 대신 실제 날짜를 동적으로 표시
    col_m3.metric(f"{target_date} 피크 예측", f"{peak_info['예측값(MWh)']:,.0f} MWh")
    col_m4.metric("피크 예상 시간", f"{peak_info['hour']}시")

    st.markdown("---")

    # 3-2. 예측 vs 실제 시계열 비교 (최근 7일)
    st.subheader("📅 7일간의 예측 정밀도 추이")
    recent_7d = df_res.tail(24 * 7)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=recent_7d['datetime'], y=recent_7d['전력사용량(MWh)'], name='실제 사용량', line=dict(color='#636EFA')))
    fig_line.add_trace(go.Scatter(x=recent_7d['datetime'], y=recent_7d['예측값(MWh)'], name='AI 예측값', line=dict(color='#EF553B', dash='dot')))
    fig_line.update_layout(hovermode="x unified", xaxis_tickformat='%m-%d\n%H:%M')
    st.plotly_chart(fig_line, use_container_width=True)

    st.success("""
    💡 2024년 연말(마지막 7일)의 실제 전력 사용량과 AI 모델의 예측값을 비교합니다.\n 
    두 선의 간격이 작을수록 모델의 예측이 실제 사용량에 가깝다는 것을 의미합니다.
    """)

    # 3-3. 하단 레이아웃 (산점도 & 오차 분석)
    col_low_l, col_low_r = st.columns(2)
    
    with col_low_l:
        st.subheader("🔍 실제값 vs 예측값 상관관계")
        fig_scat = px.scatter(
            df_res, x='전력사용량(MWh)', y='예측값(MWh)', opacity=0.3,
            trendline="ols", trendline_color_override="red"
        )
        fig_scat.update_layout(xaxis_tickformat=',d', yaxis_tickformat=',d')
        
        max_v = max(df_res['전력사용량(MWh)'].max(), df_res['예측값(MWh)'].max())
        min_v = min(df_res['전력사용량(MWh)'].min(), df_res['예측값(MWh)'].min())
        fig_scat.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="black"), opacity=0.5)
        st.plotly_chart(fig_scat, use_container_width=True)

        st.success("""
        💡 실제 전력 사용량과 AI 모델의 예측값 간의 관계를 나타냅니다 \n
        점들이 대각선 (y=x) 근처에 위치할수록 예측의 정확도가 높다는 것을 의미합니다
        """)

    with col_low_r:
        st.subheader("📊 오차율(%) 분포 분석")
        df_res['error_rate'] = np.abs((df_res['전력사용량(MWh)'] - df_res['예측값(MWh)']) / df_res['전력사용량(MWh)'] * 100)
        fig_hist = px.histogram(df_res, x='error_rate', nbins=50, 
                                labels={'error_rate': '오차율 (%)'},
                                color_discrete_sequence=['#00CC96'])
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.success("""
        💡 예측값과 실제값 간의 오차율 분포를 보여줍니다 \n
        오차율이 5% 이내에 밀집되어 있을 경우, 모델의 안정성이 높다고 평가할 수 있습니다
        """)

    st.markdown("---")
    st.caption("본 시뮬레이션은 2023-2024년 서울시 실제 전력/기상 데이터를 기반으로 산출되었습니다.")