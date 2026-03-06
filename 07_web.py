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
model = joblib.load(r'C:\Users\82104\Desktop\seoul-energy-dashboard\xgb_power_model_final.pkl')

with open(r'C:\Users\82104\Desktop\seoul-energy-dashboard\features.json', 'r', encoding='utf-8') as f:
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
    base_path = "df_final_v2.csv"
    predict_path = "df_predicted.csv"
    
    if not os.path.exists(base_path):
        base_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_final_v2_refined.csv"
    if not os.path.exists(predict_path):
        predict_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_predicted.csv"
        
    df = pd.read_csv(base_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df_res = pd.read_csv(predict_path)
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    
    return df, df_res

df, df_res = load_all_data()

# ── 탄소 배출 계수 (한국 전력 평균: 0.4594 kgCO₂/kWh = 459.4 tCO₂/GWh) ──
CARBON_FACTOR = 0.4594  # kgCO₂/kWh

# 지표 사전 계산
actual = df_res['전력사용량(MWh)']
predicted = df_res['예측값(MWh)']
calc_r2 = r2_score(actual, predicted)
calc_mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# 2. 사이드바 디자인
st.sidebar.markdown("### 🛰️ SYSTEM CONTROL") 
menu = st.sidebar.radio("Menu", ["⚡ 전력 수요 시뮬레이션", "🔍 데이터 분석"])

st.sidebar.markdown("---")
st.sidebar.header("🌡️ 기상 시나리오 설정")
s_temp = st.sidebar.slider("예상 기온 (°C)", -10.0, 40.0, 25.0, 0.5)
s_hum = st.sidebar.slider("예상 습도 (%)", 0, 100, 50, 5)
s_wind  = st.sidebar.slider("예상 풍속 (m/s)", 0.0, 15.0, 2.0, 0.5)
s_hour  = st.sidebar.slider("예상 시간 (시)",  0, 23, 14, 1)

# ── 시스템 자동 계산 변수 (나머지) ──────────────────────────────────────
now = datetime.datetime.now()
s_month = now.month
is_weekend = 1 if now.weekday() >= 5 else 0
is_holiday = 0
is_off     = is_weekend
s_day = "주말/공휴일" if is_weekend == 1 else "평일"

# ── 파생값 계산 ───────────────────────────────────────────────────────
# 풍속 반영 체감온도 (체감온도 공식)
s_sensory    = 13.12 + 0.6215 * s_temp - 11.37 * (s_wind ** 0.16) + 0.3965 * s_temp * (s_wind ** 0.16)
s_discomfort = 0.81 * s_temp + 0.01 * s_hum * (0.99 * s_temp - 14.3) + 46.3
is_weekend   = 1 if s_day in ["주말", "공휴일"] else 0
is_holiday   = 1 if s_day == "공휴일" else 0
is_off       = 1 if s_day in ["주말", "공휴일"] else 0

# ── 기본값 초기화 ─────────────────────────────────────────────────────
real_sensory  = 0.0
similar_date  = "-"
predicted_mwh = 0.0
target_row    = None

base_load        = 4500
temp_effect      = max(0, s_temp - 18) * 269.8
hum_effect       = max(0, s_hum - 50) * 15
discomfort_bonus = 1.15 if s_discomfort >= 80 else 1.0
fallback_mwh     = (base_load + temp_effect + hum_effect) * discomfort_bonus

# ── 유사 패턴 탐색 및 피처 엔지니어링 ────────────────────────────────
predicted_mwh = 0.0
similar_date = "-"
s_sensory = 13.12 + 0.6215 * s_temp - 11.37 * (s_wind ** 0.16) + 0.3965 * s_temp * (s_wind ** 0.16)
s_discomfort = 0.81 * s_temp + 0.01 * s_hum * (0.99 * s_temp - 14.3) + 46.3

if not df.empty:
    # 유사 날씨/시간대 탐색 (가장 중요한 변수 위주로 거리 계산)
    df['diff'] = (
        np.abs(df['기온(°C)'] - s_temp) +
        np.abs(df['hour'] - s_hour) * 2 +
        np.abs(df['month'] - s_month) * 0.5
    )
    target_idx = df['diff'].idxmin()
    
    input_row = df.loc[target_idx].copy()
    similar_date = input_row['datetime'].strftime('%Y-%m-%d %H시')
    
    # 사용자가 변경한 값만 업데이트
    input_row['기온(°C)'] = s_temp
    input_row['습도(%)'] = s_hum
    input_row['풍속(m/s)'] = s_wind
    input_row['hour'] = s_hour
    input_row['month'] = s_month
    input_row['체감온도'] = s_sensory
    input_row['불쾌지수'] = s_discomfort
    
    # 요일/휴일 관련 변수 업데이트
    is_off = 1 if s_day == "주말/공휴일" else 0
    input_row['is_weekend'] = is_off
    input_row['is_holiday'] = is_off # 단순화
    input_row['is_off'] = is_off

    # 계산 피처 업데이트
    cdd = max(0, s_temp - 24)
    input_row['CDD'] = cdd
    input_row['HDD'] = max(0, 18 - s_temp)
    input_row['CDD_squared'] = cdd ** 2
    input_row['is_heatwave'] = 1 if s_temp >= 33 else 0
    input_row['temp_humidity'] = s_temp * s_hum
    input_row['is_peak_hour'] = 1 if 14 <= s_hour <= 17 else 0
    input_row['peak_CDD_interaction'] = cdd * input_row['is_peak_hour']
    input_row['peak_heatwave'] = input_row['is_heatwave'] * input_row['is_peak_hour']

    try:
        # features.json에 있는 모든 컬럼이 있는지 확인하고 모델 예측
        X_input = pd.DataFrame([input_row[features]])
        predicted_mwh = float(model.predict(X_input)[0])
    except Exception as e:
        st.error(f"예측 단계 오류: {e}")
        predicted_mwh = fallback_mwh
else:
    st.warning("데이터프레임이 비어 있어 시뮬레이션을 진행할 수 없습니다.")

# 탄소 배출량 추정 (MWh → kWh 변환 후 계수 적용)
carbon_ton = predicted_mwh * 1000 * CARBON_FACTOR / 1000  # tCO₂

# 전년 동기 대비 증감률 계산
yoy_delta = None
if not df.empty:
    same_cond = df[
        (df['hour']  == s_hour) &
        (df['month'] == s_month) &
        (df['is_weekend'] == is_weekend)
    ]
    if len(same_cond) >= 2:
        avg_2023 = same_cond[same_cond['datetime'].dt.year == 2023]['전력사용량(MWh)'].mean()
        avg_2024 = same_cond[same_cond['datetime'].dt.year == 2024]['전력사용량(MWh)'].mean()
        if avg_2023 > 0:
            yoy_delta = ((avg_2024 - avg_2023) / avg_2023) * 100

# --- 페이지 1 : 전력 수요 시뮬레이션 ---
if menu == "⚡ 전력 수요 시뮬레이션":
    st.title("🔌 서울시 전력 수요 시뮬레이션")
    st.markdown("---")

    if   s_discomfort >= 80: status, color = "비상", "red"
    elif s_discomfort >= 75: status, color = "주의", "orange"
    else:                    status, color = "정상", "blue"
    
    c1, c2, c3, c4, c5, c6 = st.columns([3, 3, 2, 2, 3, 2])

    with c1:
        st.metric("예상 피크 수요", f"{predicted_mwh:,.1f} MWh")
    with c2:
        st.metric("탄소 배출 추정", f"{carbon_ton:,.1f} tCO₂")
    with c3:
        st.metric("체감온도", f"{s_sensory:.1f} °C")
    with c4:
        st.metric("불쾌지수", f"{s_discomfort:.1f}")
    with c5:
        st.metric("전년 동기 대비", f"{yoy_delta:+.1f}%" if yoy_delta else "-")
    with c6:
        st.write("**관제 상태**")
        st.markdown(f":{color}[**{status}**]")
        
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # ── 폭염 경보 게이지 ─────────────────────────────────────────
        st.subheader("🌡️ 폭염 경보 게이지")
        gauge_val = min(s_temp, 40)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            title={'text': "현재 기온 (°C)"},
            gauge={
                'axis': {'range': [-10, 40]},
                'bar': {'color': "darkred" if s_temp >= 33 else "steelblue"},
                'steps': [
                    {'range': [-10, 18], 'color': '#AED6F1'},
                    {'range': [18, 28],  'color': '#A9DFBF'},
                    {'range': [28, 33],  'color': '#F9E79F'},
                    {'range': [33, 40],  'color': '#F1948A'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 33
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=60, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
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
    
    # 컬럼을 나누어 차트와 기존 설명글을 가로로 배치
    col_pie_l, col_pie_r = st.columns([1, 1.2])
    
    with col_pie_l:
        total_effect = base_load + temp_effect + hum_effect
        pie_values = [base_load/total_effect, temp_effect/total_effect, hum_effect/total_effect]
        pie_labels = ['🏠 기본 전력 수요', '❄️ 냉방 영향도', '💦 습도 영향 계수']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, 
            values=pie_values, 
            hole=.4, 
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96']),
            textinfo='percent',
            textfont_size=18,
            insidetextorientation='radial',
        )])
        
        fig_pie.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0) # 오른쪽에 범례 표시
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_pie_r:
        st.success(f"""
        - **🏠 기본 전력 수요**: 기상 무관, 24시간 일정 소비 전력
        - **❄️ 냉방 영향도**: 기온 변화가 냉방 전력에 미치는 정도
        - **💦 습도 영향 계수**: 불쾌지수 변화가 전력 소모에 미치는 영향
        """)
                
        st.write("---")
        st.write("**🧪 예측 모델 성능 지표**")
        m1, m2 = st.columns(2)
        calc_r2_pct = calc_r2 * 100
        m1.metric("결정계수 (R²)", f"{calc_r2:.4f}") 
        m2.metric("평균 오차율 (MAPE)", f"{calc_mape:.2f}%")
        st.info(f"""
        **💡 분석 결과 요약**  \n 본 모델은 서울시 전력 수요 변동의 **{calc_r2_pct:.2f}% 이상** 정확히 설명합니다  \n 실제 사용량 대비 예측 오차율이 **평균 {calc_mape:.2f}% 미만**으로 제어되고 있어 신뢰도가 매우 높습니다.
    """)                   


# --- [페이지 2: AI 모델 진단 리포트 내부] ---
else:
    st.title("🧠 AI 모델 정밀 진단 및 예측 리포트")
    st.markdown("---")
    
    top_left, top_right = st.columns([1, 1])

    with top_left:
        # ── 월별 평균 수요 바차트 ─────────────────────────────────────
        st.subheader("📅 월별 평균 전력 수요")
        if not df.empty:
            monthly = df.groupby(df['datetime'].dt.month)['전력사용량(MWh)'].mean().reset_index()
            monthly.columns = ['월', '평균수요']
            colors = ['#F1948A' if m in [7, 8] else '#AED6F1' if m in [1, 2] else '#A9DFBF' for m in monthly['월']]
            fig_bar = go.Figure(go.Bar(
                x=[f"{m}월" for m in monthly['월']],
                y=monthly['평균수요'],
                marker_color=colors,
                text=[f"{v:,.0f}" for v in monthly['평균수요']],
                textposition='outside'
            ))
            # 현재 선택 월 강조
            fig_bar.add_vline(x=s_month - 1, line_dash='dot', line_color='orange', line_width=2)
            fig_bar.update_layout(
                height=300, margin=dict(t=20, b=20),
                yaxis_title="평균 전력수요 (MWh)",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with top_right:
        # ── 24시간 수요 패턴 ─────────────────────────────────────────
        st.subheader("🕐 24시간 수요 패턴")
        if not df.empty:
            hourly = df[
                (df['datetime'].dt.month == s_month) &
                (df['is_weekend'] == is_weekend)
            ].groupby('hour')['전력사용량(MWh)'].mean().reset_index()

            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Scatter(
                x=hourly['hour'], y=hourly['전력사용량(MWh)'],
                mode='lines+markers', name='평균 수요',
                line=dict(color='steelblue', width=2),
                marker=dict(size=5)
            ))
            # 현재 시간 하이라이트
            if s_hour in hourly['hour'].values:
                cur_val = hourly[hourly['hour'] == s_hour]['전력사용량(MWh)'].values[0]
                fig_hourly.add_trace(go.Scatter(
                    x=[s_hour], y=[cur_val],
                    mode='markers+text',
                    text=[f"{cur_val:,.0f}"],
                    textposition='top center',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='현재 시간'
                ))
            fig_hourly.update_layout(
                height=300, margin=dict(t=20, b=20),
                xaxis_title="시간", yaxis_title="평균 전력수요 (MWh)",
                xaxis=dict(tickmode='linear', dtick=2)
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

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
    st.subheader("📅 최근 7일간의 예측 정밀도 추이")
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