import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="서울시 전력 수요 시뮬레이터")

# 2. 데이터 로드
@st.cache_data
def load_data():
    path = path = "df_final_v2.csv"
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

df = load_data()

# 3. 사이드바: 시뮬레이션 설정
st.sidebar.header("🌡️ 기상 시나리오 설정")
st.sidebar.write("기상 조건을 조절해 전력 수요 변화를 예측해보세요.")

s_temp = st.sidebar.slider("예상 기온 (°C)", -10.0, 40.0, 25.0, 0.5)
s_hum = st.sidebar.slider("예상 습도 (%)", 0, 100, 50, 5)

# 🧪 시뮬레이션 수식 (기존 로직 유지)
s_discomfort = 0.81 * s_temp + 0.01 * s_hum * (0.99 * s_temp - 14.3) + 46.3

base_load = 4500 
temp_effect = max(0, s_temp - 18) * 269.8 
hum_effect = max(0, s_hum - 50) * 15 
discomfort_bonus = 1.15 if s_discomfort >= 80 else 1.0 
predicted_mwh = (base_load + temp_effect + hum_effect) * discomfort_bonus

# 4. 유사 패턴 분석 (피크 시간 예측)
df['diff'] = np.abs(df['기온(°C)'] - s_temp) + np.abs(df['습도(%)'] - s_hum) * 0.1
similar_date = df.loc[df['diff'].idxmin(), '기준일자']
similar_day_data = df[df['기준일자'] == similar_date].sort_values('기준시')
peak_hour = similar_day_data.loc[similar_day_data['전력사용량(MWh)'].idxmax(), '기준시']

# 5. 메인 화면 구성
st.title("🔌 전력 수요 시뮬레이션 & 피크 예측")
st.markdown("---")

# 상단 핵심 지표 (Metric)
c1, c2, c3, c4 = st.columns(4)
c1.metric("예상 피크 수요", f"{predicted_mwh:,.1f} MWh")
c2.metric("예상 피크 시간", f"오후 {peak_hour}시경")
c3.metric("계산된 불쾌지수", f"{s_discomfort:.1f}")

# 위험도 상태 판정
if s_discomfort >= 80: status, color = "비상", "red"
elif s_discomfort >= 75: status, color = "주의", "orange"
else: status, color = "정상", "blue"
c4.markdown(f"**현재 관제 상태:** \n ### :{color}[{status}]")

st.markdown("---")

# 6. 중앙 레이아웃 (시각화 및 리포트)
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 기상 시나리오 분석 (과거 데이터 대조)")
    fig = go.Figure()
    
    # 과거 전체 데이터 산점도 (배경)
    sample_df = df.sample(1000)
    fig.add_trace(go.Scatter(
        x=sample_df['기온(°C)'], y=sample_df['전력사용량(MWh)'], 
        mode='markers', name='과거 패턴', 
        marker=dict(color='lightgrey', opacity=0.4)
    ))
    
    # 현재 시뮬레이션 지점 (강조)
    fig.add_trace(go.Scatter(
        x=[s_temp], y=[predicted_mwh], 
        mode='markers+text', name='시뮬레이션 예측값', 
        text=["현재 예측 지점"], textposition="top center",
        marker=dict(color='red', size=15, symbol='star')
    ))
    
    fig.update_layout(
        xaxis_title="기온 (°C)", 
        yaxis_title="전력사용량 (MWh)", 
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📋 실무 분석 리포트")
    
    st.write(f"**1. 예측 근거 (유사 사례)**")
    st.info(f"현재 설정 조건은 과거 **{similar_date}**의 기상 및 전력 패턴과 가장 유사합니다.")
    
    st.write(f"**2. 시간대별 부하 특성**")
    st.write(f"과거 유사 사례 분석 결과, 해당 기상 조건에서는 **오후 {peak_hour}시**를 기점으로 전력 수요가 급증하는 경향을 보입니다.")
    
    st.write(f"**3. 단계별 대응 가이드**")
    if s_temp >= 33 or s_discomfort >= 80:
        st.error("🚨 **비상**: 전력 예비율 급락 주의. 산업용 냉방 부하 조정 및 ESS 방전 대기.")
    elif s_temp >= 30 or s_discomfort >= 75:
        st.warning("⚠️ **경계**: 피크 시간대 공공기관 냉방기기 순차 운전 정지 및 에너지 절약 권고.")
    else:
        st.success("✅ **정상**: 가용 전력 내 안정적 공급 가능. 상시 모니터링 유지.")

st.markdown("---")
st.caption("본 시뮬레이션은 2023-2024년 서울시 실제 전력/기상 데이터를 기반으로 산출되었습니다.")