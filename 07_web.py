import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from sklearn.metrics import r2_score

# 1. 페이지 설정 및 데이터 로드
st.set_page_config(layout="wide", page_title="서울시 전력 수요 관제 시스템")

@st.cache_data
def load_all_data():
    # 파일 경로 설정 (기본적으로 같은 폴더 내에 있다고 가정)
    base_path = "df_final_v2.csv"
    predict_path = "df_predicted.csv"
    
    # 로컬 경로 백업 (필요시 경로 수정)
    if not os.path.exists(base_path):
        base_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_final_v2.csv"
    if not os.path.exists(predict_path):
        predict_path = r"C:\Users\82104\Desktop\seoul-energy-dashboard\df_predicted.csv"
        
    df = pd.read_csv(base_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df_res = pd.read_csv(predict_path)
    df_res['datetime'] = pd.to_datetime(df_res['datetime'])
    
    return df, df_res
df, df_res = load_all_data()

# 지표 사전 계산 (2, 3페이지 공통 사용)
actual = df_res['전력사용량(MWh)']
predicted = df_res['예측값(MWh)']
calc_r2 = r2_score(actual, predicted)
calc_mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# 2. 사이드바 디자인
st.sidebar.markdown("### 🛰️ SYSTEM CONTROL") 
menu = st.sidebar.radio("Menu", ["⚡ 전력 수요 시뮬레이터", "🔍 상세 분석 리포트", "🧠 AI 모델 진단 리포트"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.header("🌡️ 기상 시나리오 설정")
s_temp = st.sidebar.slider("예상 기온 (°C)", -10.0, 40.0, 25.0, 0.5)
s_hum = st.sidebar.slider("예상 습도 (%)", 0, 100, 50, 5)

# [수식 로직]
s_discomfort = 0.81 * s_temp + 0.01 * s_hum * (0.99 * s_temp - 14.3) + 46.3
base_load = 4500 
temp_effect = max(0, s_temp - 18) * 269.8 
hum_effect = max(0, s_hum - 50) * 15 
discomfort_bonus = 1.15 if s_discomfort >= 80 else 1.0 
predicted_mwh = (base_load + temp_effect + hum_effect) * discomfort_bonus

# [1페이지 반영용 유사 패턴 분석] - df_res 기반으로 수정
df_res['diff'] = np.abs(df_res['기온(°C)'] - s_temp) + np.abs(df_res['습도(%)'] - s_hum) * 0.1
# 가장 유사한 날짜 추출
similar_row = df_res.loc[df_res['diff'].idxmin()]
similar_date_str = pd.to_datetime(similar_row['datetime']).strftime('%Y-%m-%d')
# 그 날의 데이터 중 피크 시간 찾기
similar_day_data = df_res[df_res['datetime'].dt.strftime('%Y-%m-%d') == similar_date_str]
real_peak_hour = similar_day_data.loc[similar_day_data['전력사용량(MWh)'].idxmax(), 'hour']

# --- 페이지 1: 전력 수요 시뮬레이터 ---
if menu == "⚡ 전력 수요 시뮬레이터":
    st.title("🔌 서울시 전력 수요 시뮬레이션")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예상 피크 수요", f"{predicted_mwh:,.1f} MWh")
    c2.metric("유사 기상 피크 시간", f"오후 {real_peak_hour}시경") # 반영 완료
    c3.metric("계산된 불쾌지수", f"{s_discomfort:.1f}")
    
    status, color = ("비상", "red") if s_discomfort >= 80 else (("주의", "orange") if s_discomfort >= 75 else ("정상", "blue"))
    c4.markdown(f"**관제 상태:** \n ### :{color}[{status}]")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("📈 기상 시나리오 분석 (과거 데이터 대조)")
        fig = go.Figure()
        sample_df = df_res.sample(min(1000, len(df_res)))
        fig.add_trace(go.Scatter(x=sample_df['기온(°C)'], y=sample_df['전력사용량(MWh)'], mode='markers', name='과거 패턴', marker=dict(color='lightgrey', opacity=0.4)))
        fig.add_trace(go.Scatter(x=[s_temp], y=[predicted_mwh], mode='markers+text', name='시뮬레이션 예측값', text=["현재 예측 지점"], textposition="top center", marker=dict(color='red', size=15, symbol='star')))
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("📋 실무 분석 리포트")
        st.info(f"선택하신 조건은 데이터 상 **{similar_date_str}**과 가장 유사합니다.")
        st.info(f"해당 기상 조건에서 AI 모델은 **오후 {real_peak_hour}시**에 가장 높은 부하를 기록했습니다.")

# --- 페이지 2: 상세 분석 리포트 ---
elif menu == "🔍 상세 분석 리포트":
    st.title("🔍 데이터 분석 상세 리포트")
    st.markdown("---")
    b_left, b_right = st.columns([1, 1])
    with b_left:
        st.subheader("🕵️ 주요 변수 기여도 (가중치 분석)")
        total_effect = base_load + temp_effect + hum_effect
        pie_values = [base_load/total_effect, temp_effect/total_effect, hum_effect/total_effect]
        pie_labels = ['🏠 상시 전력 수요', '❄️ 냉방 부하 계수', '💦 습도 보정 계수']
        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=.4, marker=dict(colors=['#636EFA', '#EF553B', '#00CC96']))])
        st.plotly_chart(fig_pie, use_container_width=True)
    with b_right:
        st.subheader("💡 항목별 상세 가이드")
        st.markdown(f"""
        - **🏠 상시 전력 수요**: 기상 변화와 무관하게 24시간 소비되는 고정 전력 소비
        - **❄️ 냉방 부하 계수**: 기온 상승에 따른 냉방 설비 가동 전력 수요
        - **💦 습도 보정 계수**: 불쾌지수 변화에 따른 전력 소모 보정치
        """)
        st.subheader("🧪 실측 모델 성능 지표") # 고정값에서 계산값으로 변경 반영
        m1, m2 = st.columns(2)
        m1.metric("결정계수 (R²)", f"{calc_r2:.4f}") 
        m2.metric("평균 오차율 (MAPE)", f"{calc_mape:.2f}%")
        st.success(f"현재 업로드된 {len(df_res):,}건의 데이터를 분석한 결과입니다.")
        st.success("""
        **💡 분석 결과 요약** \n
        본 모델은 서울시 전력 수요 변동 **99% 이상** 정확히 설명하며,  
        실제 수요와의 오차가 **평균 2% 미만**인 높은 신뢰도 분석 결과를 제공합니다.
        """)


# --- [페이지 3: AI 모델 진단 리포트 내부] ---
else:
    st.title("🧠 AI 모델 정밀 진단 및 예측 리포트")
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