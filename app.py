# =====================================================================
# [웹 대시보드] WHO 기대수명 예측 및 과대적합/규제 비교
# =====================================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 1. 페이지 기본 설정 및 리소스 로드
# ---------------------------------------------------------------------
st.set_page_config(page_title="Life Expectancy Prediction", layout="wide")

@st.cache_resource
def load_pipelines():
    """학습된 머신러닝 파이프라인 모델을 캐싱하여 로드합니다."""
    return {
        'Linear': joblib.load('Linear_model.pkl'),
        'Poly': joblib.load('Poly_model.pkl'),
        'Ridge': joblib.load('Ridge_model.pkl')
    }

@st.cache_data
def load_metrics():
    """저장된 모델 성능 지표 데이터를 캐싱하여 로드합니다."""
    return pd.read_csv('metrics.csv')

models = load_pipelines()
metrics_df = load_metrics()

# ---------------------------------------------------------------------
# 2. 대시보드 헤더 구성
# ---------------------------------------------------------------------
st.title("🌍 WHO 기대수명 예측 대시보드")
st.markdown("다항 회귀 및 릿지(Ridge) 규제 모델의 성능을 비교하고 실시간으로 기대수명을 예측해봅니다.")
st.markdown("---")

# ---------------------------------------------------------------------
# 3. 모델 성능 비교 화면 (과대적합 관찰)
# ---------------------------------------------------------------------
st.header("📊 1. 모델 성능 비교 (과대적합 관찰)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("성능 평가지표 테이블")
    st.markdown("훈련 샘플 50개 환경에서의 지표입니다. 규제가 없는 Poly 모델의 붕괴를 확인하세요.")
    st.dataframe(metrics_df, use_container_width=True)

with col2:
    st.subheader("Test R² 점수 비교")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=metrics_df, x='Model', y='Test R²', palette='magma', ax=ax)

    # 막대 그래프 위 수치 텍스트 표시
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                    textcoords='offset points')
    st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------------------
# 4. 실시간 예측 UI 구성
# ---------------------------------------------------------------------
st.header("🔮 2. 실시간 기대수명 예측")

# 사이드바: 사용자 입력 컨트롤
st.sidebar.header("입력 변수 설정")
st.sidebar.markdown("슬라이더를 움직여 값을 조정해보세요.")

bmi = st.sidebar.slider("BMI (체질량 지수)", 1.0, 90.0, 40.0)
gdp = st.sidebar.slider("1인당 GDP (USD)", 1, 120000, 5000)
alcohol = st.sidebar.slider("알코올 소비량 (리터)", 0.0, 20.0, 5.0)
polio = st.sidebar.slider("Polio (소아마비 예방접종률 %)", 1.0, 99.0, 80.0)

selected_model_name = st.selectbox(
    "예측에 사용할 머신러닝 모델을 선택하세요:",
    ['Linear', 'Poly', 'Ridge']
)

# ---------------------------------------------------------------------
# 5. 실시간 동적 예측 수행 및 결과 출력
# ---------------------------------------------------------------------
# 사용자 입력을 DataFrame으로 변환
input_df = pd.DataFrame({
    'BMI': [bmi],
    'GDP': [gdp],
    'Alcohol': [alcohol],
    'Polio': [polio]
})

# 선택된 파이프라인으로 예측
selected_pipeline = models[selected_model_name]
prediction_result = selected_pipeline.predict(input_df)[0]

# 결과 화면 렌더링
st.success(f"현재 동작 중인 모델: **{selected_model_name}**")
st.markdown(f"""
<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;">
    <h2 style="color:#1f77b4;margin:0;">예상 기대수명</h2>
    <h1 style="color:#d62728;font-size:50px;margin:0;">{prediction_result:.2f} 세</h1>
</div>
""", unsafe_allow_html=True)
