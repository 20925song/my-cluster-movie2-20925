import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 헤더 표시
st.title("🎬 영화 유형 나누기 (K-Means Clustering)")
st.write("KOBIS 영화 데이터를 활용하여 머신러닝(K-Means) 기반으로 영화의 유형을 군집화합니다.")

# 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_raw_count = len(df)
    
    # 필수 컬럼 결측치 및 첫 주 관객 0인 데이터 제거
    required_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi', 'movieNm']
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean['first_week_audi'] > 0].copy()
    
    # 속성 가공
    # 1. 스크린 수 상용로그 (log10(first_scrn + 1) 처리로 safe log)
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'] + 1)
    
    # 2. 누적 관객 상용로그
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'] + 1)
    
    # 3. 롱런 지수 = 누적 관객 / 첫 주 관객 (20 초과는 20으로 클리핑)
    long_run = df_clean['total_audi'] / df_clean['first_week_audi']
    df_clean['long_run_index'] = long_run.clip(upper=20)
    
    # 4. 10위권 일수 (기존 컬럼 그대로 사용: days_in_top10)
    
    return df, df_clean, total_raw_count

df_raw, df, total_raw_count = load_data()

# 전체 편수 및 사용 편수 안내
st.info(f"📌 **전체 데이터:** {total_raw_count}편 | **분석 대상 데이터(유효 데이터):** {len(df)}편")

st.markdown("---")

# 속성 선택
feature_dict = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 수 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys())
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 수행하려면 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_features = [feature_dict[name] for name in selected_feature_names]

# 표준화 및 K-Means 군집화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 누적 관객 수(total_audi) 평균 기준으로 묶음 이름 재정의 (㉮, ㉯, ㉰)
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
label_map = {old_cluster: new_label for old_cluster, new_label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['cluster_raw'].map(label_map)

# 군집 순서 정렬
cluster_order = ['㉮', '㉯', '㉰']

st.markdown("---")
st.subheader("📊 시각화")

# 2차원 산점도
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("2차원 산점도 X축 속성:", options=selected_feature_names, index=0)
with col2:
    y_axis = st.selectbox("2차원 산점도 Y축 속성:", options=selected_feature_names, index=min(1, len(selected_feature_names)-1))

fig_2d = px.scatter(
    df,
    x=feature_dict[x_axis],
    y=feature_dict[y_axis],
    color='cluster',
    hover_name='movieNm',
    category_orders={'cluster': cluster_order},
    color_discrete_sequence=px.colors.qualitative.Set1,
    title=f"2차원 산점도 ({x_axis} vs {y_axis})",
    labels={feature_dict[x_axis]: x_axis, feature_dict[y_axis]: y_axis, 'cluster': '묶음'}
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.subheader("🎲 3차원 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 그리려면 속성을 3개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_3d = st.selectbox("3차원 X축:", options=selected_feature_names, index=0, key="3d_x")
    with c2:
        y_3d = st.selectbox("3차원 Y축:", options=selected_feature_names, index=1, key="3d_y")
    with c3:
        z_3d = st.selectbox("3차원 Z축:", options=selected_feature_names, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_3d],
        y=feature_dict[y_3d],
        z=feature_dict[z_3d],
        color='cluster',
        hover_name='movieNm',
        category_orders={'cluster': cluster_order},
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=f"3차원 산점도 ({x_3d} vs {y_3d} vs {z_3d})",
        labels={
            feature_dict[x_3d]: x_3d,
            feature_dict[y_3d]: y_3d,
            feature_dict[z_3d]: z_3d,
            'cluster': '묶음'
        }
    )
    # 점 크기 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 묶음별 통계 표
st.subheader("📈 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = df.groupby('cluster').agg(
    편수=('movieNm', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객수_평균=('total_audi', 'mean'),
    top10일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run_index', 'mean')
).reindex(cluster_order)

# 소수점 정렬 및 서식 변경
summary_df_formatted = summary_df.copy()
summary_df_formatted['스크린수_평균'] = summary_df_formatted['스크린수_평균'].apply(lambda x: f"{x:,.1f}개")
summary_df_formatted['누적관객수_평균'] = summary_df_formatted['누적관객수_평균'].apply(lambda x: f"{x:,.0f}명")
summary_df_formatted['top10일수_평균'] = summary_df_formatted['top10일수_평균'].apply(lambda x: f"{x:,.1f}일")
summary_df_formatted['롱런지수_평균'] = summary_df_formatted['롱런지수_평균'].apply(lambda x: f"{x:,.2f}")

st.dataframe(summary_df_formatted, use_container_width=True)

# 묶음별 누적 관객 수 TOP 5 영화
st.subheader("🏆 묶음별 누적 관객 수 Top 5 영화")

cols = st.columns(3)
for idx, cluster_name in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_name}")
        top5_movies = (
            df[df['cluster'] == cluster_name]
            .sort_values(by='total_audi', ascending=False)
            .head(5)
        )
        for rank, (_, row) in enumerate(top5_movies.iterrows(), start=1):
            st.write(f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)")
