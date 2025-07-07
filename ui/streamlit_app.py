# Streamlit UI 구현 파일 
import time
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_restaurants, init_logs
from utils.similarity import compute_similarity
from envs.restaurant_env import RestaurantRecEnv
from agents.ppo_agent import SimplePPOAgent

st.set_page_config(page_title='간단한 PPO 추천', layout='wide')

# ── 데이터 로드 및 초기화 ─────────────────────────────────────────
restaurants, feature_matrix, feature_cols = load_restaurants()
n_users = 10
n_items = len(restaurants)
context_dim = len(feature_cols)  # feature 차원 수

# 1) env/agent 세션 저장
if 'env' not in st.session_state:
    st.session_state.env = RestaurantRecEnv(n_users, n_items)
    st.session_state.agent = SimplePPOAgent(n_users, n_items, context_dim=context_dim)
env = st.session_state.env
agent = st.session_state.agent

# 2) 유저 선호 벡터 및 카운트 초기화
if 'user_vecs' not in st.session_state:
    np.random.seed(42)  # 재현성을 위해 시드 설정
    user_vecs = {}
    for uid in range(n_users):
        vec = np.random.rand(len(feature_cols))
        vec /= np.sum(vec)
        user_vecs[uid] = vec
    st.session_state.user_vecs = user_vecs
if 'user_counts' not in st.session_state:
    st.session_state.user_counts = {uid: 0 for uid in range(n_users)}

if 'training_step' not in st.session_state:
    st.session_state.training_step = 0

# 3) 로그 파일 경로 보장
log_path = init_logs()

# 4) state 초기화
if 'state' not in st.session_state:
    st.session_state.state = 0  # 기본값으로 user_id 0 설정

# ── 메인 인터페이스 ───────────────────────────────────────────────
st.title('PPO 기반 식당 추천 시스템')

# ── 사이드바 사용자 선택 ────────────────────────────────────────────
st.sidebar.header('🎛️ 시스템 설정')
user_id = st.sidebar.selectbox('👤 사용자 선택', range(n_users), format_func=lambda x: f"User {x}")

# PPO 하이퍼파라미터 표시
st.sidebar.subheader('PPO 설정')
st.sidebar.write(f"사용자 수: {n_users}")
st.sidebar.write(f"아이템 수: {n_items}")
st.sidebar.write(f"Context 차원: {context_dim}")
st.sidebar.write(f"경험 버퍼: {len(agent.experience_buffer)}")

if 'state' not in st.session_state or st.session_state.get('last_user') != user_id:
    st.session_state.state = env.reset(user_id)
    st.session_state.last_user = user_id

# 선호도 가중치 표시
st.sidebar.subheader('현재 선호도 가중치')
current_vec = st.session_state.user_vecs[user_id]
# 상위 5개 특성만 표시
pref_df = pd.DataFrame({'feature': feature_cols, 'weight': current_vec})
pref_df = pref_df.sort_values('weight', ascending=False).head(5)
st.sidebar.dataframe(pref_df, use_container_width=True)

# ── SB3-PPO 추천 생성 ────────────────────────────────────────────────
user_vec = st.session_state.user_vecs[user_id]
context = user_vec  # 사용자 선호도를 컨텍스트로 사용

# PPO 추천 생성
try:
    with st.spinner("PPO 추천 생성 중..."):
        ppo_recommendations = agent.get_top_k_recommendations(user_id, k=n_items, context=context)
    
    # 결과를 DataFrame으로 변환
    final_df = restaurants.copy()
    final_df['ppo_score'] = 0.0
    
    for item_id, score in ppo_recommendations:
        if item_id < len(final_df):
            final_df.loc[final_df['item_id'] == item_id, 'ppo_score'] = score
    
    final_df = final_df.sort_values('ppo_score', ascending=False)
    

    
except Exception as e:
    st.error(f"⚠️ PPO 추천 생성 중 오류: {str(e)}")
    st.info("💡 폴백 모드: 기본 추천 시스템을 사용합니다.")
    
    # 폴백: 기본 추천 (유사도 기반)
    final_df = restaurants.copy()
    sim_scores = compute_similarity(user_vec, feature_matrix)
    final_df['ppo_score'] = sim_scores + np.random.normal(0, 0.1, len(sim_scores))
    final_df = final_df.sort_values('ppo_score', ascending=False)

# ── 추천 결과 표시 ───────────────────────────────────────────────────

# 상위 추천 카드 형태로 표시
st.subheader("🍽️ PPO 추천 식당 (상위 6개)")
cols = st.columns(3)

for idx, (_, row) in enumerate(final_df.head(6).iterrows()):
    col = cols[idx % 3]
    with col:
        with st.container():
            st.markdown(f"""
            <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; margin-bottom: 1rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
                <h4>🏪 {row['name']}</h4>
                <p><strong>순위:</strong> #{idx + 1}</p>
                <p><strong>PPO 점수:</strong> {row['ppo_score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button(f"🍽️ 선택!", key=f"select_{int(row['item_id'])}_main"):
                # 보상 계산
                reward = 1.0 + np.random.normal(0, 0.1)  # 약간의 노이즈 추가
                
                # PPO 에이전트 업데이트
                context = st.session_state.user_vecs[user_id]
                agent.store_experience(
                    user_id=user_id,
                    action=int(row['item_id']),
                    reward=reward,
                    context=context
                )
                
                # 배치가 충분히 쌓이면 업데이트
                if len(agent.experience_buffer) >= 64:
                    agent.update(total_timesteps=100)
                    st.session_state.training_step += 1
                
                # 로그 저장
                logs_df = pd.read_csv(log_path)
                logs_df.loc[len(logs_df)] = [user_id, int(row['item_id']), reward, int(time.time())]
                logs_df.to_csv(log_path, index=False)
                
                # 사용자 선호도 업데이트
                old_vec = st.session_state.user_vecs[user_id]
                count = st.session_state.user_counts[user_id]
                item_vec = feature_matrix[int(row['item_id'])]
                new_count = count + 1
                new_vec = (old_vec * count + item_vec) / new_count
                st.session_state.user_vecs[user_id] = new_vec
                st.session_state.user_counts[user_id] = new_count
                
                st.success(f"✅ {row['name']} 선택 완료! PPO 모델이 학습되었습니다.")
                st.rerun()

# ── 전체 추천 리스트 ──────────────────────────────────────────────
with st.expander("📋 전체 PPO 추천 리스트 보기"):
    st.subheader(f"전체 식당 추천 ({len(final_df)}개)")
    
    display_df = final_df[['name', 'ppo_score']].copy()
    display_df.index = range(1, len(display_df) + 1)  # 순위로 인덱스 설정
    st.dataframe(display_df, use_container_width=True)

# ── 사용자 선호도 분석 ────────────────────────────────────────────
st.subheader("👤 사용자 선호도 분석")
col1, col2 = st.columns(2)

with col1:
    # 현재 선호도 시각화
    import plotly.express as px
    current_vec = st.session_state.user_vecs[user_id]
    # 상위 10개만 표시
    top_features = pd.DataFrame({'feature': feature_cols, 'weight': current_vec}).sort_values('weight', ascending=False).head(10)
    
    fig_pref = px.bar(
        top_features,
        x='weight', 
        y='feature',
        orientation='h',
        title=f"User {user_id}의 상위 선호도 (Top 10)",
        labels={'weight': '가중치', 'feature': '특성'}
    )
    fig_pref.update_layout(height=400)
    st.plotly_chart(fig_pref, use_container_width=True)

with col2:
    # 선택 횟수
    st.metric("총 선택 횟수", st.session_state.user_counts[user_id])
    
    # PPO 훈련 상태
    training_stats = agent.get_training_stats()
    if training_stats['rewards']:
        st.metric("평균 보상", f"{np.mean(training_stats['rewards'][-10:]):.3f}")
    
    # 모델 평가 버튼
    if st.button("🔍 모델 성능 평가"):
        with st.spinner("모델 평가 중..."):
            try:
                eval_results = agent.evaluate(n_episodes=5)
                st.write("**평가 결과:**")
                st.write(f"• 평균 보상: {eval_results['mean_reward']:.3f}")
                st.write(f"• 표준편차: {eval_results['std_reward']:.3f}")
                st.write(f"• 최고 보상: {eval_results['max_reward']:.3f}")
            except Exception as e:
                st.error(f"평가 중 오류: {e}")
    
    # 최근 활동 로그
    if st.button("📊 최근 활동 보기"):
        try:
            logs_df = pd.read_csv(log_path)
            user_logs = logs_df[logs_df['user_id'] == user_id].tail(10)
            if not user_logs.empty:
                st.write("**최근 10개 활동:**")
                for _, log in user_logs.iterrows():
                    restaurant_name = restaurants[restaurants['item_id'] == log['item_id']]['name'].iloc[0]
                    st.write(f"• {restaurant_name} (보상: {log['reward']:.3f})")
            else:
                st.write("아직 활동 기록이 없습니다.")
        except Exception as e:
            st.write("로그를 불러올 수 없습니다.")

# ── PPO 훈련 시각화 ──────────────────────────────────────────────
training_stats = agent.get_training_stats()
if training_stats['rewards'] and len(training_stats['rewards']) > 1:
    st.subheader("📈 PPO 훈련 성능")
    
    # 보상 추이 그래프
    import plotly.graph_objects as go
    fig_rewards = go.Figure()
    fig_rewards.add_trace(go.Scatter(
        y=training_stats['rewards'][-50:],
        name='평균 보상',
        line=dict(color='green', width=3)
    ))
    fig_rewards.update_layout(
        title="PPO 보상 추이",
        xaxis_title="훈련 스텝",
        yaxis_title="보상",
        height=400
    )
    st.plotly_chart(fig_rewards, use_container_width=True)