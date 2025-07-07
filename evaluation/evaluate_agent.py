#!/usr/bin/env python3
"""
PPO 에이전트 성능 평가 스크립트
"""

import sys
import os
# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from utils.data_loader import load_restaurants
from evaluation.metrics import RecommendationEvaluator
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 matplotlib 사용
import matplotlib.pyplot as plt

def main():
    print("🚀 PPO 에이전트 평가 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("📊 데이터 로딩...")
    restaurants, feature_matrix, feature_cols = load_restaurants()
    n_users = 10
    n_items = len(restaurants)
    
    print(f"사용자 수: {n_users}")
    print(f"아이템 수: {n_items}")
    print(f"특성 수: {len(feature_cols)}")
    print()
    
    # 2. 에이전트 초기화
    print("🤖 진짜 PPO 에이전트 초기화...")
    agent = PPOAgent(
        n_users=n_users,
        n_items=n_items,
        context_dim=len(feature_cols),
        learning_rate=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )
    
    # 3. 평가자 초기화
    print("📋 평가 시스템 초기화...")
    evaluator = RecommendationEvaluator(restaurants, feature_cols)
    
    # 4. 에이전트에 약간의 학습 데이터 제공 (시뮬레이션)
    print("🎯 에이전트 사전 학습 (시뮬레이션)...")
    for user_id in range(min(5, n_users)):
        context = feature_matrix[user_id % len(feature_matrix)]  # 임의 컨텍스트
        
        # 몇 개 아이템에 대해 가상의 상호작용 생성
        for _ in range(10):  # 사용자당 10개 상호작용
            item_id = agent.select_action(user_id, context)
            # 가상의 보상 (실제로는 사용자 피드백)
            reward = 1.0 if item_id < n_items // 2 else 0.5  # 앞쪽 아이템들이 더 좋다고 가정
            agent.store_experience(user_id, item_id, reward, context)
        
        # 배치 학습
        if len(agent.experience_buffer) >= 16:
            agent.update()
    
    print(f"총 경험 수: {len(agent.experience_buffer)}")
    print()
    
    # 5. 성능 평가 실행
    print("🔍 성능 평가 실행...")
    all_metrics, avg_metrics = evaluator.evaluate_agent_performance(
        agent, n_users=min(8, n_users), n_recommendations=10
    )
    
    # 6. 결과 저장 경로 설정 및 이전 결과 삭제
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    # 이전 결과 파일들 삭제
    if os.path.exists(results_dir):
        print("🗑️ 이전 평가 결과 삭제 중...")
        import shutil
        try:
            shutil.rmtree(results_dir)
            print("✅ 이전 결과 삭제 완료")
        except Exception as e:
            print(f"⚠️ 이전 결과 삭제 실패: {e}")
    
    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 결과 저장 디렉토리 생성: {results_dir}")
    
    # 7. 시각화 생성
    print("\n📊 시각화 생성...")
    try:
        fig = evaluator.plot_evaluation_results(
            os.path.join(results_dir, 'evaluation_results.png')
        )
        print("✅ 그래프 저장 완료: evaluation/results/evaluation_results.png")
    except Exception as e:
        print(f"⚠️ 그래프 생성 실패: {e}")
    
    # 8. 보고서 생성
    print("\n📝 보고서 생성...")
    try:
        report = evaluator.generate_evaluation_report(
            agent, save_path=os.path.join(results_dir, 'evaluation_report.txt')
        )
        print("✅ 보고서 저장 완료: evaluation/results/evaluation_report.txt")
    except Exception as e:
        print(f"⚠️ 보고서 생성 실패: {e}")
    
    # 9. 추가 분석
    print("\n🔬 추가 분석:")
    print("-" * 30)
    
    # 사용자별 추천 예시
    print("사용자별 상위 3개 추천:")
    for user_id in range(min(3, n_users)):
        context = feature_matrix[user_id % len(feature_matrix)]
        recommendations = agent.get_top_k_recommendations(user_id, k=3, context=context)
        
        print(f"\n사용자 {user_id}:")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            restaurant_name = restaurants.iloc[item_id]['name']
            print(f"  {rank}. {restaurant_name} (점수: {score:.3f})")
    
    # 학습 상태 분석
    print(f"\n학습 통계:")
    print(f"  경험 버퍼 크기: {len(agent.experience_buffer)}")
    print(f"  총 보상 횟수: {len(agent.training_stats['rewards'])}")
    
    # PPO 학습 통계
    eval_stats = agent.evaluate()
    print(f"  평균 보상: {eval_stats['mean_reward']:.3f}")
    print(f"  보상 표준편차: {eval_stats['std_reward']:.3f}")
    print(f"  Policy Loss: {eval_stats['policy_loss']:.4f}")
    print(f"  Value Loss: {eval_stats['value_loss']:.4f}")
    
    # 10. 모델 저장
    print("\n💾 모델 저장...")
    try:
        model_path = os.path.join(results_dir, 'ppo_model')
        agent.save_model(model_path)
        print(f"✅ 모델 저장 완료: {model_path}.pth")
    except Exception as e:
        print(f"⚠️ 모델 저장 실패: {e}")
    
    print("\n🎉 평가 완료!")
    print("=" * 50)
    print("📁 생성된 파일:")
    print(f"  - {os.path.join(results_dir, 'evaluation_results.png')} (시각화)")
    print(f"  - {os.path.join(results_dir, 'evaluation_report.txt')} (보고서)")
    print(f"  - {os.path.join(results_dir, 'ppo_model.pth')} (모델)")
    print("\n💡 파일을 확인해보세요!")

if __name__ == "__main__":
    main() 