#!/usr/bin/env python3
"""
현실적인 시뮬레이션 기반 PPO 에이전트 평가
"""

import sys
import os
# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from utils.data_loader import load_restaurants
from evaluation.metrics import RecommendationEvaluator
from evaluation.realistic_simulation import RealisticUserSimulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("🚀 현실적인 시뮬레이션 기반 PPO 평가 시작")
    print("=" * 60)
    
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
    print("🤖 PPO 에이전트 초기화...")
    agent = PPOAgent(
        n_users=n_users,
        n_items=n_items,
        context_dim=13,  # 동적 컨텍스트 차원
        learning_rate=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )
    
    # 3. 현실적인 사용자 시뮬레이터 초기화
    print("🎭 현실적인 사용자 시뮬레이터 초기화...")
    user_simulator = RealisticUserSimulator(restaurants, feature_cols, n_users)
    
    # 4. 평가자 초기화
    print("📋 평가 시스템 초기화...")
    evaluator = RecommendationEvaluator(restaurants, feature_cols)
    
    # 5. 결과 저장 경로 설정
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if os.path.exists(results_dir):
        print("🗑️ 이전 평가 결과 삭제 중...")
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # 6. 현실적인 시뮬레이션 실행
    print("\n🎬 현실적인 시뮬레이션 실행...")
    interactions = user_simulator.generate_realistic_interactions(
        agent, n_episodes=200
    )
    
    # 7. 시뮬레이션 품질 분석
    simulation_stats = user_simulator.evaluate_simulation_quality(interactions)
    
    # 8. 추천 성능 평가
    print("\n🔍 추천 성능 평가...")
    all_metrics, avg_metrics = evaluator.evaluate_agent_performance(
        agent, n_users=n_users, n_recommendations=10
    )
    
    # 9. 상호작용 데이터 분석
    print("\n📈 상호작용 데이터 분석...")
    analyze_interactions(interactions, results_dir)
    
    # 10. 시각화 생성
    print("\n📊 시각화 생성...")
    try:
        # 기본 평가 그래프
        evaluator.plot_evaluation_results(
            os.path.join(results_dir, 'evaluation_results.png')
        )
        
        # 시뮬레이션 특화 그래프
        plot_simulation_analysis(interactions, results_dir)
        
        print("✅ 모든 그래프 저장 완료")
    except Exception as e:
        print(f"⚠️ 그래프 생성 실패: {e}")
    
    # 11. 종합 보고서 생성
    print("\n📝 종합 보고서 생성...")
    generate_comprehensive_report(
        agent, simulation_stats, avg_metrics, interactions, results_dir
    )
    
    # 12. 결과 요약 출력
    print("\n🎉 평가 완료!")
    print("=" * 60)
    print_summary(simulation_stats, avg_metrics)
    
    print(f"\n📁 생성된 파일:")
    print(f"  - {results_dir}/evaluation_results.png (기본 평가)")
    print(f"  - {results_dir}/simulation_analysis.png (시뮬레이션 분석)")
    print(f"  - {results_dir}/comprehensive_report.txt (종합 보고서)")
    print(f"  - {results_dir}/interactions.csv (상호작용 데이터)")

def analyze_interactions(interactions, results_dir):
    """
    상호작용 데이터 분석
    """
    # DataFrame으로 변환
    df = pd.DataFrame(interactions)
    
    # CSV 저장
    df.to_csv(os.path.join(results_dir, 'interactions.csv'), index=False)
    
    # 기본 통계
    print(f"  총 상호작용: {len(df)}")
    print(f"  클릭률: {df['clicked'].mean():.3f}")
    print(f"  평균 보상: {df['reward'].mean():.3f}")
    
    # 사용자 타입별 성과
    type_stats = df.groupby('user_type').agg({
        'clicked': 'mean',
        'reward': 'mean'
    }).round(3)
    
    print(f"\n  사용자 타입별 성과:")
    for user_type, stats in type_stats.iterrows():
        print(f"    {user_type:10s}: 클릭률 {stats['clicked']:.3f}, 보상 {stats['reward']:.3f}")

def plot_simulation_analysis(interactions, results_dir):
    """
    시뮬레이션 분석 그래프 생성
    """
    df = pd.DataFrame(interactions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Realistic Simulation Analysis', fontsize=16, fontweight='bold')
    
    # 1. 시간에 따른 성능 변화
    window_size = 20
    df['reward_rolling'] = df['reward'].rolling(window=window_size).mean()
    df['click_rolling'] = df['clicked'].rolling(window=window_size).mean()
    
    axes[0, 0].plot(df['episode'], df['reward_rolling'], label='Reward', color='blue')
    axes[0, 0].set_title('Performance Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Rolling Average Reward')
    axes[0, 0].legend()
    
    # 2. 사용자 타입별 성과
    type_stats = df.groupby('user_type').agg({
        'clicked': 'mean',
        'reward': 'mean'
    })
    
    x_pos = range(len(type_stats))
    axes[0, 1].bar([x - 0.2 for x in x_pos], type_stats['clicked'], 
                   width=0.4, label='Click Rate', alpha=0.7)
    axes[0, 1].bar([x + 0.2 for x in x_pos], type_stats['reward'], 
                   width=0.4, label='Avg Reward', alpha=0.7)
    axes[0, 1].set_title('Performance by User Type')
    axes[0, 1].set_xlabel('User Type')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(type_stats.index, rotation=45)
    axes[0, 1].legend()
    
    # 3. 아이템 추천 분포
    item_counts = df['item_id'].value_counts().head(10)
    axes[1, 0].bar(range(len(item_counts)), item_counts.values)
    axes[1, 0].set_title('Top 10 Recommended Items')
    axes[1, 0].set_xlabel('Item Rank')
    axes[1, 0].set_ylabel('Recommendation Count')
    
    # 4. 클릭률 vs 보상 분포
    clicked_df = df[df['clicked'] == True]
    axes[1, 1].hist(clicked_df['reward'], bins=20, alpha=0.7, color='green')
    axes[1, 1].set_title('Reward Distribution (Clicked Items)')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'simulation_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(agent, simulation_stats, avg_metrics, interactions, results_dir):
    """
    종합 보고서 생성
    """
    report = []
    report.append("=" * 80)
    report.append("현실적인 시뮬레이션 기반 PPO 에이전트 평가 보고서")
    report.append("=" * 80)
    report.append(f"평가 일시: {pd.Timestamp.now()}")
    report.append("")
    
    # 시뮬레이션 설정
    report.append("📋 시뮬레이션 설정:")
    report.append("-" * 40)
    report.append(f"총 에피소드 수: {len(interactions)}")
    report.append(f"사용자 수: {agent.n_users}")
    report.append(f"아이템 수: {agent.n_items}")
    report.append(f"사용자 타입: 5가지 (foodie, budget, romantic, family, business)")
    report.append(f"컨텍스트 요소: 시간대, 날씨, 동반자, 예산")
    report.append("")
    
    # 시뮬레이션 결과
    report.append("🎬 시뮬레이션 결과:")
    report.append("-" * 40)
    report.append(f"전체 클릭률: {simulation_stats['click_rate']:.3f}")
    report.append(f"평균 보상: {simulation_stats['avg_reward']:.3f}")
    report.append("")
    
    # 사용자 타입별 성과
    report.append("👥 사용자 타입별 성과:")
    report.append("-" * 40)
    df = pd.DataFrame(interactions)
    type_stats = df.groupby('user_type').agg({
        'clicked': 'mean',
        'reward': 'mean'
    })
    
    for user_type, stats in type_stats.iterrows():
        report.append(f"{user_type:12s}: 클릭률 {stats['clicked']:.3f}, 평균 보상 {stats['reward']:.3f}")
    report.append("")
    
    # PPO 학습 성과
    report.append("🤖 PPO 학습 성과:")
    report.append("-" * 40)
    eval_stats = agent.evaluate()
    report.append(f"Policy Loss: {eval_stats['policy_loss']:.4f}")
    report.append(f"Value Loss: {eval_stats['value_loss']:.4f}")
    report.append(f"총 학습 횟수: {agent.learning_count}")
    report.append("")
    
    # 추천 품질 지표
    report.append("📊 추천 품질 지표:")
    report.append("-" * 40)
    report.append(f"다양성 (Diversity): {avg_metrics['diversity']:.3f}")
    report.append(f"신규성 (Novelty): {avg_metrics['novelty']:.3f}")
    report.append(f"커버리지 (Coverage): {avg_metrics['coverage']:.3f}")
    report.append("")
    
    # 시뮬레이션 품질 평가
    report.append("🔍 시뮬레이션 품질 평가:")
    report.append("-" * 40)
    report.append("✅ 현실적인 요소들:")
    report.append("  - 다양한 사용자 타입 (5가지)")
    report.append("  - 동적 컨텍스트 (시간, 날씨, 동반자, 예산)")
    report.append("  - 확률적 사용자 응답 (클릭 확률 + 만족도)")
    report.append("  - 아이템 인기도 기반 보너스")
    report.append("  - 개인별 선호도 차이 반영")
    report.append("")
    
    report.append("⚠️ 한계점:")
    report.append("  - 실제 사용자 데이터 부족")
    report.append("  - 장기적 선호도 변화 미반영")
    report.append("  - 사회적 영향 (리뷰, 평점) 미고려")
    report.append("")
    
    # 개선 제안
    report.append("💡 개선 제안:")
    report.append("-" * 40)
    if simulation_stats['click_rate'] < 0.3:
        report.append("- 클릭률이 낮음: 추천 정확도 개선 필요")
    if avg_metrics['diversity'] < 0.5:
        report.append("- 다양성 부족: 다양한 카테고리 추천 필요")
    if avg_metrics['novelty'] < 0.4:
        report.append("- 신규성 부족: 새로운 아이템 발굴 필요")
    
    report.append("")
    report.append("=" * 80)
    
    # 파일 저장
    with open(os.path.join(results_dir, 'comprehensive_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def print_summary(simulation_stats, avg_metrics):
    """
    결과 요약 출력
    """
    print("📊 평가 결과 요약:")
    print("-" * 30)
    print(f"🎯 클릭률: {simulation_stats['click_rate']:.3f}")
    print(f"⭐ 평균 보상: {simulation_stats['avg_reward']:.3f}")
    print(f"🎨 다양성: {avg_metrics['diversity']:.3f}")
    print(f"🆕 신규성: {avg_metrics['novelty']:.3f}")
    print(f"📈 커버리지: {avg_metrics['coverage']:.3f}")

if __name__ == "__main__":
    main() 