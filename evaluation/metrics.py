# Evaluation Metrics 구현 파일 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import os
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트를 찾아서 설정 시도
try:
    # 시스템에서 사용 가능한 한글 폰트 찾기
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_fonts = []
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            if any(keyword in font_name.lower() for keyword in ['nanum', 'malgun', 'batang', 'dotum', 'gulim']):
                korean_fonts.append(font_name)
        except:
            continue
    
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts[0]
        print(f"한글 폰트 설정 완료: {korean_fonts[0]}")
    else:
        print("한글 폰트를 찾을 수 없어 기본 폰트 사용")
        
except Exception as e:
    print(f"폰트 설정 중 오류: {e}, 기본 폰트 사용")

class RecommendationEvaluator:
    """
    추천 시스템 평가 클래스 (정답 데이터 없이도 평가 가능)
    """
    def __init__(self, restaurants_df, feature_cols):
        self.restaurants_df = restaurants_df
        self.feature_cols = feature_cols
        self.evaluation_history = []
        
    def evaluate_recommendations(self, user_id, recommendations, user_interactions=None):
        """
        추천 결과 종합 평가
        """
        metrics = {}
        
        # 1. 다양성 평가
        metrics['diversity'] = self.calculate_diversity(recommendations)
        
        # 2. 커버리지 평가
        metrics['coverage'] = self.calculate_coverage(recommendations)
        
        # 3. 신규성 평가
        metrics['novelty'] = self.calculate_novelty(recommendations, user_interactions)
        
        # 4. 카테고리 분포 평가
        metrics['category_distribution'] = self.calculate_category_distribution(recommendations)
        
        # 5. 평균 점수
        metrics['avg_score'] = np.mean([score for _, score in recommendations])
        
        # 평가 기록 저장
        evaluation_record = {
            'user_id': user_id,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'recommendations': recommendations
        }
        self.evaluation_history.append(evaluation_record)
        
        return metrics
    
    def calculate_diversity(self, recommendations):
        """
        추천 결과의 다양성 계산 (카테고리 기반)
        """
        if len(recommendations) == 0:
            return 0.0
            
        # 추천된 아이템들의 특성 벡터 가져오기
        item_features = []
        for item_id, _ in recommendations:
            if item_id < len(self.restaurants_df):
                features = self.restaurants_df.iloc[item_id][self.feature_cols].values
                item_features.append(features)
        
        if len(item_features) < 2:
            return 0.0
        
        # 아이템 간 평균 거리 계산 (코사인 거리)
        distances = []
        for i in range(len(item_features)):
            for j in range(i+1, len(item_features)):
                # 코사인 유사도 계산
                dot_product = np.dot(item_features[i], item_features[j])
                norm_i = np.linalg.norm(item_features[i])
                norm_j = np.linalg.norm(item_features[j])
                
                if norm_i > 0 and norm_j > 0:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    cosine_distance = 1 - cosine_sim
                    distances.append(cosine_distance)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_coverage(self, recommendations):
        """
        커버리지 계산 (전체 아이템 중 추천된 비율)
        """
        total_items = len(self.restaurants_df)
        recommended_items = set([item_id for item_id, _ in recommendations])
        return len(recommended_items) / total_items
    
    def calculate_novelty(self, recommendations, user_interactions=None):
        """
        신규성 계산 (인기도 기반)
        """
        if user_interactions is None:
            # 인기도 데이터가 없으면 랜덤하게 생성
            popularity_scores = np.random.exponential(0.1, len(self.restaurants_df))
        else:
            # 실제 상호작용 데이터로 인기도 계산
            item_counts = Counter([interaction['item_id'] for interaction in user_interactions])
            popularity_scores = [item_counts.get(i, 0) for i in range(len(self.restaurants_df))]
        
        # 추천된 아이템들의 평균 신규성 (인기도 역수)
        novelty_scores = []
        for item_id, _ in recommendations:
            if item_id < len(popularity_scores):
                # 인기도가 낮을수록 신규성이 높음
                novelty = 1.0 / (1.0 + popularity_scores[item_id])
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def calculate_category_distribution(self, recommendations):
        """
        카테고리 분포 계산
        """
        category_counts = defaultdict(int)
        
        for item_id, _ in recommendations:
            if item_id < len(self.restaurants_df):
                # 가장 높은 특성값을 가진 카테고리 찾기
                features = self.restaurants_df.iloc[item_id][self.feature_cols].values
                max_feature_idx = np.argmax(features)
                category = self.feature_cols[max_feature_idx]
                category_counts[category] += 1
        
        # 분포의 균등성 계산 (엔트로피 기반)
        total = sum(category_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in category_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # 정규화 (최대 엔트로피로 나누기)
        max_entropy = np.log2(len(self.feature_cols))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def evaluate_agent_performance(self, agent, n_users=5, n_recommendations=10):
        """
        에이전트 전체 성능 평가
        """
        print("🔍 에이전트 성능 평가 시작...")
        
        all_metrics = []
        
        for user_id in range(min(n_users, agent.n_users)):
            # 사용자별 컨텍스트 생성
            context = np.random.rand(agent.context_dim)
            context /= np.sum(context)
            
            # 추천 생성
            recommendations = agent.get_top_k_recommendations(
                user_id, k=n_recommendations, context=context
            )
            
            # 평가 수행
            metrics = self.evaluate_recommendations(user_id, recommendations)
            all_metrics.append(metrics)
            
            print(f"User {user_id}:")
            print(f"  다양성: {metrics['diversity']:.3f}")
            print(f"  커버리지: {metrics['coverage']:.3f}")
            print(f"  신규성: {metrics['novelty']:.3f}")
            print(f"  카테고리 균등성: {metrics['category_distribution']:.3f}")
            print(f"  평균 점수: {metrics['avg_score']:.3f}")
            print()
        
        # 전체 평균 계산
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'category_distribution':  # dict는 제외
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print("📊 전체 평균 성능:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.3f}")
        
        return all_metrics, avg_metrics
    
    def plot_evaluation_results(self, save_path='evaluation_results.png'):
        """
        평가 결과 시각화
        """
        if not self.evaluation_history:
            print("평가 기록이 없습니다.")
            return
        
        # 데이터 준비
        metrics_df = []
        for record in self.evaluation_history:
            row = {
                'user_id': record['user_id'],
                'timestamp': record['timestamp'],
                **record['metrics']
            }
            # dict 타입 제외
            row = {k: v for k, v in row.items() if not isinstance(v, dict)}
            metrics_df.append(row)
        
        metrics_df = pd.DataFrame(metrics_df)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recommendation System Performance Evaluation', fontsize=16, fontweight='bold')
        
        # 1. 다양성 분포
        axes[0, 0].hist(metrics_df['diversity'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Diversity Distribution')
        axes[0, 0].set_xlabel('Diversity Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 커버리지 vs 신규성
        axes[0, 1].scatter(metrics_df['coverage'], metrics_df['novelty'], alpha=0.6, color='orange')
        axes[0, 1].set_title('Coverage vs Novelty')
        axes[0, 1].set_xlabel('Coverage')
        axes[0, 1].set_ylabel('Novelty')
        
        # 3. 시간에 따른 평균 점수
        if len(metrics_df) > 1:
            metrics_df = metrics_df.sort_values('timestamp')
            axes[1, 0].plot(range(len(metrics_df)), metrics_df['avg_score'], marker='o', color='green')
            axes[1, 0].set_title('Average Score Over Time')
            axes[1, 0].set_xlabel('Evaluation Order')
            axes[1, 0].set_ylabel('Average Score')
        
        # 4. 사용자별 성능 비교
        user_performance = metrics_df.groupby('user_id')[['diversity', 'novelty', 'coverage']].mean()
        user_performance.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Performance by User')
        axes[1, 1].set_xlabel('User ID')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 저장
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            os.makedirs('.', exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 메모리 해제
        print(f"📊 평가 결과 그래프 저장: {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, agent, save_path='evaluation_report.txt'):
        """
        평가 보고서 생성
        """
        print("📝 평가 보고서 생성 중...")
        
        # 성능 평가 실행
        all_metrics, avg_metrics = self.evaluate_agent_performance(agent)
        
        # 보고서 작성
        report = []
        report.append("=" * 60)
        report.append("추천 시스템 성능 평가 보고서")
        report.append("=" * 60)
        report.append(f"평가 일시: {pd.Timestamp.now()}")
        report.append(f"평가 대상: {type(agent).__name__}")
        report.append(f"사용자 수: {agent.n_users}")
        report.append(f"아이템 수: {agent.n_items}")
        report.append(f"컨텍스트 차원: {agent.context_dim}")
        report.append("")
        
        report.append("📊 전체 평균 성능:")
        report.append("-" * 30)
        for key, value in avg_metrics.items():
            report.append(f"{key:20}: {value:.4f}")
        report.append("")
        
        report.append("📈 성능 해석:")
        report.append("-" * 30)
        
        # 다양성 해석
        diversity_score = avg_metrics['diversity']
        if diversity_score > 0.7:
            report.append("✅ 다양성: 우수 - 다양한 카테고리의 아이템을 추천하고 있습니다.")
        elif diversity_score > 0.4:
            report.append("⚠️ 다양성: 보통 - 추천 다양성을 개선할 여지가 있습니다.")
        else:
            report.append("❌ 다양성: 낮음 - 비슷한 아이템들만 추천하고 있습니다.")
        
        # 신규성 해석
        novelty_score = avg_metrics['novelty']
        if novelty_score > 0.6:
            report.append("✅ 신규성: 우수 - 새로운 아이템들을 잘 발굴하고 있습니다.")
        elif novelty_score > 0.3:
            report.append("⚠️ 신규성: 보통 - 인기 아이템과 신규 아이템의 균형이 필요합니다.")
        else:
            report.append("❌ 신규성: 낮음 - 인기 아이템에만 치중되어 있습니다.")
        
        # 커버리지 해석
        coverage_score = avg_metrics['coverage']
        if coverage_score > 0.5:
            report.append("✅ 커버리지: 우수 - 전체 아이템의 절반 이상을 활용하고 있습니다.")
        elif coverage_score > 0.2:
            report.append("⚠️ 커버리지: 보통 - 더 많은 아이템을 추천에 활용할 수 있습니다.")
        else:
            report.append("❌ 커버리지: 낮음 - 제한된 아이템들만 추천하고 있습니다.")
        
        report.append("")
        report.append("💡 개선 제안:")
        report.append("-" * 30)
        
        if diversity_score < 0.5:
            report.append("- 다양성 개선: 서로 다른 카테고리의 아이템을 균형있게 추천")
        if novelty_score < 0.4:
            report.append("- 신규성 개선: 인기도가 낮은 새로운 아이템도 추천에 포함")
        if coverage_score < 0.3:
            report.append("- 커버리지 개선: 더 많은 아이템을 추천 후보로 고려")
        
        report.append("")
        report.append("=" * 60)
        
        # 파일 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📝 평가 보고서 저장: {save_path}")
        
        # 터미널에도 출력
        print('\n'.join(report))
        
        return '\n'.join(report) 