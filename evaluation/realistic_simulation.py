#!/usr/bin/env python3
"""
현실적인 추천 시스템 시뮬레이션 환경
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from collections import defaultdict

class RealisticUserSimulator:
    """
    현실적인 사용자 행동 시뮬레이터
    """
    def __init__(self, restaurants_df, feature_cols, n_users=10):
        self.restaurants_df = restaurants_df
        self.feature_cols = feature_cols
        self.n_users = n_users
        self.n_items = len(restaurants_df)
        
        # 사용자 프로파일 생성
        self.user_profiles = self._create_user_profiles()
        
        # 아이템 인기도 (실제 데이터에서는 과거 클릭 수 등)
        self.item_popularity = self._calculate_item_popularity()
        
        # 시간에 따른 트렌드
        self.trend_factor = np.random.rand(self.n_items)
        
        print(f"🎭 현실적인 사용자 시뮬레이터 초기화 완료")
        print(f"   사용자 수: {n_users}")
        print(f"   아이템 수: {self.n_items}")
        print(f"   사용자 타입: {len(set(self.user_profiles.values()))}가지")
    
    def _create_user_profiles(self):
        """
        다양한 사용자 프로파일 생성
        """
        # 사용자 타입 정의
        user_types = {
            'foodie': {'가격': 0.2, '맛': 0.4, '분위기': 0.3, '서비스': 0.1},
            'budget': {'가격': 0.6, '맛': 0.2, '분위기': 0.1, '서비스': 0.1},  
            'romantic': {'가격': 0.1, '맛': 0.2, '분위기': 0.6, '서비스': 0.1},
            'family': {'가격': 0.3, '맛': 0.3, '분위기': 0.2, '서비스': 0.2},
            'business': {'가격': 0.2, '맛': 0.2, '분위기': 0.3, '서비스': 0.3}
        }
        
        # 각 사용자에게 타입 할당
        user_profiles = {}
        type_names = list(user_types.keys())
        
        for user_id in range(self.n_users):
            # 확률적으로 사용자 타입 선택
            user_type = np.random.choice(type_names)
            user_profiles[user_id] = user_type
            
        return user_profiles
    
    def _calculate_item_popularity(self):
        """
        아이템 인기도 계산 (실제로는 과거 데이터 기반)
        """
        # 가격과 맛을 기반으로 인기도 추정
        popularity = np.zeros(self.n_items)
        
        for i in range(self.n_items):
            # 가격이 적당하고 맛이 좋으면 인기도 높음
            price_score = 1.0 - self.restaurants_df.iloc[i].get('가격', 0.5)
            taste_score = self.restaurants_df.iloc[i].get('맛', 0.5)
            
            # 노이즈 추가
            noise = np.random.normal(0, 0.1)
            popularity[i] = (price_score * 0.4 + taste_score * 0.6 + noise)
            
        # 정규화
        popularity = np.clip(popularity, 0, 1)
        return popularity
    
    def get_user_preference(self, user_id):
        """
        사용자 선호도 벡터 반환
        """
        user_type = self.user_profiles[user_id]
        
        # 사용자 타입별 기본 선호도
        base_preferences = {
            'foodie': np.array([0.1, 0.5, 0.3, 0.1]),    # 맛 중심
            'budget': np.array([0.6, 0.2, 0.1, 0.1]),    # 가격 중심  
            'romantic': np.array([0.1, 0.2, 0.6, 0.1]),  # 분위기 중심
            'family': np.array([0.25, 0.25, 0.25, 0.25]), # 균형
            'business': np.array([0.2, 0.2, 0.3, 0.3])   # 분위기+서비스
        }
        
        # 개인차 추가 (노이즈)
        base_pref = base_preferences[user_type]
        noise = np.random.normal(0, 0.1, len(base_pref))
        personal_pref = base_pref + noise
        
        # 정규화
        personal_pref = np.clip(personal_pref, 0, 1)
        personal_pref = personal_pref / np.sum(personal_pref)
        
        return personal_pref
    
    def simulate_user_response(self, user_id, item_id, context=None):
        """
        사용자 응답 시뮬레이션 (클릭 여부 + 만족도)
        """
        if item_id >= self.n_items:
            return 0.0, False
        
        # 사용자 선호도
        user_pref = self.get_user_preference(user_id)
        
        # 아이템 특성
        item_features = self.restaurants_df.iloc[item_id][self.feature_cols[:4]].values
        
        # 선호도 매칭 점수
        preference_score = np.dot(user_pref, item_features)
        
        # 인기도 보너스
        popularity_bonus = self.item_popularity[item_id] * 0.2
        
        # 컨텍스트 효과 (시간, 날씨 등)
        context_effect = 0.0
        if context is not None:
            # 예: 비 오는 날에는 실내 분위기 좋은 곳 선호
            if len(context) > 2:  # 날씨 정보가 있다면
                context_effect = np.random.normal(0, 0.1)
        
        # 랜덤 노이즈 (예측 불가능한 요소)
        noise = np.random.normal(0, 0.15)
        
        # 최종 만족도 점수
        satisfaction = preference_score + popularity_bonus + context_effect + noise
        satisfaction = np.clip(satisfaction, 0, 1)
        
        # 클릭 확률 (만족도 기반)
        click_prob = satisfaction ** 2  # 비선형 관계
        clicked = np.random.random() < click_prob
        
        # 실제 보상 (클릭한 경우에만)
        if clicked:
            # 만족도에 따른 보상 (평점 등)
            if satisfaction > 0.8:
                reward = 1.0  # 매우 만족
            elif satisfaction > 0.6:
                reward = 0.8  # 만족
            elif satisfaction > 0.4:
                reward = 0.5  # 보통
            else:
                reward = 0.2  # 불만족
        else:
            reward = 0.0  # 클릭하지 않음
        
        return reward, clicked
    
    def generate_realistic_interactions(self, agent, n_episodes=100):
        """
        현실적인 상호작용 데이터 생성
        """
        interactions = []
        
        print(f"🎬 현실적인 상호작용 시뮬레이션 시작 ({n_episodes} 에피소드)")
        
        for episode in range(n_episodes):
            # 랜덤 사용자 선택
            user_id = np.random.randint(0, self.n_users)
            
            # 동적 컨텍스트 생성
            context = self._generate_dynamic_context()
            
            # 에이전트가 추천 생성
            item_id = agent.select_action(user_id, context)
            
            # 사용자 응답 시뮬레이션
            reward, clicked = self.simulate_user_response(user_id, item_id, context)
            
            # 상호작용 기록
            interaction = {
                'episode': episode,
                'user_id': user_id,
                'item_id': item_id,
                'reward': reward,
                'clicked': clicked,
                'context': context,
                'user_type': self.user_profiles[user_id]
            }
            interactions.append(interaction)
            
            # 에이전트 학습
            agent.store_experience(user_id, item_id, reward, context)
            
            # 주기적 업데이트
            if len(agent.experience_buffer) >= 16:
                agent.update()
            
            # 진행률 출력
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean([i['reward'] for i in interactions[-20:]])
                click_rate = np.mean([i['clicked'] for i in interactions[-20:]])
                print(f"  에피소드 {episode+1}: 평균 보상 {avg_reward:.3f}, 클릭률 {click_rate:.3f}")
        
        return interactions
    
    def _generate_dynamic_context(self):
        """
        동적 컨텍스트 생성 (시간, 날씨, 동반자 등)
        """
        # 시간대 (0: 아침, 1: 점심, 2: 저녁, 3: 야식)
        time_of_day = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.5, 0.1])
        
        # 날씨 (0: 맑음, 1: 비, 2: 추움)
        weather = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
        
        # 동반자 (0: 혼자, 1: 연인, 2: 가족, 3: 친구, 4: 비즈니스)
        companion = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        
        # 예산 수준 (0: 저예산, 1: 중예산, 2: 고예산)
        budget = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        # 원핫 인코딩으로 변환
        context = np.zeros(13)  # 4+3+5+3 = 15
        context[time_of_day] = 1.0
        context[4 + weather] = 1.0
        context[7 + companion] = 1.0
        context[12 + budget] = 1.0
        
        return context
    
    def evaluate_simulation_quality(self, interactions):
        """
        시뮬레이션 품질 평가
        """
        print("\n📊 시뮬레이션 품질 분석:")
        print("-" * 40)
        
        # 전체 통계
        total_interactions = len(interactions)
        total_clicks = sum(1 for i in interactions if i['clicked'])
        avg_reward = np.mean([i['reward'] for i in interactions])
        click_rate = total_clicks / total_interactions
        
        print(f"총 상호작용 수: {total_interactions}")
        print(f"총 클릭 수: {total_clicks}")
        print(f"전체 클릭률: {click_rate:.3f}")
        print(f"평균 보상: {avg_reward:.3f}")
        
        # 사용자 타입별 분석
        user_type_stats = defaultdict(list)
        for interaction in interactions:
            user_type = interaction['user_type']
            user_type_stats[user_type].append(interaction)
        
        print(f"\n사용자 타입별 성과:")
        for user_type, type_interactions in user_type_stats.items():
            type_click_rate = np.mean([i['clicked'] for i in type_interactions])
            type_avg_reward = np.mean([i['reward'] for i in type_interactions])
            print(f"  {user_type:10s}: 클릭률 {type_click_rate:.3f}, 평균 보상 {type_avg_reward:.3f}")
        
        # 아이템 인기도 분석
        item_interactions = defaultdict(int)
        item_rewards = defaultdict(list)
        
        for interaction in interactions:
            item_id = interaction['item_id']
            item_interactions[item_id] += 1
            if interaction['clicked']:
                item_rewards[item_id].append(interaction['reward'])
        
        # 가장 인기 있는 아이템 Top 5
        popular_items = sorted(item_interactions.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n인기 아이템 Top 5:")
        for item_id, count in popular_items:
            restaurant_name = self.restaurants_df.iloc[item_id]['name']
            avg_reward = np.mean(item_rewards[item_id]) if item_rewards[item_id] else 0
            print(f"  {restaurant_name}: {count}회 추천, 평균 보상 {avg_reward:.3f}")
        
        return {
            'total_interactions': total_interactions,
            'click_rate': click_rate,
            'avg_reward': avg_reward,
            'user_type_stats': dict(user_type_stats),
            'popular_items': popular_items
        } 