# Farm System AI_Team 1

본 레포지토리는 PPO(Proximal Policy Optimization) 강화학습 알고리즘을 통해서 개인화된 추천 시스템을 구현하기 위한 코드를 제공합니다.

## 프로젝트 개요
* **목표**: PPO 강화학습 알고리즘을 이용한 레스토랑 추천 시스템 구현
* **전략**: PyTorch를 이용하여 Policy Network와 Value Network를 구현하고, 실제 레스토랑 데이터를 활용한 시뮬레이션을 통해 성능 테스트

## ⚙️ 환경 세팅 & 실행 방법

### 1. 사전 준비 

```bash
git clone https://github.com/your-username/4th-AI-proj1-team1.git
cd 4th-AI-proj1-team1
pip install -r requirements.txt
```

### 2. 실험 실행

```bash
# Streamlit 웹 UI 실행
streamlit run ui/streamlit_app.py

# PPO 에이전트 평가 실행
python run_evaluation.py

# 개별 모듈 테스트
python -m evaluation.evaluate_agent
python -m evaluation.realistic_evaluate
```


## 폴더 구조
```
📁 4th-AI-proj1-team1/
├── agents/                 # PPO 에이전트 구현
│   ├── ppo_agent.py       # PPO 알고리즘 메인 구현체
│   └── __init__.py        # 패키지 초기화 파일
├── data/                   # 데이터 관련 파일
│   ├── restaurant_data.csv # 레스토랑 데이터셋
│   ├── logs/              # 학습 로그 저장 폴더
│   └── __init__.py        # 패키지 초기화 파일
├── envs/                   # 강화학습 환경 정의
│   ├── restaurant_env.py  # 레스토랑 추천 환경 구현
│   └── __init__.py        # 패키지 초기화 파일
├── evaluation/             # 평가 및 시뮬레이션
│   ├── evaluate_agent.py  # 기본 에이전트 평가
│   ├── realistic_evaluate.py # 현실적인 시뮬레이션 평가
│   ├── realistic_simulation.py # 고급 시뮬레이션 로직
│   ├── metrics.py         # 평가 지표 계산
│   ├── results/           # 평가 결과 저장 폴더
│   └── __init__.py        # 패키지 초기화 파일
├── ui/                     # 웹 인터페이스
│   ├── streamlit_app.py   # Streamlit 웹 애플리케이션
│   └── __init__.py        # 패키지 초기화 파일
├── utils/                  # 유틸리티 함수
│   ├── data_loader.py     # 데이터 로딩 유틸리티
│   ├── similarity.py      # 유사도 계산 함수
│   └── __init__.py        # 패키지 초기화 파일
├── run_evaluation.py       # 메인 실행 스크립트
├── requirements.txt        # 프로젝트 의존성 패키지
└── README.md              # 프로젝트 설명서
```

## 🚀 주요 기능

### PPO 에이전트
- **Policy Network**: 사용자 상태를 입력받아 아이템 선택 확률 출력
- **Value Network**: 상태 가치 함수 근사
- **경험 버퍼**: 학습을 위한 경험 데이터 저장
- **GAE (Generalized Advantage Estimation)**: 어드밴티지 계산

### 평가 시스템
- **기본 평가**: 간단한 시뮬레이션을 통한 성능 측정
- **현실적인 평가**: 복잡한 사용자 행동을 모델링한 고급 시뮬레이션
- **성능 지표**: 정확도, 다양성, 참신성 등 다양한 추천 품질 지표

### 웹 인터페이스
- **Streamlit 기반**: 직관적인 웹 UI를 통한 실시간 추천 시스템 데모
- **시각화**: 학습 과정 및 성능 지표 시각화

## 📊 성능 지표
- **정확도 (Accuracy)**: 추천 아이템의 정확성
- **다양성 (Diversity)**: 추천 목록의 다양성
- **참신성 (Novelty)**: 새로운 아이템 발견 능력
- **커버리지 (Coverage)**: 전체 아이템 중 추천되는 비율
