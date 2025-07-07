# Data Loader 유틸리티 구현 파일 
import pandas as pd
import os

# 프로젝트 루트 project 폴더 기준
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def load_restaurants(path=None):
    """
    TF-IDF CSV를 로드하고
    returns: df, feature_matrix, feature_columns
    """
    csv_path = path or os.path.join(BASE_DIR, 'data', 'restaurant_data.csv')
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'가게명': 'name'})
    df.insert(0, 'item_id', range(len(df)))
    # NaN 전체 컬럼 제거
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=drop_cols)
    # 특성 컬럼 추출
    feature_cols = [c for c in df.columns if c not in ['item_id','name']]
    # print(f"item_id 컬럼: {df['item_id'].unique()}")
    # print(f"feature_cols: {feature_cols}")
    feature_matrix = df[feature_cols].fillna(0).values
    return df, feature_matrix, feature_cols


def init_logs(path=None):
    log_csv = path or os.path.join(BASE_DIR, 'data', 'logs', 'user_logs.csv')
    log_dir = os.path.dirname(log_csv)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_csv):
        pd.DataFrame(columns=['user_id','item_id','reward','timestamp']).to_csv(log_csv, index=False)
    return log_csv 