#!/usr/bin/env python3
"""
PPO 에이전트 평가 실행 스크립트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔍 평가 방법을 선택하세요:")
    print("1. 기본 평가 (간단한 시뮬레이션)")
    print("2. 현실적인 평가 (고급 시뮬레이션) - 추천!")
    
    try:
        choice = input("\n선택 (1 또는 2): ").strip()
        
        if choice == "1":
            print("\n📊 기본 평가 실행...")
            from evaluation.evaluate_agent import main as basic_eval
            basic_eval()
        elif choice == "2":
            print("\n🎭 현실적인 평가 실행...")
            from evaluation.realistic_evaluate import main as realistic_eval
            realistic_eval()
        else:
            print("❌ 잘못된 선택입니다. 기본 평가를 실행합니다.")
            from evaluation.evaluate_agent import main as basic_eval
            basic_eval()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 평가가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 