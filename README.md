# TEM 나노입자 자동 검출·계측 (nanoparticle-analysis)

투과전자현미경(TEM) 이미지에서 나노입자를 자동으로 찾아 크기를 재고, 크기 분포 통계를 만드는 Python 도구입니다.
응집된 입자가 많아 수작업 계측에 시간이 오래 걸리고 측정자마다 결과가 달라지는 문제를 줄이기 위해 만들었습니다.

AI 전문가 양성 온라인 부트캠프(960시간) 최종 프로젝트로 개발했으며, 이 프로젝트로 최우수상(한국표준협회장상)을 받았습니다.

## 무엇을 하나
- **입자 검출**: LoG(Laplacian of Gaussian) 블롭 검출로 크기가 다른 입자를 함께 찾습니다.
- **응집 입자 분리**: Watershed 분할로 붙어 있는 입자를 나눕니다.
- **전처리**: CLAHE 대비 향상과 적응형 임계값으로 대비가 약한 입자도 검출합니다.
- **신뢰도 점수**: 원형도·종횡비·크기를 바탕으로 입자마다 0~1 점수를 매기고, 기준 미만은 걸러 냅니다.
- **통계·리포트**: 입자별 CSV, 검출 결과 이미지, 크기 분포 히스토그램과 가우시안 피팅, 요약 통계(txt)를 자동으로 만듭니다.

## 결과 (부트캠프 평가용 시험 이미지 기준)
- 수작업 계측값과 비교한 평균 크기 오차 약 0.4% (정확도 99.6%)
- 이미지 1장(입자 약 100개) 분석 시간: 수작업 약 60분 → 약 5분
- 같은 이미지·같은 설정에서 반복 실행 시 동일한 결과

> 위 수치는 프로젝트에서 사용한 시험 이미지와 수동 측정 샘플(`particle_samples.csv`)로 확인한 값입니다. 장비·배율·시료가 달라지면 매개변수 조정이 필요합니다.

## 설치
```bash
git clone https://github.com/Naturascienza/nanoparticle-analysis.git
cd nanoparticle-analysis
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 사용법
```bash
# 메인 분석 (대화형: 전처리 방법, ROI, Watershed 사용 여부, 크기 필터 선택)
python emd_analysis_Size_histogram_Au.py
```
입력은 EMD 형식의 TEM 이미지(HyperSpy로 읽음)입니다.

## 폴더 구성
- `emd_analysis*.py`, `particle_detection_*.py`, `improved_particle_detection_*.py` — 검출·계측 스크립트(개발 단계별 버전)
- `core/` — 핵심 분석 모듈
- `visualization/` — LoG·Watershed 설명 그림, 워크플로 다이어그램, 신뢰도 분석 그림 생성
- `docs/` — 검출 문제 분석과 개선 기록
- `particle_summary_*.txt` — 시험 이미지 분석 결과 요약(입자 수, ROI, 신뢰도 기준)

## 사용 기술
Python, HyperSpy, scikit-image, SciPy, NumPy, pandas, Matplotlib

## 다음 단계
- 수동 측정 샘플을 늘려 배율·시료별 검증
- 고전적 검출 결과를 학습 데이터로 삼아 딥러닝 분할 모델(U-Net 계열)과 비교

---

## English summary
A Python tool that automatically detects nanoparticles in TEM (EMD) images, separates agglomerated particles (LoG + Watershed + CLAHE), scores each particle's reliability, and reports size distributions. On the test images used in the project it matched manual measurements within about 0.4% in mean size and cut analysis time per image from about 60 min to about 5 min. Developed as the final project of a 960-hour AI training bootcamp (top award, Korean Standards Association).

License: MIT
