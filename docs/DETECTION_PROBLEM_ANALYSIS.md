# 입자 검출 성능 문제 분석 및 해결방안

## 📋 현재 상황

### 기존 방법 (emd_analysis_Size_histogram_Pt.py)
- **검출 방법**: 임계값 기반 이진화 + Watershed 분할
- **최근 결과**: 1,067개 입자 (ROI 기준)
- **문제점**: 
  1. 임계값 선택에 매우 민감
  2. 작은 입자와 노이즈 구별 어려움
  3. 응집된 입자 분리 불완전
  4. 과도한 필터링으로 실제 입자 누락 가능

### LoG 기반 방법 시도 결과
- **검출 수**: 1,568개 (전체 이미지)
- **문제점**: 
  - 대부분 최소 크기(1.5nm)로 검출됨
  - 검출 범위 설정 문제
  - 스케일 파라미터 최적화 필요

## 🔍 주요 문제점

### 1. 픽셀 크기 및 검출 범위
```
픽셀 크기: 0.0335 nm/pixel
목표 입자: 1.5-3.0 nm (45-90 픽셀)
```

**문제**: LoG 방법에서 검출 범위가 부적절하게 설정됨

### 2. 전처리 문제
- 백그라운드 제거가 불완전
- 대비가 충분하지 않음
- 노이즈와 실제 입자 구별 어려움

### 3. 임계값 문제
- Otsu 임계값이 실제 데이터에 맞지 않을 수 있음
- 입자 개수 기반 자동 조정이 필터링 전 개수 기준이라 부정확

### 4. Watershed 파라미터
- footprint 크기가 적절하지 않을 수 있음
- 응집체 분리가 과도하거나 부족할 수 있음

## ✅ 해결 방안

### 방안 1: 기존 방법 개선 (권장)

#### 1-1. 임계값 최적화
```python
# 현재 (문제)
threshold = filters.threshold_otsu(analysis_data)  # 전역 임계값

# 개선안
# 1) 로컬 임계값 사용
threshold = filters.threshold_local(analysis_data, block_size=101)

# 2) 여러 방법 비교
otsu = filters.threshold_otsu(analysis_data)
li = filters.threshold_li(analysis_data)
yen = filters.threshold_yen(analysis_data)
# 중앙값 사용 또는 사용자가 선택
```

#### 1-2. 전처리 강화
```python
# 백그라운드 제거 개선
from scipy.ndimage import gaussian_filter, median_filter

# 1) Rolling ball 대신 Top-hat 변환
from skimage.morphology import disk, white_tophat
radius = 20  # 입자 크기의 2-3배
selem = disk(radius)
bg_removed = white_tophat(original_data, selem)

# 2) 언샵 마스킹 (엣지 강화)
blurred = gaussian_filter(bg_removed, sigma=1)
sharpened = bg_removed + 0.5 * (bg_removed - blurred)

# 3) CLAHE (대비 개선)
from skimage.exposure import equalize_adapthist
enhanced = equalize_adapthist(sharpened, clip_limit=0.03)
```

#### 1-3. Watershed 최적화
```python
# 현재
watershed_footprint = 7  # 고정값

# 개선안: 픽셀 크기 기반 자동 계산
target_separation_nm = 3.0  # nm
watershed_footprint = int(target_separation_nm / pixel_size_x)
watershed_footprint = max(3, min(15, watershed_footprint))

# 추가: distance_transform 전 침식(erosion)으로 경계 명확화
from skimage.morphology import erosion
eroded = erosion(binary_cleaned, disk(1))
distance = ndimage.distance_transform_edt(eroded)
```

#### 1-4. 크기 필터링 개선
```python
# 현재: 단순 면적 제거
morphology.remove_small_objects(binary, min_size=min_particle_size)

# 개선안: 다단계 필터링
# 1) 매우 작은 노이즈 제거 (면적)
cleaned1 = morphology.remove_small_objects(binary, min_size=10)

# 2) 구멍 채우기
cleaned2 = morphology.remove_small_holes(cleaned1, area_threshold=20)

# 3) 최종 크기 필터 (직경 기준)
# regionprops로 각 입자 직경 확인 후 필터링
```

### 방안 2: 하이브리드 접근법

```python
# 1) 임계값 방법으로 대략적 검출
binary = image > threshold_otsu(image)
labeled = measure.label(binary)

# 2) 각 영역에 LoG 적용 (지역적 정밀 검출)
for region in measure.regionprops(labeled):
    minr, minc, maxr, maxc = region.bbox
    roi = image[minr:maxr, minc:maxc]
    
    # ROI 내에서 LoG 검출
    blobs = feature.blob_log(roi, ...)
    
# 3) 결과 병합
```

### 방안 3: 매개변수 자동 최적화

```python
def auto_optimize_detection(image, target_size_range=(1.5, 3.0), 
                            expected_count_range=(800, 1200)):
    """
    여러 매개변수 조합을 시도하여 최적 설정 찾기
    """
    best_score = float('inf')
    best_params = None
    
    # 임계값 범위
    for threshold_method in ['otsu', 'li', 'yen']:
        # Watershed footprint 범위
        for footprint in [5, 7, 9]:
            # 검출 수행
            particles = detect_particles(image, threshold_method, footprint)
            
            # 점수 계산 (목표 범위와의 차이)
            count_diff = abs(len(particles) - np.mean(expected_count_range))
            size_diff = abs(np.median(particles['size']) - np.mean(target_size_range))
            
            score = count_diff + size_diff * 100
            
            if score < best_score:
                best_score = score
                best_params = (threshold_method, footprint)
    
    return best_params
```

## 🎯 즉시 적용 가능한 개선 (우선순위)

### 1단계: 임계값 방법 개선
```python
# Li 방법 시도 (Otsu보다 나을 수 있음)
threshold = filters.threshold_li(analysis_data)

# 또는 다중 임계값 평균
thresh_otsu = filters.threshold_otsu(analysis_data)
thresh_li = filters.threshold_li(analysis_data)
threshold = (thresh_otsu + thresh_li) / 2
```

### 2단계: 최소 입자 크기 조정
```python
# 현재
min_particle_size = recommended_min_pixels  # 면적 기준

# 개선: 더 작게 설정하여 과도한 필터링 방지
min_particle_size = recommended_min_pixels // 2  # 절반으로
```

### 3단계: Watershed 민감도 조정
```python
# 현재
watershed_footprint = recommended_footprint  # 3

# 개선: 더 민감하게 (작은 입자도 분리)
watershed_footprint = max(2, recommended_footprint - 1)
```

### 4단계: 신뢰성 필터 완화
```python
# 현재
reliability_score_threshold = 0.6

# 개선: 더 낮게 설정 (더 많은 입자 포함)
reliability_score_threshold = 0.4
```

## 📊 검증 방법

1. **수동 측정과 비교**: 일부 영역을 수동으로 측정하여 자동 검출 결과와 비교
2. **시각적 검증**: 검출된 입자를 원본 이미지에 오버레이하여 누락/오검출 확인
3. **크기 분포 비교**: 히스토그램이 예상되는 분포(가우시안)와 유사한지 확인
4. **다른 이미지 테스트**: 여러 EMD 파일에 동일한 매개변수 적용하여 일관성 확인

## 🔧 권장 실행 계획

1. **현재 방법 개선** (가장 빠른 효과)
   - 임계값 Li 방법 시도
   - 최소 크기 절반으로 감소
   - Watershed footprint 2-3으로 설정
   - 신뢰성 임계값 0.4로 완화

2. **전처리 강화** (중기)
   - Top-hat 변환 추가
   - 언샵 마스킹 적용
   - CLAHE 매개변수 조정

3. **하이브리드 방법 개발** (장기)
   - 임계값 + LoG 결합
   - 자동 최적화 시스템 구축

## 💡 즉시 테스트할 코드

다음 페이지에 즉시 테스트 가능한 개선 코드를 제공하겠습니다.
