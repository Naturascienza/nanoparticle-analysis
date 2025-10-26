"""
최종 수정 버전 - 과검출 문제 해결
주요 개선:
1. 훨씬 더 엄격한 임계값 (노이즈와 실제 입자 명확히 구분)
2. 강력한 형태학적 필터링
3. 강도 기반 검증 (주변보다 확실히 어두운 것만)
4. 시각적 검증 용이
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle
from skimage import filters, segmentation, measure, morphology, exposure
from scipy import ndimage
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔧 과검출 문제 해결 버전")
print("="*70)

# 1. 파일 로드
print("\n📁 EMD 파일 로딩...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")

if isinstance(signal, list):
    signal = signal[0]

# 픽셀 크기
try:
    x_axis = signal.axes_manager.signal_axes[0]
    pixel_size_x = x_axis.scale if hasattr(x_axis, 'scale') else 1.0
    x_units = x_axis.units if hasattr(x_axis, 'units') else 'pixels'
except:
    pixel_size_x = 1.0
    x_units = 'pixels'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")

# 2. ROI 설정
print("\n" + "="*70)
print("🎯 ROI 설정")
print("="*70)

use_roi = input("\nROI 설정? [y/N]: ").strip().lower()
roi_coords = None

if use_roi in ['y', 'yes']:
    try:
        center_x = int(input(f"ROI 중심 X [{signal.data.shape[1]//2}]: ") or str(signal.data.shape[1]//2))
        center_y = int(input(f"ROI 중심 Y [{signal.data.shape[0]//2}]: ") or str(signal.data.shape[0]//2))
        roi_size = int(input("ROI 크기 [250]: ") or "250")
        
        half = roi_size // 2
        x_start = max(0, center_x - half)
        y_start = max(0, center_y - half)
        x_end = min(signal.data.shape[1], center_x + half)
        y_end = min(signal.data.shape[0], center_y + half)
        
        roi_coords = (x_start, y_start, x_end, y_end)
        analysis_data = signal.data[y_start:y_end, x_start:x_end].copy()
        print(f"✅ ROI: ({x_start},{y_start}) → ({x_end},{y_end})")
    except:
        analysis_data = signal.data.copy()
else:
    analysis_data = signal.data.copy()

print(f"분석 영역 크기: {analysis_data.shape}")

# 3. 개선된 전처리 (노이즈 제거 강화)
print("\n" + "="*70)
print("🎨 전처리 (노이즈 제거 강화)")
print("="*70)

# 3.1 백그라운드 제거
print("1️⃣ 백그라운드 제거...")
from scipy.ndimage import gaussian_filter
background = gaussian_filter(analysis_data, sigma=30)  # 큰 스케일
bg_removed = analysis_data - background
bg_removed = np.clip(bg_removed, 0, None)

# 3.2 정규화
p1, p99 = np.percentile(bg_removed, [1, 99])
normalized = np.clip((bg_removed - p1) / (p99 - p1), 0, 1)

# 3.3 약간의 가우시안 블러 (작은 노이즈 제거)
print("2️⃣ 노이즈 제거...")
smoothed = gaussian_filter(normalized, sigma=1.0)

print("✅ 전처리 완료")

# 4. 임계값 - 훨씬 더 엄격하게
print("\n" + "="*70)
print("🎯 임계값 설정 (매우 엄격)")
print("="*70)

# 이미지 반전 (어두운 입자 → 밝게)
inverted = 1.0 - smoothed

# 여러 임계값 계산
thresh_otsu = filters.threshold_otsu(inverted)
thresh_li = filters.threshold_li(inverted)

# ⭐ 핵심: 훨씬 더 높은 임계값 사용
# 실제 입자만 검출하도록 상위 백분위수 사용
thresh_percentile = np.percentile(inverted, 99)  # 상위 1%만

print(f"임계값 비교:")
print(f"  Otsu: {thresh_otsu:.4f}")
print(f"  Li: {thresh_li:.4f}")
print(f"  99 percentile: {thresh_percentile:.4f} ⭐")

print(f"\n임계값 선택:")
print("1. 99 percentile (매우 엄격 - 권장)")
print("2. 98 percentile (엄격)")
print("3. 95 percentile (보통)")
print("4. Otsu")
print("5. 수동")

choice = input("선택 [1-5, 기본: 1]: ").strip() or "1"

if choice == "1":
    threshold = thresh_percentile
    method_name = "99%ile"
elif choice == "2":
    threshold = np.percentile(inverted, 98)
    method_name = "98%ile"
elif choice == "3":
    threshold = np.percentile(inverted, 95)
    method_name = "95%ile"
elif choice == "4":
    threshold = thresh_otsu
    method_name = "Otsu"
elif choice == "5":
    threshold = float(input("임계값 (0-1): "))
    method_name = "Manual"
else:
    threshold = thresh_percentile
    method_name = "99%ile"

print(f"✅ 임계값: {threshold:.4f} ({method_name})")

# 5. 이진화 및 강력한 필터링
print("\n🔍 이진화 및 필터링...")

binary_mask = inverted > threshold

# 매우 작은 것 제거 (면적)
min_area_nm2 = 0.8  # 0.8 nm² 미만 제거
min_area_pixels = int(min_area_nm2 / (pixel_size_x ** 2))
min_area_pixels = max(15, min_area_pixels)

print(f"  최소 면적: {min_area_pixels} 픽셀² (≈ {min_area_nm2} {x_units}²)")

binary_cleaned = morphology.remove_small_objects(binary_mask, min_size=min_area_pixels)
binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_area_pixels//2)

# 추가 모폴로지: opening (작은 돌기 제거)
binary_cleaned = morphology.binary_opening(binary_cleaned, morphology.disk(2))

print(f"  초기 검출 영역: {measure.label(binary_cleaned).max()}개")

# 6. Watershed (매우 보수적)
print("\n🌊 Watershed 분할 (보수적)...")

distance = ndimage.distance_transform_edt(binary_cleaned)

# 로컬 최대값 - 매우 엄격하게
from skimage.feature import peak_local_max
min_distance_nm = 3.0  # 3nm 이내는 하나로 간주
min_distance_px = max(5, int(min_distance_nm / pixel_size_x))

print(f"  최소 분리 거리: {min_distance_px} 픽셀 (≈ {min_distance_px * pixel_size_x:.2f} {x_units})")

coords = peak_local_max(distance, 
                       min_distance=min_distance_px,
                       labels=binary_cleaned,
                       threshold_abs=3.0)  # distance가 3픽셀 이상인 것만

print(f"  피크 검출: {len(coords)}개")

if len(coords) > 0:
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    labeled = segmentation.watershed(-distance, markers, mask=binary_cleaned)
else:
    labeled = measure.label(binary_cleaned)

n_detected = labeled.max()
print(f"  ✅ Watershed 후: {n_detected}개")

# 7. 강도 기반 검증 (⭐ 핵심)
print("\n⭐ 강도 기반 검증 (실제 입자인지 확인)...")

props = measure.regionprops(labeled, intensity_image=smoothed)

particle_data = []
for prop in props:
    # 기본 측정
    area_pixels = prop.area
    equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
    equiv_diameter_real = equiv_diameter_pixels * pixel_size_x
    
    # ⭐ 강도 검증
    mean_intensity = prop.mean_intensity  # 입자 내부 평균 강도
    
    # 입자 주변 영역의 강도
    minr, minc, maxr, maxc = prop.bbox
    # 경계 확장
    pad = 5
    minr_pad = max(0, minr - pad)
    minc_pad = max(0, minc - pad)
    maxr_pad = min(smoothed.shape[0], maxr + pad)
    maxc_pad = min(smoothed.shape[1], maxc + pad)
    
    roi_patch = smoothed[minr_pad:maxr_pad, minc_pad:maxc_pad]
    mask_patch = labeled[minr_pad:maxr_pad, minc_pad:maxc_pad] == prop.label
    
    # 주변 영역 (입자 제외)
    background_mask = ~mask_patch
    if np.sum(background_mask) > 0:
        background_intensity = np.mean(roi_patch[background_mask])
    else:
        background_intensity = np.mean(smoothed)
    
    # 대비 계산 (실제 입자는 주변보다 어두워야 함)
    # smoothed는 반전 전이므로, 입자는 낮은 값
    contrast_ratio = background_intensity / (mean_intensity + 1e-10)
    
    print(f"  입자 {prop.label}: 크기={equiv_diameter_real:.2f}{x_units}, "
          f"대비={contrast_ratio:.2f}x")
    
    # ⭐ 필터 조건 (매우 엄격)
    # 1. 크기 필터
    if equiv_diameter_real < 0.6 or equiv_diameter_real > 5.0:
        print(f"    ❌ 크기 범위 외")
        continue
    
    # 2. 대비 필터 (주변보다 확실히 어두워야 함)
    if contrast_ratio < 1.3:  # 주변보다 최소 30% 어두워야 함
        print(f"    ❌ 대비 부족 (주변과 비슷)")
        continue
    
    # 3. 형태 필터
    if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            if aspect_ratio > 3.0:
                print(f"    ❌ 너무 길쭉함 (aspect={aspect_ratio:.1f})")
                continue
    
    print(f"    ✅ 통과")
    
    # ROI 좌표 보정
    centroid_x = prop.centroid[1] + (roi_coords[0] if roi_coords else 0)
    centroid_y = prop.centroid[0] + (roi_coords[1] if roi_coords else 0)
    
    particle_data.append({
        'Particle_ID': len(particle_data) + 1,
        'Centroid_X': centroid_x,
        'Centroid_Y': centroid_y,
        f'Diameter_{x_units}': equiv_diameter_real,
        f'Area_{x_units}2': area_pixels * (pixel_size_x ** 2),
        'Contrast_Ratio': contrast_ratio,
        'Mean_Intensity': mean_intensity,
        'Background_Intensity': background_intensity
    })

n_particles = len(particle_data)

print(f"\n📊 최종 결과:")
print(f"  초기 검출: {n_detected}개")
print(f"  강도 검증 후: {n_particles}개")

# 8. 결과
if n_particles > 0:
    df = pd.DataFrame(particle_data)
    sizes = df[f'Diameter_{x_units}']
    
    print(f"\n✅ 검출된 입자: {n_particles}개")
    print(f"\n📊 크기 통계:")
    print(f"  평균: {sizes.mean():.3f} {x_units}")
    print(f"  표준편차: {sizes.std():.3f} {x_units}")
    print(f"  중앙값: {sizes.median():.3f} {x_units}")
    print(f"  범위: {sizes.min():.3f} - {sizes.max():.3f} {x_units}")
    
    # 9. 시각화
    print("\n📈 시각화...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Fixed Detection (n={n_particles})', fontsize=16, fontweight='bold')
    
    # 원본 (ROI 또는 전체)
    vmin, vmax = np.percentile(analysis_data, [2, 98])
    axes[0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original (Analysis Region)')
    
    # 전처리
    axes[1].imshow(smoothed, cmap='gray', origin='lower')
    axes[1].set_title('Preprocessed')
    
    # 검출 결과
    axes[2].imshow(smoothed, cmap='gray', origin='lower')
    
    # 입자 표시
    for _, particle in df.iterrows():
        if roi_coords:
            x = particle['Centroid_X'] - roi_coords[0]
            y = particle['Centroid_Y'] - roi_coords[1]
        else:
            x = particle['Centroid_X']
            y = particle['Centroid_Y']
        
        r = particle[f'Diameter_{x_units}'] / (2 * pixel_size_x)
        
        circle = Circle((x, y), r, fill=False, color='red', linewidth=2)
        axes[2].add_patch(circle)
        
        # ID와 크기 표시
        axes[2].text(x, y - r - 5, f"#{particle['Particle_ID']}\n{particle[f'Diameter_{x_units}']:.2f}{x_units}",
                    ha='center', va='bottom', fontsize=10, color='yellow', 
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    axes[2].set_title(f'Detected: {n_particles} particles')
    
    for ax in axes:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    # 저장
    suffix = f"_fixed_{n_particles}p"
    if roi_coords:
        suffix += "_ROI"
    
    fig.savefig(f'detection_result{suffix}.png', dpi=300, bbox_inches='tight')
    df.to_csv(f'particle_data{suffix}.csv', index=False)
    
    print(f"\n💾 저장:")
    print(f"  detection_result{suffix}.png")
    print(f"  particle_data{suffix}.csv")
    
    # 대비 분석
    print(f"\n📊 대비 분석:")
    for _, p in df.iterrows():
        print(f"  입자 {p['Particle_ID']}: 대비 {p['Contrast_Ratio']:.2f}x "
              f"(입자={p['Mean_Intensity']:.3f}, 배경={p['Background_Intensity']:.3f})")
    
    plt.show()
    
else:
    print("\n✅ 검출된 입자 없음")
    print("이는 정상일 수 있습니다 (임계값이 매우 엄격함)")
    
    # 시각화 (참고용)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    vmin, vmax = np.percentile(analysis_data, [2, 98])
    axes[0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original')
    
    axes[1].imshow(binary_mask, cmap='gray', origin='lower')
    axes[1].set_title(f'Binary (threshold={threshold:.4f})')
    
    plt.tight_layout()
    plt.savefig('detection_result_none.png', dpi=300, bbox_inches='tight')
    print("💾 detection_result_none.png")
    
    plt.show()

print("\n" + "="*70)
print("프로그램 종료")
print("="*70)
