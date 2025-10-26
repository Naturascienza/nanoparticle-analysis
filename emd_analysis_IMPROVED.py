"""
개선된 입자 검출 스크립트
주요 개선사항:
1. Li 임계값 방법 사용
2. 완화된 최소 크기 필터
3. 최적화된 Watershed 매개변수
4. 향상된 전처리
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
print("🚀 개선된 입자 분석 프로그램")
print("="*70)

# 1. 파일 로드
print("\n📁 EMD 파일 로딩 중...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")

if isinstance(signal, list):
    signal = signal[0]

print(f"📊 데이터 형태: {signal.data.shape}")
print(f"📈 데이터 타입: {signal.data.dtype}")

# 픽셀 크기 정보
try:
    x_axis = signal.axes_manager.signal_axes[0]
    y_axis = signal.axes_manager.signal_axes[1]
    pixel_size_x = x_axis.scale if hasattr(x_axis, 'scale') else 1.0
    pixel_size_y = y_axis.scale if hasattr(y_axis, 'scale') else 1.0
    x_units = x_axis.units if hasattr(x_axis, 'units') else 'pixels'
    y_units = y_axis.units if hasattr(y_axis, 'units') else 'pixels'
except:
    pixel_size_x = pixel_size_y = 1.0
    x_units = y_units = 'pixels'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")
print(f"📏 이미지 크기: {signal.data.shape[1] * pixel_size_x:.1f} × {signal.data.shape[0] * pixel_size_y:.1f} {x_units}")

# 2. 사용자 입력
print("\n" + "="*70)
print("🔧 분석 설정")
print("="*70)

# ROI 설정
use_roi = input("\nROI를 설정하시겠습니까? [y/N]: ").strip().lower()
roi_coords = None

if use_roi in ['y', 'yes']:
    print(f"이미지 크기: {signal.data.shape[1]} × {signal.data.shape[0]} 픽셀")
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
        analysis_data = signal.data[y_start:y_end, x_start:x_end]
        print(f"✅ ROI 설정: ({x_start},{y_start}) → ({x_end},{y_end})")
    except:
        print("❌ ROI 설정 실패. 전체 이미지를 사용합니다.")
        analysis_data = signal.data.copy()
else:
    analysis_data = signal.data.copy()
    print("✅ 전체 이미지 분석")

# 3. 개선된 전처리
print("\n" + "="*70)
print("🎨 개선된 전처리")
print("="*70)

print("1️⃣ Top-hat 변환으로 백그라운드 제거...")
# 입자보다 큰 구조 요소
radius = int(5.0 / pixel_size_x)  # 5nm
selem = morphology.disk(radius)
tophat = morphology.white_tophat(analysis_data, selem)
print(f"   ✅ 완료 (radius={radius} pixels)")

print("2️⃣ 언샵 마스킹으로 엣지 강화...")
from scipy.ndimage import gaussian_filter
blurred = gaussian_filter(tophat, sigma=1.5)
sharpened = tophat + 0.3 * (tophat - blurred)
sharpened = np.clip(sharpened, 0, None)
print("   ✅ 완료")

print("3️⃣ CLAHE로 대비 개선...")
# 정규화
p2, p98 = np.percentile(sharpened, [2, 98])
normalized = np.clip((sharpened - p2) / (p98 - p2), 0, 1)
# CLAHE
clahe = exposure.equalize_adapthist(normalized, clip_limit=0.03)
print("   ✅ 완료")

print("4️⃣ 가우시안 필터로 노이즈 제거...")
processed_data = gaussian_filter(clahe, sigma=0.8)
print("   ✅ 완료")

# 4. 개선된 임계값 설정
print("\n" + "="*70)
print("🎯 개선된 임계값 설정")
print("="*70)

# 이미지 반전
inverted_data = 1.0 - processed_data

# 여러 임계값 방법 비교
thresh_otsu = filters.threshold_otsu(inverted_data)
thresh_li = filters.threshold_li(inverted_data)
try:
    thresh_yen = filters.threshold_yen(inverted_data)
except:
    thresh_yen = thresh_otsu

print(f"임계값 방법별 결과:")
print(f"  - Otsu: {thresh_otsu:.4f}")
print(f"  - Li: {thresh_li:.4f}")
print(f"  - Yen: {thresh_yen:.4f}")

# 사용자 선택
print(f"\n임계값 방법 선택:")
print("1. Li (권장 - 일반적으로 더 민감)")
print("2. Otsu (기본)")
print("3. Yen (낮은 대비 이미지용)")
print("4. 평균값")
print("5. 수동 입력")

choice = input("선택 [1-5, 기본: 1]: ").strip() or "1"

if choice == "1":
    threshold = thresh_li
    method_name = "Li"
elif choice == "2":
    threshold = thresh_otsu
    method_name = "Otsu"
elif choice == "3":
    threshold = thresh_yen
    method_name = "Yen"
elif choice == "4":
    threshold = (thresh_otsu + thresh_li + thresh_yen) / 3
    method_name = "Average"
elif choice == "5":
    threshold = float(input("임계값 입력 (0-1): "))
    method_name = "Manual"
else:
    threshold = thresh_li
    method_name = "Li"

print(f"✅ 선택: {method_name} 방법, 임계값 = {threshold:.4f}")

# 5. 이진화
print("\n🔍 이진화 및 모폴로지 연산...")
binary_mask = inverted_data > threshold

# 개선된 최소 크기 계산
target_min_diameter_nm = 1.2  # nm (더 작게)
target_min_area_nm2 = np.pi * (target_min_diameter_nm / 2) ** 2
min_particle_size = int(np.ceil(target_min_area_nm2 / (pixel_size_x ** 2)))
min_particle_size = max(8, min_particle_size)  # 최소 8픽셀²

print(f"  최소 입자 면적: {min_particle_size} 픽셀² (≈ {target_min_diameter_nm} {x_units} 직경)")

# 노이즈 제거 (더 완화)
binary_cleaned = morphology.remove_small_objects(binary_mask, min_size=min_particle_size)
binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_particle_size//2)
print(f"  ✅ 노이즈 제거 완료")

# 6. 개선된 Watershed 분할
print("\n🌊 Watershed 분할...")

# 더 민감한 footprint
target_separation_nm = 2.5  # nm (더 작게)
watershed_footprint = int(target_separation_nm / pixel_size_x)
watershed_footprint = max(2, min(10, watershed_footprint))

print(f"  Footprint: {watershed_footprint} 픽셀 (≈ {watershed_footprint * pixel_size_x:.2f} {x_units})")

# 거리 변환
distance = ndimage.distance_transform_edt(binary_cleaned)

# 로컬 최대값
from skimage.feature import peak_local_max
coords = peak_local_max(distance, 
                       footprint=np.ones((watershed_footprint, watershed_footprint)),
                       labels=binary_cleaned)

# 마커 생성
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)

# Watershed
labeled_particles = segmentation.watershed(-distance, markers, mask=binary_cleaned)
n_particles_raw = labeled_particles.max()

print(f"  ✅ Watershed 완료: {n_particles_raw}개 검출")

# 7. 입자 속성 측정
print("\n📊 입자 속성 측정 중...")
particle_props = measure.regionprops(labeled_particles)

particle_data = []
for prop in particle_props:
    # 기본 측정
    area_pixels = prop.area
    area_real = area_pixels * (pixel_size_x ** 2)
    equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
    equiv_diameter_real = equiv_diameter_pixels * pixel_size_x
    
    # ROI 좌표 보정
    centroid_x_local = prop.centroid[1]
    centroid_y_local = prop.centroid[0]
    
    if roi_coords is not None:
        centroid_x = centroid_x_local + roi_coords[0]
        centroid_y = centroid_y_local + roi_coords[1]
    else:
        centroid_x = centroid_x_local
        centroid_y = centroid_y_local
    
    # 형태 특성
    if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
        else:
            aspect_ratio = 1.0
    else:
        aspect_ratio = 1.0
    
    perimeter = prop.perimeter if hasattr(prop, 'perimeter') else 2 * np.sqrt(np.pi * area_pixels)
    circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
    
    # 완화된 필터 (더 많은 입자 포함)
    # 기본적인 필터만 적용
    if equiv_diameter_real < 0.8 or equiv_diameter_real > 10.0:
        continue  # 너무 작거나 큰 것만 제외
    
    if aspect_ratio > 5.0:  # 너무 길쭉한 것만 제외
        continue
    
    particle_data.append({
        'Particle_ID': len(particle_data) + 1,
        'Area_pixels': area_pixels,
        f'Area_{x_units}2': area_real,
        f'Diameter_{x_units}': equiv_diameter_real,
        'Aspect_Ratio': aspect_ratio,
        'Circularity': circularity,
        'Centroid_X': centroid_x,
        'Centroid_Y': centroid_y
    })

n_particles = len(particle_data)
print(f"  검출: {n_particles_raw}개")
print(f"  필터링 후: {n_particles}개")

# 8. 결과 분석
if n_particles > 0:
    df = pd.DataFrame(particle_data)
    sizes = df[f'Diameter_{x_units}']
    
    print(f"\n📊 크기 통계:")
    print(f"  평균: {sizes.mean():.3f} {x_units}")
    print(f"  표준편차: {sizes.std():.3f} {x_units}")
    print(f"  중앙값: {sizes.median():.3f} {x_units}")
    print(f"  범위: {sizes.min():.3f} - {sizes.max():.3f} {x_units}")
    
    # 크기 분포
    small = len(sizes[sizes < 1.5])
    medium = len(sizes[(sizes >= 1.5) & (sizes < 3.0)])
    large = len(sizes[sizes >= 3.0])
    
    print(f"\n📊 크기 분포:")
    print(f"  <1.5 {x_units}: {small}개 ({small/n_particles*100:.1f}%)")
    print(f"  1.5-3.0 {x_units}: {medium}개 ({medium/n_particles*100:.1f}%)")
    print(f"  ≥3.0 {x_units}: {large}개 ({large/n_particles*100:.1f}%)")
    
    # 9. 시각화
    print("\n📈 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Improved Detection Results (n={n_particles})', fontsize=16, fontweight='bold')
    
    # 9.1 원본
    display_data = analysis_data if roi_coords is None else signal.data
    vmin, vmax = np.percentile(display_data, [2, 98])
    axes[0, 0].imshow(display_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    # ROI 표시
    if roi_coords is not None:
        x_start, y_start, x_end, y_end = roi_coords
        roi_rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                            fill=False, edgecolor='red', linewidth=2, linestyle='--')
        axes[0, 0].add_patch(roi_rect)
    
    axes[0, 0].set_title('Original Image' + (' with ROI' if roi_coords else ''))
    
    # 9.2 전처리
    axes[0, 1].imshow(processed_data, cmap='gray', origin='lower')
    axes[0, 1].set_title('Preprocessed (Top-hat + Sharpen + CLAHE)')
    
    # 9.3 검출 결과
    axes[1, 0].imshow(processed_data, cmap='gray', origin='lower')
    
    # 입자 표시 (최대 100개)
    sample_size = min(100, n_particles)
    sample_indices = np.random.choice(n_particles, sample_size, replace=False)
    
    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap_colors = plt.cm.viridis
    
    for idx in sample_indices:
        particle = df.iloc[idx]
        
        if roi_coords is not None:
            x = particle['Centroid_X'] - roi_coords[0]
            y = particle['Centroid_Y'] - roi_coords[1]
        else:
            x = particle['Centroid_X']
            y = particle['Centroid_Y']
        
        r = particle[f'Diameter_{x_units}'] / (2 * pixel_size_x)
        color = cmap_colors(norm(particle[f'Diameter_{x_units}']))
        
        circle = Circle((x, y), r, fill=False, color=color, linewidth=1.5, alpha=0.8)
        axes[1, 0].add_patch(circle)
    
    axes[1, 0].set_title(f'Detected Particles ({sample_size} shown)')
    
    sm = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 0])
    cbar.set_label(f'Diameter ({x_units})')
    
    # 9.4 히스토그램
    n_bins = min(30, max(10, n_particles//10))
    counts, edges, _ = axes[1, 1].hist(sizes, bins=n_bins, alpha=0.7,
                                       color='blue', edgecolor='black', density=True)
    
    # 가우시안 피팅
    try:
        def gaussian(x, amp, mean, std):
            return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        centers = (edges[:-1] + edges[1:]) / 2
        popt, _ = curve_fit(gaussian, centers, counts,
                           p0=[counts.max(), sizes.mean(), sizes.std()],
                           maxfev=2000)
        
        amp, mean, std = popt
        x_fit = np.linspace(sizes.min(), sizes.max(), 300)
        y_fit = gaussian(x_fit, amp, mean, std)
        
        axes[1, 1].plot(x_fit, y_fit, 'r-', linewidth=2,
                       label=f'Mean: {mean:.3f} {x_units}\nStd: {std:.3f} {x_units}')
        axes[1, 1].axvline(mean, color='red', linestyle='--', alpha=0.7)
    except:
        pass
    
    axes[1, 1].axvline(sizes.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {sizes.median():.3f} {x_units}')
    
    axes[1, 1].set_xlabel(f'Diameter ({x_units})', fontweight='bold')
    axes[1, 1].set_ylabel('Probability Density', fontweight='bold')
    axes[1, 1].set_title('Size Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 10. 저장
    print("\n💾 결과 저장 중...")
    
    suffix = f"_improved_{n_particles}p"
    if roi_coords:
        suffix += f"_ROI"
    
    fig.savefig(f'detection_result{suffix}.png', dpi=300, bbox_inches='tight')
    df.to_csv(f'particle_data{suffix}.csv', index=False)
    
    print(f"  ✅ detection_result{suffix}.png")
    print(f"  ✅ particle_data{suffix}.csv")
    
    print("\n" + "="*70)
    print("✅ 분석 완료!")
    print("="*70)
    print(f"\n📊 최종 결과:")
    print(f"  입자 수: {n_particles}개")
    print(f"  평균 크기: {sizes.mean():.3f} {x_units}")
    print(f"  중앙값: {sizes.median():.3f} {x_units}")
    print(f"  임계값 방법: {method_name}")
    
    plt.show()
    
else:
    print("\n❌ 검출된 입자가 없습니다.")
    print("💡 임계값을 조정하거나 전처리 매개변수를 변경해보세요.")
