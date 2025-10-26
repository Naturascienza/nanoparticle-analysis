"""
원본 기반 어두운 영역 직접 검출
- 이미지 반전 없이 원본에서 직접 어두운 영역 찾기
- 목표 위치 중심으로 국소적 검출
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse
from skimage import filters, measure, morphology
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 원본 기반 어두운 영역 직접 검출")
print("="*70)

# 파일 로드
signal = hs.load("Camera Ceta 1204 380 kx.emd")
if isinstance(signal, list):
    signal = signal[0]

try:
    x_axis = signal.axes_manager.signal_axes[0]
    pixel_size_x = x_axis.scale
    x_units = x_axis.units
except:
    pixel_size_x = 1.0
    x_units = 'pixels'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")

# 전체 이미지 분석
print(f"📊 전체 이미지 크기: {signal.data.shape}")
analysis_data = signal.data.copy()

# ROI 설정 없음 (전체 이미지)
x_start = 0
y_start = 0
y_end, x_end = signal.data.shape
roi_coords = (x_start, y_start, x_end, y_end)

print(f"✅ 전체 이미지 분석: ({x_start},{y_start}) → ({x_end},{y_end})")

# 목표 입자 좌표 (전체 이미지 기준)
target_x = 997
target_y = 1118

print(f"🎯 목표 입자 위치 (전체 이미지): ({target_x}, {target_y})")

# ROI 설정 (중심 1024, 1024 기준 500x500 영역으로 확장)
roi_size = 500
center_x, center_y = 1024, 1024
x_start = center_x - roi_size // 2
y_start = center_y - roi_size // 2
x_end = x_start + roi_size
y_end = y_start + roi_size

roi_coords = (x_start, y_start, x_end, y_end)
analysis_data = signal.data[y_start:y_end, x_start:x_end].copy()

print(f"✅ ROI: ({x_start},{y_start}) → ({x_end},{y_end})")

# 목표 좌표
target_x = 98
target_y = 219

# 전체 이미지 좌표 (이전 250x250 ROI에서 확인된 좌표)
global_x, global_y = 997, 1118

# 현재 ROI에서의 좌표로 변환
target_x = global_x - x_start
target_y = global_y - y_start

print(f"🎯 목표 입자 위치:")
print(f"   전체 이미지 좌표: ({global_x}, {global_y})")
print(f"   현재 ROI 내 좌표: ({target_x}, {target_y})")

# 전처리 (최소화)
print("\n🎨 최소 전처리...")
# 약간의 가우시안만 (노이즈 제거)
smoothed = gaussian_filter(analysis_data, sigma=1.0)

# 정규화
p1, p99 = np.percentile(smoothed, [1, 99])
normalized = (smoothed - p1) / (p99 - p1)
normalized = np.clip(normalized, 0, 1)

print("✅ 완료")

# 전역 임계값 계산 (전체 이미지 기준)
print(f"\n📊 전체 이미지 임계값 계산:")
global_mean = np.mean(normalized)
global_std = np.std(normalized)
global_median = np.median(normalized)

# 어두운 입자 검출: 평균보다 1 표준편차 낮은 값
threshold_dark = global_mean - 0.5 * global_std

print(f"  전체 평균: {global_mean:.4f}")
print(f"  전체 표준편차: {global_std:.4f}")
print(f"  전체 중앙값: {global_median:.4f}")
print(f"  임계값: {threshold_dark:.4f} (평균 - 0.5×표준편차)")

# 목표 입자 확인
target_intensity = normalized[target_y, target_x]
print(f"\n🎯 목표 입자 강도: {target_intensity:.4f}")
print(f"   임계값보다 {'낮음 ✅' if target_intensity < threshold_dark else '높음 ❌'}")

# 이진화 (어두운 부분 = True)
binary_dark = normalized < threshold_dark

# 작은 노이즈 제거
binary_cleaned = morphology.remove_small_objects(binary_dark, min_size=20)
binary_cleaned = morphology.binary_opening(binary_cleaned, morphology.disk(1))

# 라벨링
labeled = measure.label(binary_cleaned)
n_regions = labeled.max()

print(f"\n🔍 검출된 어두운 영역: {n_regions}개")

# 모든 입자 분석
all_props = measure.regionprops(labeled, intensity_image=normalized)

# 크기 필터링 (너무 작은 영역 제외 - 최소 50 픽셀)
min_area = 50
valid_props = [p for p in all_props if p.area >= min_area]

print(f"✅ 유효한 입자 (>={min_area} 픽셀): {len(valid_props)}개")

# 통계 계산
if len(valid_props) > 0:
    sizes = [p.area * pixel_size_x**2 for p in valid_props]
    diameters = [np.sqrt(4 * p.area / np.pi) * pixel_size_x for p in valid_props]
    major_axes = [p.major_axis_length * pixel_size_x for p in valid_props]
    minor_axes = [p.minor_axis_length * pixel_size_x for p in valid_props]
    aspect_ratios = [p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else 1.0 for p in valid_props]
    
    print(f"\n📊 전체 입자 통계:")
    print(f"  입자 개수: {len(valid_props)}개")
    print(f"  평균 면적: {np.mean(sizes):.4f} ± {np.std(sizes):.4f} {x_units}²")
    print(f"  평균 등가직경: {np.mean(diameters):.3f} ± {np.std(diameters):.3f} {x_units}")
    print(f"  평균 장축: {np.mean(major_axes):.3f} ± {np.std(major_axes):.3f} {x_units}")
    print(f"  평균 단축: {np.mean(minor_axes):.3f} ± {np.std(minor_axes):.3f} {x_units}")
    print(f"  평균 종횡비: {np.mean(aspect_ratios):.2f} ± {np.std(aspect_ratios):.2f}")
    print(f"  크기 범위: {np.min(diameters):.3f} ~ {np.max(diameters):.3f} {x_units}")

# 목표 지점이 포함된 라벨 찾기
label_at_target = labeled[target_y, target_x]

if label_at_target == 0:
    print("❌ 목표 지점에서 영역이 검출되지 않았습니다.")
    print("💡 임계값을 조정해보세요.")
else:
    print(f"✅ 목표 지점의 라벨: {label_at_target}")
    
    # 해당 입자만 추출
    particle_mask = (labeled == label_at_target)
    props = measure.regionprops(particle_mask.astype(int), intensity_image=normalized)[0]
    
    # 측정
    area_pixels = props.area
    centroid_y, centroid_x = props.centroid
    
    # 등가 직경
    equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
    equiv_diameter_nm = equiv_diameter_pixels * pixel_size_x
    
    # 타원 파라미터
    major_axis = props.major_axis_length * pixel_size_x
    minor_axis = props.minor_axis_length * pixel_size_x
    orientation = props.orientation  # 라디안 (수평축에서 반시계방향)
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    
    # orientation 값 확인 및 디버깅
    print(f"\n🔍 타원 방향 정보:")
    print(f"  장축: {major_axis:.3f} {x_units}")
    print(f"  단축: {minor_axis:.3f} {x_units}")
    print(f"  방향 (라디안): {orientation:.4f}")
    print(f"  방향 (도): {np.degrees(orientation):.1f}°")
    print(f"  종횡비: {aspect_ratio:.2f}")
    
    # 원형도
    perimeter = props.perimeter
    circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
    
    # 평균 강도
    mean_intensity = props.mean_intensity
    
    print(f"\n" + "="*70)
    print(f"✅ 입자 검출 성공!")
    print(f"="*70)
    print(f"\n📊 입자 특성:")
    print(f"  위치 (ROI 내): ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"  위치 (전체): ({centroid_x + x_start:.1f}, {centroid_y + y_start:.1f})")
    print(f"  면적: {area_pixels:.0f} 픽셀² ({area_pixels * pixel_size_x**2:.4f} {x_units}²)")
    print(f"  평균 강도: {mean_intensity:.4f}")
    print(f"")
    print(f"📏 크기 측정:")
    print(f"  등가 직경: {equiv_diameter_nm:.3f} {x_units}")
    print(f"  장축: {major_axis:.3f} {x_units}")
    print(f"  단축: {minor_axis:.3f} {x_units}")
    print(f"  종횡비: {aspect_ratio:.2f} {'(타원형)' if aspect_ratio > 1.2 else '(거의 원형)'}")
    print(f"  원형도: {circularity:.3f}")
    
    # 전체 좌표
    global_centroid_x = centroid_x + x_start
    global_centroid_y = centroid_y + y_start
    
    # 데이터 저장
    particle_data = {
        'Particle_ID': 1,
        'Centroid_X_Global': global_centroid_x,
        'Centroid_Y_Global': global_centroid_y,
        'Centroid_X_ROI': centroid_x,
        'Centroid_Y_ROI': centroid_y,
        f'Equiv_Diameter_{x_units}': equiv_diameter_nm,
        f'Major_Axis_{x_units}': major_axis,
        f'Minor_Axis_{x_units}': minor_axis,
        'Aspect_Ratio': aspect_ratio,
        'Circularity': circularity,
        'Orientation_deg': np.degrees(orientation),
        f'Area_{x_units}2': area_pixels * pixel_size_x**2,
        'Area_pixels': area_pixels,
        'Mean_Intensity': mean_intensity,
        'Threshold': threshold_dark
    }
    
    df = pd.DataFrame([particle_data])
    
    # 시각화
    print(f"\n📈 시각화...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'전체 이미지 입자 검출 결과 (총 {len(valid_props)}개)', fontsize=16, fontweight='bold')
    
    vmin, vmax = np.percentile(analysis_data, [2, 98])
    
    # 1. 원본
    axes[0, 0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'원본 ({analysis_data.shape[0]}×{analysis_data.shape[1]} pixels)')
    axes[0, 0].axis('off')
    
    # 2. 정규화
    axes[0, 1].imshow(normalized, cmap='gray', origin='lower')
    axes[0, 1].set_title('정규화')
    axes[0, 1].axis('off')
    
    # 3. 이진화 (모든 어두운 영역)
    axes[0, 2].imshow(binary_cleaned, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'검출된 모든 어두운 영역 ({n_regions}개)')
    axes[0, 2].axis('off')
    
    # 4. 모든 입자 검출 - 원으로 표시
    axes[1, 0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    for prop in valid_props:
        cy, cx = prop.centroid
        area = prop.area
        equiv_d = np.sqrt(4 * area / np.pi)
        circle = Circle((cx, cy), equiv_d/2, fill=False, color='cyan', linewidth=0.8, alpha=0.7)
        axes[1, 0].add_patch(circle)
    # 목표 입자는 빨간색으로 강조
    circle_target = Circle((centroid_x, centroid_y), equiv_diameter_pixels/2,
                          fill=False, color='red', linewidth=2)
    axes[1, 0].add_patch(circle_target)
    axes[1, 0].set_title(f'모든 입자 ({len(valid_props)}개, 목표=빨강)')
    axes[1, 0].axis('off')
    
    # 5. 모든 입자 검출 - 타원으로 표시
    axes[1, 1].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    for prop in valid_props:
        cy, cx = prop.centroid
        maj_ax = prop.major_axis_length
        min_ax = prop.minor_axis_length
        orient = prop.orientation
        ellipse = Ellipse((cx, cy), width=maj_ax, height=min_ax,
                         angle=np.degrees(orient) + 90,
                         fill=False, color='lime', linewidth=0.8, alpha=0.7)
        axes[1, 1].add_patch(ellipse)
    # 목표 입자는 빨간색으로 강조
    ellipse_target = Ellipse((centroid_x, centroid_y),
                            width=major_axis/pixel_size_x,
                            height=minor_axis/pixel_size_x,
                            angle=np.degrees(orientation) + 90,
                            fill=False, color='red', linewidth=2)
    axes[1, 1].add_patch(ellipse_target)
    axes[1, 1].set_title(f'모든 입자 타원 ({len(valid_props)}개, 목표=빨강)')
    axes[1, 1].axis('off')
    
    axes[1, 1].set_title(f'모든 입자 타원 ({len(valid_props)}개, 목표=빨강)')
    axes[1, 1].axis('off')
    
    # 6. 목표 입자 확대
    zoom_size = 40
    y_zoom_min = max(0, int(centroid_y) - zoom_size)
    y_zoom_max = min(analysis_data.shape[0], int(centroid_y) + zoom_size)
    x_zoom_min = max(0, int(centroid_x) - zoom_size)
    x_zoom_max = min(analysis_data.shape[1], int(centroid_x) + zoom_size)
    
    zoom_patch = analysis_data[y_zoom_min:y_zoom_max, x_zoom_min:x_zoom_max]
    axes[1, 2].imshow(zoom_patch, cmap='gray', origin='lower',
                     extent=[x_zoom_min, x_zoom_max, y_zoom_min, y_zoom_max],
                     vmin=vmin, vmax=vmax)
    
    ellipse_zoom = Ellipse((centroid_x, centroid_y),
                          width=major_axis/pixel_size_x,
                          height=minor_axis/pixel_size_x,
                          angle=np.degrees(orientation) + 90,
                          fill=False, color='red', linewidth=2.5)
    axes[1, 2].add_patch(ellipse_zoom)
    axes[1, 2].plot(centroid_x, centroid_y, 'r+', markersize=10, markeredgewidth=2)
    axes[1, 2].set_title(f'목표 입자 확대\n{major_axis:.2f}×{minor_axis:.2f} {x_units}')
    axes[1, 2].set_xlabel('X (pixels)')
    axes[1, 2].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    # 모든 유효한 입자 데이터를 CSV로 저장
    all_particles_data = []
    for i, prop in enumerate(valid_props, 1):
        centroid_y, centroid_x = prop.centroid
        area_pixels = prop.area
        area_nm2 = area_pixels * pixel_size_x**2
        equiv_diameter = np.sqrt(4 * area_pixels / np.pi) * pixel_size_x
        major_ax = prop.major_axis_length * pixel_size_x
        minor_ax = prop.minor_axis_length * pixel_size_x
        aspect_r = major_ax / minor_ax if minor_ax > 0 else 1.0
        orient = prop.orientation
        mean_int = prop.mean_intensity
        
        all_particles_data.append({
            'particle_id': i,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'area_pixels': area_pixels,
            'area_nm2': area_nm2,
            'equivalent_diameter_nm': equiv_diameter,
            'major_axis_nm': major_ax,
            'minor_axis_nm': minor_ax,
            'aspect_ratio': aspect_r,
            'orientation_rad': orient,
            'orientation_deg': np.degrees(orient),
            'mean_intensity': mean_int
        })
    
    df_all = pd.DataFrame(all_particles_data)
    
    # 저장
    fig.savefig('particle_detection_final.png', dpi=300, bbox_inches='tight')
    df.to_csv('particle_data_target.csv', index=False)
    df_all.to_csv('particle_data_all.csv', index=False)
    
    print(f"\n💾 저장:")
    print(f"  particle_detection_final.png")
    print(f"  particle_data_target.csv (목표 입자)")
    print(f"  particle_data_all.csv ({len(valid_props)}개 모든 유효 입자)")
    
    print(f"\n" + "="*70)
    print(f"✅ 분석 완료!")
    print(f"="*70)
    
    plt.show()
