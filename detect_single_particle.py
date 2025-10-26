"""
정확한 좌표 기반 입자 검출
- 사용자 지정 좌표: (102, 220) - ROI 내 좌표
- 해당 위치 중심으로 입자 검출
- 타원형 입자 지원 (가로가 더 긴 형태)
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse
from skimage import filters, measure, morphology
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 정확한 좌표 기반 입자 검출")
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

# ROI 설정 (사용자가 본 영역)
center_x = 1024
center_y = 1024
roi_size = 250

half = roi_size // 2
x_start = center_x - half
y_start = center_y - half
x_end = center_x + half
y_end = center_y + half

roi_coords = (x_start, y_start, x_end, y_end)
analysis_data = signal.data[y_start:y_end, x_start:x_end].copy()

print(f"✅ ROI: ({x_start},{y_start}) → ({x_end},{y_end})")

# 사용자가 지정한 입자 좌표 (ROI 내 로컬 좌표)
target_x = 98
target_y = 219

print(f"🎯 목표 입자 위치 (ROI 내): ({target_x}, {target_y})")
print(f"   전체 이미지 좌표: ({target_x + x_start}, {target_y + y_start})")

# 전처리
print("\n🎨 전처리...")
from scipy.ndimage import gaussian_filter

# 백그라운드 제거
background = gaussian_filter(analysis_data, sigma=20)
bg_removed = analysis_data - background
bg_removed = np.clip(bg_removed, 0, None)

# 정규화
p1, p99 = np.percentile(bg_removed, [1, 99])
normalized = np.clip((bg_removed - p1) / (p99 - p1), 0, 1)

# 약간의 블러
smoothed = gaussian_filter(normalized, sigma=0.8)

# 이미지 반전
inverted = 1.0 - smoothed

print("✅ 전처리 완료")

# 목표 지점 주변 강도 확인
print(f"\n📊 목표 위치 강도 분석:")
window_size = 10
y_min = max(0, target_y - window_size)
y_max = min(analysis_data.shape[0], target_y + window_size)
x_min = max(0, target_x - window_size)
x_max = min(analysis_data.shape[1], target_x + window_size)

local_patch = inverted[y_min:y_max, x_min:x_max]
center_intensity = inverted[target_y, target_x]
local_mean = np.mean(local_patch)
local_max = np.max(local_patch)
local_min = np.min(local_patch)

print(f"  중심 강도: {center_intensity:.4f}")
print(f"  주변 평균: {local_mean:.4f}")
print(f"  주변 최대: {local_max:.4f}")
print(f"  주변 최소: {local_min:.4f}")

# 목표 지점의 강도를 기준으로 임계값 설정
# 중심 강도의 90%를 임계값으로 사용
threshold_suggested = center_intensity * 0.9

print(f"\n💡 제안 임계값: {threshold_suggested:.4f} (중심 강도의 90%)")

# 임계값 자동 계산들
thresh_otsu = filters.threshold_otsu(inverted)
thresh_li = filters.threshold_li(inverted)

print(f"   비교 - Otsu: {thresh_otsu:.4f}")
print(f"   비교 - Li: {thresh_li:.4f}")

# 여러 임계값 시도
print(f"\n🔍 여러 임계값으로 검출 테스트:")

threshold_candidates = [
    ("중심 90%", threshold_suggested),
    ("중심 80%", center_intensity * 0.8),
    ("중심 70%", center_intensity * 0.7),
    ("Otsu", thresh_otsu),
    ("Li", thresh_li),
]

best_threshold = None
best_score = float('inf')

for name, threshold in threshold_candidates:
    binary = inverted > threshold
    binary_cleaned = morphology.remove_small_objects(binary, min_size=50)
    labeled = measure.label(binary_cleaned)
    
    # 목표 지점이 포함된 라벨 찾기
    label_at_target = labeled[target_y, target_x]
    
    if label_at_target > 0:
        # 해당 라벨만 추출
        particle_mask = labeled == label_at_target
        area = np.sum(particle_mask)
        
        # 점수: 면적이 300~1000 픽셀 범위에 있으면 좋음 (약 0.3~1.1 nm²)
        ideal_area = 700  # 약 1nm 직경
        score = abs(area - ideal_area)
        
        print(f"  {name:12s} ({threshold:.4f}): 라벨 {label_at_target}, 면적 {area:4.0f} 픽셀²", end="")
        
        if area < 100:
            print(" ❌ 너무 작음")
        elif area > 5000:
            print(" ❌ 너무 큼")
        else:
            print(" ✅")
            if score < best_score:
                best_score = score
                best_threshold = (name, threshold)
    else:
        print(f"  {name:12s} ({threshold:.4f}): 검출 안됨 ❌")

if best_threshold is None:
    print("\n❌ 적절한 임계값을 찾지 못했습니다.")
    print("💡 수동으로 임계값을 입력하세요.")
    threshold = float(input("임계값 (0-1): "))
    method_name = "Manual"
else:
    method_name, threshold = best_threshold
    print(f"\n✅ 최적 임계값: {method_name} ({threshold:.4f})")

# 최종 검출
print(f"\n🎯 최종 검출 (임계값={threshold:.4f})...")

binary = inverted > threshold
binary_cleaned = morphology.remove_small_objects(binary, min_size=50)
binary_cleaned = morphology.binary_opening(binary_cleaned, morphology.disk(1))

labeled = measure.label(binary_cleaned)
label_at_target = labeled[target_y, target_x]

if label_at_target == 0:
    print("❌ 목표 지점에서 입자가 검출되지 않았습니다.")
    print("💡 임계값을 낮춰보세요.")
else:
    # 해당 입자만 추출
    particle_mask = labeled == label_at_target
    props = measure.regionprops(particle_mask.astype(int), intensity_image=smoothed)[0]
    
    # 측정
    area_pixels = props.area
    centroid_y, centroid_x = props.centroid
    
    # 등가 직경 (원형 가정)
    equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
    equiv_diameter_nm = equiv_diameter_pixels * pixel_size_x
    
    # 타원 파라미터
    major_axis = props.major_axis_length * pixel_size_x
    minor_axis = props.minor_axis_length * pixel_size_x
    orientation = props.orientation  # 라디안
    
    # 종횡비
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    
    # 원형도
    perimeter = props.perimeter
    circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
    
    print(f"\n✅ 입자 검출 성공!")
    print(f"="*70)
    print(f"📊 입자 특성:")
    print(f"  위치 (ROI 내): ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"  위치 (전체): ({centroid_x + x_start:.1f}, {centroid_y + y_start:.1f})")
    print(f"  면적: {area_pixels:.0f} 픽셀² ({area_pixels * pixel_size_x**2:.4f} {x_units}²)")
    print(f"")
    print(f"📏 크기 측정:")
    print(f"  등가 직경: {equiv_diameter_nm:.3f} {x_units} (원형 가정)")
    print(f"  장축 길이: {major_axis:.3f} {x_units}")
    print(f"  단축 길이: {minor_axis:.3f} {x_units}")
    print(f"  종횡비: {aspect_ratio:.2f} {'(타원형 ✓)' if aspect_ratio > 1.2 else '(거의 원형)'}")
    print(f"  원형도: {circularity:.3f}")
    print(f"  방향: {np.degrees(orientation):.1f}°")
    
    # 전체 이미지 좌표
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
        'Detection_Method': method_name,
        'Threshold': threshold
    }
    
    df = pd.DataFrame([particle_data])
    
    # 시각화
    print(f"\n📈 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('정확한 좌표 기반 입자 검출 결과', fontsize=16, fontweight='bold')
    
    # 1. 원본 전체
    vmin, vmax = np.percentile(analysis_data, [2, 98])
    axes[0, 0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].plot(target_x, target_y, 'r+', markersize=15, markeredgewidth=2)
    axes[0, 0].set_title('원본 (목표 위치 표시)')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    
    # 2. 전처리 (반전)
    axes[0, 1].imshow(inverted, cmap='gray', origin='lower')
    axes[0, 1].plot(target_x, target_y, 'r+', markersize=15, markeredgewidth=2)
    axes[0, 1].set_title('전처리 (반전)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    
    # 3. 이진화
    axes[0, 2].imshow(particle_mask, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'이진화 (threshold={threshold:.3f})')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    
    # 4. 검출 결과 - 원으로 표시
    axes[1, 0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    circle = Circle((centroid_x, centroid_y), equiv_diameter_pixels/2, 
                   fill=False, color='red', linewidth=2.5)
    axes[1, 0].add_patch(circle)
    axes[1, 0].plot(centroid_x, centroid_y, 'r+', markersize=12, markeredgewidth=2)
    axes[1, 0].text(centroid_x, centroid_y - equiv_diameter_pixels/2 - 10,
                   f"{equiv_diameter_nm:.2f} {x_units}",
                   ha='center', va='bottom', fontsize=11, color='yellow', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    axes[1, 0].set_title('검출 결과 (등가직경 원)')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    
    # 5. 검출 결과 - 타원으로 표시
    axes[1, 1].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ellipse = Ellipse((centroid_x, centroid_y), 
                     width=major_axis/pixel_size_x,
                     height=minor_axis/pixel_size_x,
                     angle=np.degrees(orientation),
                     fill=False, color='lime', linewidth=2.5)
    axes[1, 1].add_patch(ellipse)
    axes[1, 1].plot(centroid_x, centroid_y, 'r+', markersize=12, markeredgewidth=2)
    axes[1, 1].text(centroid_x, centroid_y - major_axis/(2*pixel_size_x) - 10,
                   f"{major_axis:.2f}×{minor_axis:.2f} {x_units}",
                   ha='center', va='bottom', fontsize=11, color='yellow', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    axes[1, 1].set_title('검출 결과 (실제 타원)')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    
    # 6. 확대 뷰
    zoom_size = 50
    y_zoom_min = max(0, int(centroid_y) - zoom_size)
    y_zoom_max = min(analysis_data.shape[0], int(centroid_y) + zoom_size)
    x_zoom_min = max(0, int(centroid_x) - zoom_size)
    x_zoom_max = min(analysis_data.shape[1], int(centroid_x) + zoom_size)
    
    zoom_patch = analysis_data[y_zoom_min:y_zoom_max, x_zoom_min:x_zoom_max]
    axes[1, 2].imshow(zoom_patch, cmap='gray', origin='lower', 
                     extent=[x_zoom_min, x_zoom_max, y_zoom_min, y_zoom_max],
                     vmin=vmin, vmax=vmax)
    
    # 타원 추가
    ellipse_zoom = Ellipse((centroid_x, centroid_y),
                          width=major_axis/pixel_size_x,
                          height=minor_axis/pixel_size_x,
                          angle=np.degrees(orientation),
                          fill=False, color='lime', linewidth=2)
    axes[1, 2].add_patch(ellipse_zoom)
    axes[1, 2].plot(centroid_x, centroid_y, 'r+', markersize=10, markeredgewidth=2)
    axes[1, 2].set_title('확대 뷰')
    axes[1, 2].set_xlabel('X (pixels)')
    axes[1, 2].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    # 저장
    fig.savefig('particle_detection_single.png', dpi=300, bbox_inches='tight')
    df.to_csv('particle_data_single.csv', index=False)
    
    print(f"\n💾 저장 완료:")
    print(f"  particle_detection_single.png")
    print(f"  particle_data_single.csv")
    
    print(f"\n" + "="*70)
    print(f"✅ 입자 분석 완료!")
    print(f"="*70)
    
    plt.show()
