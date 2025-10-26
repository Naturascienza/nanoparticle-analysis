"""
개선된 나노입자 검출 알고리즘 (최종 버전)
- Watershed 분할로 응집 입자 분리
- CLAHE로 국소 대비 향상
- 적절한 매개변수로 50개 미만 입자 검출
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation, feature, exposure
from scipy import ndimage
from matplotlib.patches import Circle
import pandas as pd

# 파일 로드
signal = hs.load("Camera Ceta 1112 380 kx 0001.emd")
if isinstance(signal, list):
    signal = signal[0]

# 픽셀 크기 정보
x_axis = signal.axes_manager.signal_axes[0]
pixel_size_x = x_axis.scale
x_units = x_axis.units

print("=" * 80)
print("🔬 개선된 나노입자 검출 알고리즘 (최종 버전)")
print("=" * 80)
print(f"픽셀 크기: {pixel_size_x:.4f} {x_units}/픽셀")

# 1. 이미지 전처리
print("\n1️⃣ 이미지 전처리...")
# 초기 정규화
vmin_init, vmax_init = np.percentile(signal.data, [2, 98])
normalized = np.clip(signal.data, vmin_init, vmax_init)
normalized = (normalized - vmin_init) / (vmax_init - vmin_init)

# 가우시안 필터로 노이즈 제거
smoothed = filters.gaussian(normalized, sigma=1.2)
print("   ✓ 가우시안 필터 적용 (sigma=1.2)")

# CLAHE로 국소 대비 향상
clahe = exposure.equalize_adapthist(smoothed, clip_limit=0.02)
print("   ✓ CLAHE 적용 (국소 대비 향상)")

# 어두운 입자를 위해 반전
inverted_data = 1.0 - clahe
print("   ✓ 이미지 반전 완료")

# 2. 이진화
print("\n2️⃣ 이진화...")
# 적절한 임계값과 최소 크기 설정
threshold = 0.56  # 더 낮은 임계값으로 더 많은 입자 검출
min_particle_size = 1100  # 약 1.3nm 직경

binary = inverted_data > threshold
cleaned = morphology.remove_small_objects(binary, min_size=min_particle_size)
cleaned = morphology.remove_small_holes(cleaned, area_threshold=350)
print(f"   ✓ 임계값: {threshold}")
print(f"   ✓ 최소 입자 크기: {min_particle_size} 픽셀²")

# 3. Watershed 분할
print("\n3️⃣ Watershed 분할 (응집 입자 분리)...")
distance = ndimage.distance_transform_edt(cleaned)

# footprint를 적절히 설정하여 응집된 입자 분리
# 8x8 footprint: 약 0.27nm x 0.27nm 영역
# 실제 입자 직경이 2-3nm이므로 적절한 분리
coords = feature.peak_local_max(distance, 
                               footprint=np.ones((8, 8)),  # 분리 정도 조절
                               labels=cleaned,
                               threshold_rel=0.5)  # 상대 임계값 (높을수록 적게 분리)
print(f"   ✓ {len(coords)}개의 입자 중심점 발견")

# 마커 생성
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)

# Watershed 알고리즘 적용
labeled = segmentation.watershed(-distance, markers, mask=cleaned)
n_particles = labeled.max()
print(f"   ✅ Watershed 후: {n_particles}개 영역")

# 4. 입자 속성 측정 및 필터링
print("\n4️⃣ 입자 필터링...")
props = measure.regionprops(labeled)

filtered_props = []
for prop in props:
    area_nm2 = prop.area * (pixel_size_x ** 2)
    diameter_nm = np.sqrt(4 * area_nm2 / np.pi)
    
    # 크기 필터: 1.0nm ~ 20nm (더 넓은 범위)
    if diameter_nm < 1.0 or diameter_nm > 20:
        continue
    
    # 종횡비 필터: 매우 길쭉한 것만 제외 (완화)
    if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            if aspect_ratio > 5.0:  # 더 완화
                continue
    
    # 원형도 필터 거의 적용 안함 (노이즈만 제거)
    # lattice fringe를 포함한 입자는 원형도가 낮을 수 있음
    if hasattr(prop, 'perimeter') and prop.perimeter > 0:
        circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        if circularity < 0.05:  # 극도로 불규칙한 것만 제외
            continue
    
    filtered_props.append(prop)

print(f"   초기 검출: {len(props)}개")
print(f"   필터링 후: {len(filtered_props)}개")
print(f"   제거된 입자: {len(props) - len(filtered_props)}개")

# 5. 데이터 수집
particle_data = []
for i, prop in enumerate(filtered_props, 1):
    area_pixels = prop.area
    area_nm2 = area_pixels * (pixel_size_x ** 2)
    equiv_diameter_nm = np.sqrt(4 * area_nm2 / np.pi)
    
    y, x = prop.centroid
    
    major_axis = prop.major_axis_length * pixel_size_x if hasattr(prop, 'major_axis_length') else equiv_diameter_nm
    minor_axis = prop.minor_axis_length * pixel_size_x if hasattr(prop, 'minor_axis_length') else equiv_diameter_nm
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    
    perimeter = prop.perimeter if hasattr(prop, 'perimeter') else 2 * np.sqrt(np.pi * area_pixels)
    circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
    
    particle_data.append({
        'ID': i,
        'X': x,
        'Y': y,
        f'Diameter_{x_units}': equiv_diameter_nm,
        f'Area_{x_units}2': area_nm2,
        f'Major_Axis_{x_units}': major_axis,
        f'Minor_Axis_{x_units}': minor_axis,
        'Aspect_Ratio': aspect_ratio,
        'Circularity': circularity
    })

df = pd.DataFrame(particle_data)

# 6. 통계
print(f"\n📊 검출 통계 (등가직경, {x_units}):")
print(f"   총 입자 수: {len(df)}개")
print(f"   평균: {df[f'Diameter_{x_units}'].mean():.2f} {x_units}")
print(f"   중앙값: {df[f'Diameter_{x_units}'].median():.2f} {x_units}")
print(f"   표준편차: {df[f'Diameter_{x_units}'].std():.2f} {x_units}")
print(f"   최소: {df[f'Diameter_{x_units}'].min():.2f} {x_units}")
print(f"   최대: {df[f'Diameter_{x_units}'].max():.2f} {x_units}")

bins = [0, 1.5, 2, 2.5, 3, 4, 5, 10, 100]
bin_labels = ['<1.5', '1.5-2', '2-2.5', '2.5-3', '3-4', '4-5', '5-10', '>10']
df['Size_Category'] = pd.cut(df[f'Diameter_{x_units}'], bins=bins, labels=bin_labels)

print(f"\n📈 크기별 분포:")
size_dist = df['Size_Category'].value_counts().sort_index()
for category, count in size_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   {category:>6} nm: {count:3d}개 ({percentage:5.1f}%)")

# 7. 상위 20개
print(f"\n🏆 가장 큰 입자 Top 20:")
print("-" * 80)
top20 = df.nlargest(min(20, len(df)), f'Diameter_{x_units}')
for idx, row in top20.iterrows():
    print(f"{row['ID']:3d}. 위치: ({row['X']:6.1f}, {row['Y']:6.1f}), "
          f"크기: {row[f'Diameter_{x_units}']:6.2f} {x_units}, "
          f"종횡비: {row['Aspect_Ratio']:4.2f}, "
          f"원형도: {row['Circularity']:4.2f}")

# CSV 저장
csv_filename = f'final_particle_detection_{len(df)}particles.csv'
df.to_csv(csv_filename, index=False)
print(f"\n💾 데이터 저장: {csv_filename}")

# 8. 시각화 - 상위 10개 확대
print(f"\n🎨 상위 10개 입자 상세 분석...")
top10 = df.nlargest(min(10, len(df)), f'Diameter_{x_units}')

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(top10.iterrows()):
    if idx >= 10:
        break
    
    ax = axes[idx]
    
    x_center, y_center = int(row['X']), int(row['Y'])
    zoom_size = 150
    
    y_start = max(0, y_center - zoom_size)
    y_end = min(signal.data.shape[0], y_center + zoom_size)
    x_start = max(0, x_center - zoom_size)
    x_end = min(signal.data.shape[1], x_center + zoom_size)
    
    zoom_img = signal.data[y_start:y_end, x_start:x_end]
    zoom_labeled = labeled[y_start:y_end, x_start:x_end]
    
    zoom_vmin, zoom_vmax = np.percentile(zoom_img, [2, 98])
    ax.imshow(zoom_img, cmap='gray', origin='lower', vmin=zoom_vmin, vmax=zoom_vmax)
    
    # 입자 경계 표시
    contours = measure.find_contours(zoom_labeled > 0, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2.5)
    
    # 중심점 표시
    local_x = x_center - x_start
    local_y = y_center - y_start
    ax.plot(local_x, local_y, 'y+', markersize=20, markeredgewidth=3)
    
    ax.set_title(f'Particle #{row["ID"]:d}\n'
                f'Size: {row[f"Diameter_{x_units}"]:.2f} {x_units}\n'
                f'Aspect: {row["Aspect_Ratio"]:.2f}, Circ: {row["Circularity"]:.2f}',
                fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # 스케일바 (5nm)
    scale_nm = 5
    scale_px = scale_nm / pixel_size_x
    scale_y_pos = zoom_img.shape[0] - 15
    ax.plot([15, 15 + scale_px], [scale_y_pos, scale_y_pos], 'w-', linewidth=3)
    ax.text(15 + scale_px/2, scale_y_pos - 10, f'{scale_nm} nm',
           ha='center', va='top', color='white', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.suptitle(f'Top {len(top10)} Particles - Detailed Analysis (Total: {len(df)} particles)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('final_top_particles_detailed.png', dpi=300, bbox_inches='tight')
print(f"✅ 저장: final_top_particles_detailed.png")

# 9. 전체 검출 결과 및 크기 분포
print(f"\n🎨 전체 검출 결과 시각화...")
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

# (0) 원본 이미지
axes[0].imshow(signal.data, cmap='gray', origin='lower')
axes[0].set_title('Original TEM Image', fontsize=16, fontweight='bold')
axes[0].axis('off')

# (1) Watershed 분할 결과
im = axes[1].imshow(labeled, cmap='nipy_spectral', origin='lower')
axes[1].set_title(f'Watershed Segmentation ({n_particles} regions)', 
                 fontsize=16, fontweight='bold')
axes[1].axis('off')

# (2) 최종 검출 결과
axes[2].imshow(signal.data, cmap='gray', origin='lower')
for _, row in df.iterrows():
    x, y = row['X'], row['Y']
    radius = (row[f'Diameter_{x_units}'] / pixel_size_x) / 2
    circle = Circle((x, y), radius, fill=False, color='red', linewidth=2, alpha=0.9)
    axes[2].add_patch(circle)
    if row['ID'] <= 20:
        axes[2].text(x, y, str(int(row['ID'])), ha='center', va='center',
                    color='yellow', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.6))

axes[2].set_title(f'Final Detection: {len(df)} Particles', fontsize=16, fontweight='bold')
axes[2].axis('off')

# 스케일바
scale_length_nm = 50
scale_length_px = scale_length_nm / pixel_size_x
scale_y = signal.data.shape[0] - 100
scale_x_start = signal.data.shape[1] - scale_length_px - 100
scale_x_end = scale_x_start + scale_length_px
axes[2].plot([scale_x_start, scale_x_end], [scale_y, scale_y], 'w-', linewidth=6)
axes[2].text((scale_x_start + scale_x_end) / 2, scale_y - 35,
            f'{scale_length_nm} nm', ha='center', va='top',
            color='white', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))

plt.tight_layout()
plt.savefig('final_detection_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 저장: final_detection_comparison.png")

# 10. 크기 분포 히스토그램
fig, ax = plt.subplots(figsize=(12, 8))
sizes = df[f'Diameter_{x_units}']

ax.hist(sizes, bins=15, alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
ax.axvline(sizes.mean(), color='red', linestyle='--', linewidth=3,
          label=f'Mean: {sizes.mean():.2f} {x_units}')
ax.axvline(sizes.median(), color='green', linestyle='--', linewidth=3,
          label=f'Median: {sizes.median():.2f} {x_units}')

ax.set_xlabel(f'Particle Diameter ({x_units})', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title(f'Particle Size Distribution (n={len(df)})', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_size_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ 저장: final_size_distribution.png")

print("\n" + "=" * 80)
print("✅ 개선된 입자 검출 완료!")
print("=" * 80)
print(f"\n💡 개선 사항:")
print(f"   ✓ CLAHE로 국소 대비 향상 → 입자 경계 명확화")
print(f"   ✓ Watershed 분할 (footprint=10×10) → 응집 입자 적절히 분리")
print(f"   ✓ 형태 기반 필터링 → 노이즈 및 부적절한 영역 제거")
print(f"   ✓ 총 {len(df)}개 입자 검출")
print(f"   ✓ 평균 크기: {df[f'Diameter_{x_units}'].mean():.2f} nm")
print(f"\n📋 생성된 파일:")
print(f"   - {csv_filename}: 전체 입자 데이터")
print(f"   - final_top_particles_detailed.png: 상위 10개 입자 확대")
print(f"   - final_detection_comparison.png: 전체 검출 결과 비교")
print(f"   - final_size_distribution.png: 크기 분포 히스토그램")
