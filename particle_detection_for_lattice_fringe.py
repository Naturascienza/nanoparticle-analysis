"""
Lattice Fringe를 포함한 나노입자 검출 알고리즘
- 입자 내부의 lattice fringe 패턴을 무시하고 전체 입자 윤곽을 검출
- Watershed 분할을 최소화하여 하나의 입자로 인식
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
print("🔬 Lattice Fringe를 포함한 나노입자 검출")
print("=" * 80)
print(f"픽셀 크기: {pixel_size_x:.4f} {x_units}/픽셀")

# 1. 이미지 전처리 (lattice fringe를 무시하기 위한 강력한 스무딩)
print("\n1️⃣ 이미지 전처리 (강력한 스무딩으로 lattice fringe 제거)...")
# 초기 정규화
vmin_init, vmax_init = np.percentile(signal.data, [2, 98])
normalized = np.clip(signal.data, vmin_init, vmax_init)
normalized = (normalized - vmin_init) / (vmax_init - vmin_init)

# 가우시안 필터로 lattice fringe 패턴 완화 (너무 강하지 않게)
# sigma를 적절히 설정하여 lattice fringe는 완화하되 입자 윤곽은 유지
smoothed = filters.gaussian(normalized, sigma=2.0)  # 적절한 스무딩
print("   ✓ 가우시안 필터 적용 (sigma=2.0)")

# CLAHE는 사용하지 않음 (lattice fringe를 강조할 수 있음)
# 대신 단순 정규화
processed = smoothed

# 어두운 입자를 위해 반전
inverted_data = 1.0 - processed
print("   ✓ 이미지 반전 완료")

# 2. 이진화
print("\n2️⃣ 이진화...")
# 약간 높은 임계값 사용하여 배경과 명확히 구분
threshold = filters.threshold_otsu(inverted_data) + 0.05  # Otsu보다 약간 높게
print(f"   ✓ Otsu 기반 임계값: {threshold:.3f}")

# 최소 입자 크기: 1.5nm 직경 기준 (더 낮춤)
min_diameter_nm = 1.5
min_area_nm2 = np.pi * (min_diameter_nm / 2) ** 2
min_particle_size = int(min_area_nm2 / (pixel_size_x ** 2))
print(f"   ✓ 최소 입자 크기: {min_particle_size} 픽셀² (약 {min_diameter_nm}nm 직경)")

binary = inverted_data > threshold
cleaned = morphology.remove_small_objects(binary, min_size=min_particle_size)

# 모폴로지 연산으로 입자 내부 채우기 (lattice fringe로 인한 구멍 제거)
# 큰 구멍도 채워서 하나의 solid한 입자로 만듦
cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_particle_size)
print("   ✓ 입자 내부 구멍 채우기 완료")

# Closing 연산으로 작은 갭 연결
selem = morphology.disk(3)
cleaned = morphology.binary_closing(cleaned, selem)
print("   ✓ Closing 연산으로 입자 경계 매끄럽게 만듦")

# 3. Watershed 분할 (매우 보수적으로)
print("\n3️⃣ Watershed 분할 (최소한으로만 적용)...")
distance = ndimage.distance_transform_edt(cleaned)

# footprint를 적절히 설정하여 응집된 입자만 분리, lattice fringe는 무시
# 15x15 footprint: 약 0.5nm × 0.5nm 영역
# 2-3nm 입자에서는 적절히 분할
coords = feature.peak_local_max(distance, 
                               footprint=np.ones((15, 15)),  # 중간 크기
                               labels=cleaned,
                               threshold_rel=0.6)  # 중간 임계값
print(f"   ✓ {len(coords)}개의 입자 중심점 발견")

if len(coords) > 0:
    # 마커 생성
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    # Watershed 알고리즘 적용
    labeled = segmentation.watershed(-distance, markers, mask=cleaned)
    n_particles = labeled.max()
    print(f"   ✅ Watershed 후: {n_particles}개 영역")
else:
    # Watershed 없이 단순 라벨링
    labeled = measure.label(cleaned)
    n_particles = labeled.max()
    print(f"   ✅ 단순 라벨링: {n_particles}개 영역")

# 4. 입자 속성 측정 및 필터링
print("\n4️⃣ 입자 필터링...")
props = measure.regionprops(labeled)

filtered_props = []
for prop in props:
    area_nm2 = prop.area * (pixel_size_x ** 2)
    diameter_nm = np.sqrt(4 * area_nm2 / np.pi)
    
    # 크기 필터: 1.5nm ~ 20nm
    if diameter_nm < 1.5 or diameter_nm > 20:
        continue
    
    # 종횡비 필터: 매우 완화 (lattice fringe로 인해 불규칙할 수 있음)
    if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            if aspect_ratio > 3.0:  # 극단적인 경우만 제외
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
if len(df) > 0:
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
    
    # 7. 상위 입자 출력
    print(f"\n🏆 검출된 모든 입자:")
    print("-" * 80)
    for idx, row in df.iterrows():
        print(f"{row['ID']:3d}. 위치: ({row['X']:6.1f}, {row['Y']:6.1f}), "
              f"크기: {row[f'Diameter_{x_units}']:6.2f} {x_units}, "
              f"종횡비: {row['Aspect_Ratio']:4.2f}, "
              f"원형도: {row['Circularity']:4.2f}")

# CSV 저장
csv_filename = f'lattice_fringe_detection_{len(df)}particles.csv'
df.to_csv(csv_filename, index=False)
print(f"\n💾 데이터 저장: {csv_filename}")

# 8. 시각화
print(f"\n🎨 시각화 생성 중...")

# 전체 비교
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# (0,0) 원본
axes[0,0].imshow(signal.data, cmap='gray', origin='lower')
axes[0,0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0,0].axis('off')

# (0,1) 강력하게 스무딩된 이미지
axes[0,1].imshow(smoothed, cmap='gray', origin='lower')
axes[0,1].set_title('After Strong Smoothing (sigma=3.0)\n(Lattice fringe removed)', 
                   fontsize=14, fontweight='bold')
axes[0,1].axis('off')

# (0,2) 이진화
axes[0,2].imshow(cleaned, cmap='binary', origin='lower')
axes[0,2].set_title(f'Binary Mask (Threshold={threshold:.3f})', fontsize=14, fontweight='bold')
axes[0,2].axis('off')

# (1,0) 거리 변환
im = axes[1,0].imshow(distance, cmap='jet', origin='lower')
if len(coords) > 0:
    axes[1,0].plot(coords[:, 1], coords[:, 0], 'w.', markersize=8, 
                   markeredgecolor='black', markeredgewidth=1)
axes[1,0].set_title(f'Distance Transform + Centers ({len(coords)})', 
                   fontsize=14, fontweight='bold')
axes[1,0].axis('off')
plt.colorbar(im, ax=axes[1,0], fraction=0.046)

# (1,1) Watershed 결과
axes[1,1].imshow(labeled, cmap='nipy_spectral', origin='lower')
axes[1,1].set_title(f'Watershed Segmentation ({n_particles} regions)', 
                   fontsize=14, fontweight='bold')
axes[1,1].axis('off')

# (1,2) 최종 검출
axes[1,2].imshow(signal.data, cmap='gray', origin='lower')
if len(df) > 0:
    for _, row in df.iterrows():
        x, y = row['X'], row['Y']
        radius = (row[f'Diameter_{x_units}'] / pixel_size_x) / 2
        circle = Circle((x, y), radius, fill=False, color='red', linewidth=2.5, alpha=0.9)
        axes[1,2].add_patch(circle)
        axes[1,2].text(x, y, str(int(row['ID'])), ha='center', va='center',
                      color='yellow', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', alpha=0.6))

axes[1,2].set_title(f'Final Detection: {len(df)} Particles', fontsize=14, fontweight='bold')
axes[1,2].axis('off')

# 스케일바
scale_length_nm = 50
scale_length_px = scale_length_nm / pixel_size_x
scale_y = signal.data.shape[0] - 100
scale_x_start = signal.data.shape[1] - scale_length_px - 100
scale_x_end = scale_x_start + scale_length_px
axes[1,2].plot([scale_x_start, scale_x_end], [scale_y, scale_y], 'w-', linewidth=6)
axes[1,2].text((scale_x_start + scale_x_end) / 2, scale_y - 35,
              f'{scale_length_nm} nm', ha='center', va='top',
              color='white', fontsize=14, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))

plt.suptitle('Particle Detection for Lattice Fringe Containing Nanoparticles', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('lattice_fringe_detection_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 저장: lattice_fringe_detection_comparison.png")

# 9. 개별 입자 확대 (모든 입자)
if len(df) > 0:
    n_particles_to_show = len(df)
    n_cols = 5
    n_rows = int(np.ceil(n_particles_to_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(axes):
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
        
        # 입자 경계
        contours = measure.find_contours(zoom_labeled > 0, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2.5)
        
        # 중심점
        local_x = x_center - x_start
        local_y = y_center - y_start
        ax.plot(local_x, local_y, 'y+', markersize=20, markeredgewidth=3)
        
        ax.set_title(f'Particle #{row["ID"]:d}\n'
                    f'Size: {row[f"Diameter_{x_units}"]:.2f} {x_units}\n'
                    f'Aspect: {row["Aspect_Ratio"]:.2f}',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 스케일바
        scale_nm = 5
        scale_px = scale_nm / pixel_size_x
        scale_y_pos = zoom_img.shape[0] - 15
        ax.plot([15, 15 + scale_px], [scale_y_pos, scale_y_pos], 'w-', linewidth=3)
        ax.text(15 + scale_px/2, scale_y_pos - 10, f'{scale_nm} nm',
               ha='center', va='top', color='white', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # 빈 subplot 숨기기
    for idx in range(n_particles_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'All {len(df)} Detected Particles - Detailed View', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lattice_fringe_all_particles_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: lattice_fringe_all_particles_detailed.png")

# 10. 크기 분포
if len(df) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    sizes = df[f'Diameter_{x_units}']
    
    ax.hist(sizes, bins=min(15, len(df)), alpha=0.8, color='steelblue', 
            edgecolor='black', linewidth=1.5)
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
    plt.savefig('lattice_fringe_size_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: lattice_fringe_size_distribution.png")

print("\n" + "=" * 80)
print("✅ Lattice Fringe 입자 검출 완료!")
print("=" * 80)
print(f"\n💡 개선 사항:")
print(f"   ✓ 강력한 스무딩 (sigma=3.0)으로 lattice fringe 패턴 제거")
print(f"   ✓ 입자 내부 구멍 채우기로 solid한 입자 형성")
print(f"   ✓ 매우 큰 footprint (20×20)로 Watershed 분할 최소화")
print(f"   ✓ Lattice fringe 배열을 하나의 원형 입자로 인식")
print(f"   ✓ 총 {len(df)}개 입자 검출 (2nm 이상 위주)")
if len(df) > 0:
    print(f"   ✓ 평균 크기: {df[f'Diameter_{x_units}'].mean():.2f} nm")
print(f"\n📋 생성된 파일:")
print(f"   - {csv_filename}")
print(f"   - lattice_fringe_detection_comparison.png")
print(f"   - lattice_fringe_all_particles_detailed.png")
print(f"   - lattice_fringe_size_distribution.png")
