#!/usr/bin/env python
"""115kx 배율 이미지에 개선된 자동 설정 테스트"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure, segmentation
from scipy import ndimage
from skimage.feature import peak_local_max
from matplotlib.patches import Circle

filename = "Camera Ceta 1111 115 kx 0002.emd"
print(f"🔍 {filename} 개선된 설정으로 테스트\n")

signal = hs.load(filename)
if isinstance(signal, list):
    signal = signal[0]

pixel_size = 0.11377  # nm/pixel
data = signal.data.astype(float)

# 전처리
vmin, vmax = np.percentile(data, [2, 98])
processed = np.clip(data, vmin, vmax)
processed = (processed - vmin) / (vmax - vmin)

# 반전
inverted = 1.0 - processed

# 🆕 배율 기반 설정 - 면적 기준으로 수정!
target_min_size_nm = 1.5  # 최소 지름
target_min_area_nm2 = np.pi * (target_min_size_nm / 2) ** 2  # 면적
min_particle_size = int(np.ceil(target_min_area_nm2 / (pixel_size ** 2)))  # 면적을 픽셀 수로 변환
min_particle_size = max(10, min_particle_size)  # 최소 10픽셀

target_footprint_nm = 3.5
watershed_footprint = int(np.round(target_footprint_nm / pixel_size))
watershed_footprint = max(3, min(15, watershed_footprint))

print(f"📐 자동 계산된 설정:")
print(f"   픽셀 크기: {pixel_size} nm/pixel")
print(f"   목표 최소 지름: {target_min_size_nm} nm")
print(f"   목표 최소 면적: {target_min_area_nm2:.2f} nm²")
print(f"   최소 입자 면적: {min_particle_size} 픽셀² ({min_particle_size * (pixel_size**2):.2f} nm²)")
print(f"   → 지름 환산: {2 * np.sqrt(min_particle_size * (pixel_size**2) / np.pi):.2f} nm")
print(f"   Watershed footprint: {watershed_footprint} 픽셀 ({watershed_footprint * pixel_size:.2f} nm)\n")

# 임계값 테스트 (더 높은 값들)
thresholds_to_test = [0.8, 0.85, 0.9]

print("\n" + "="*70)
print("🔬 임계값 비교 테스트")
print("="*70)

best_threshold = 0.8
best_mean = 0

for threshold in thresholds_to_test:
    binary = inverted > threshold
    binary_cleaned = morphology.remove_small_objects(binary, min_size=min_particle_size)
    binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_particle_size//2)
    
    distance = ndimage.distance_transform_edt(binary_cleaned)
    coords = peak_local_max(distance, footprint=np.ones((watershed_footprint, watershed_footprint)), labels=binary_cleaned)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labeled = segmentation.watershed(-distance, markers, mask=binary_cleaned)
    
    props = measure.regionprops(labeled)
    sizes = [np.sqrt(4 * p.area / np.pi) * pixel_size for p in props]
    
    mean_size = np.mean(sizes) if len(sizes) > 0 else 0
    print(f"\nThreshold {threshold}:")
    print(f"  입자 수: {len(sizes)}")
    print(f"  평균 크기: {mean_size:.3f} nm")
    print(f"  중앙값: {np.median(sizes):.3f} nm" if len(sizes) > 0 else "  중앙값: N/A")
    print(f"  크기 범위: {np.min(sizes):.3f} ~ {np.max(sizes):.3f} nm" if len(sizes) > 0 else "  크기 범위: N/A")
    
    # 2nm 근처 평균을 가진 임계값 선택
    if 1.5 < mean_size < 3.0:
        if abs(mean_size - 2.0) < abs(best_mean - 2.0):
            best_threshold = threshold
            best_mean = mean_size

print(f"\n✅ 최적 임계값: {best_threshold} (평균 크기: {best_mean:.3f} nm)")

# 최적 임계값으로 다시 검출
threshold = best_threshold
binary = inverted > threshold
binary_cleaned = morphology.remove_small_objects(binary, min_size=min_particle_size)
binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_particle_size//2)

# Watershed 적용
distance = ndimage.distance_transform_edt(binary_cleaned)
coords = peak_local_max(distance, footprint=np.ones((watershed_footprint, watershed_footprint)), labels=binary_cleaned)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)
labeled = segmentation.watershed(-distance, markers, mask=binary_cleaned)
n_particles = labeled.max()

# 측정
props = measure.regionprops(labeled)
sizes = [np.sqrt(4 * p.area / np.pi) * pixel_size for p in props]

print(f"✅ 검출 결과:")
print(f"   입자 수: {n_particles}")
print(f"   평균 크기: {np.mean(sizes):.3f} nm")
print(f"   중앙값: {np.median(sizes):.3f} nm")
print(f"   크기 범위: {np.min(sizes):.3f} ~ {np.max(sizes):.3f} nm")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'115kx Improved Detection (N={n_particles})', fontsize=14, fontweight='bold')

axes[0].imshow(processed, cmap='gray', origin='lower')
axes[0].set_title('Original (Enhanced)')
axes[0].axis('off')

axes[1].imshow(binary_cleaned, cmap='binary', origin='lower')
axes[1].set_title(f'Binary Mask\n(min={min_particle_size}px={min_particle_size*pixel_size:.2f}nm)')
axes[1].axis('off')

# 입자 표시 (샘플링)
axes[2].imshow(processed, cmap='gray', origin='lower')
n_show = min(100, len(props))
import random
sampled_props = random.sample(props, n_show) if len(props) > n_show else props
for prop in sampled_props:
    y, x = prop.centroid
    r = np.sqrt(prop.area / np.pi)
    circle = Circle((x, y), r, fill=False, color='red', linewidth=1)
    axes[2].add_patch(circle)
axes[2].set_title(f'Detected Particles (showing {n_show})')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('improved_115kx_detection.png', dpi=150, bbox_inches='tight')
print("\n✅ 결과 이미지: improved_115kx_detection.png")

# 히스토그램
fig2, ax = plt.subplots(figsize=(10, 6))
counts, bins, patches = ax.hist(sizes, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.2f} nm')
ax.axvline(np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.2f} nm')
ax.set_xlabel('Particle Size (nm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Particle Size Distribution (115kx, N={n_particles})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('improved_115kx_histogram.png', dpi=150, bbox_inches='tight')
print("✅ 히스토그램: improved_115kx_histogram.png")
