#!/usr/bin/env python
"""Watershed 및 개선된 크기 측정 방법 테스트"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology, segmentation, filters
from scipy import ndimage
from skimage.feature import peak_local_max

print("🧪 Watershed 알고리즘 테스트 시작\n")

# Load data
signal = hs.load("Camera Ceta 1042 27000 x 0001.emd")
if isinstance(signal, list):
    signal = signal[0]

pixel_size = 0.5  # nm/pixel
data = signal.data.astype(float)

# Preprocessing
vmin, vmax = np.percentile(data, [2, 98])
processed = np.clip(data, vmin, vmax)
processed = (processed - vmin) / (vmax - vmin)

# Invert for dark particles
inverted = 1.0 - processed

# Threshold
threshold = 0.8
binary_mask = inverted > threshold

# Clean up
binary_cleaned = morphology.remove_small_objects(binary_mask, min_size=10)
binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=5)

# WITHOUT Watershed
labeled_normal = measure.label(binary_cleaned)
n_normal = labeled_normal.max()
print(f"❌ Watershed 없이: {n_normal}개 입자 검출")

# Manual reference
manual_ref = 2.09  # Manual measurement reference

# WITH Watershed - 민감도 조정 테스트
print("\n🌊 Watershed 분할 적용 (다양한 민감도 테스트)...")

for footprint_size in [5, 7, 9]:
    distance = ndimage.distance_transform_edt(binary_cleaned)
    coords = peak_local_max(distance, footprint=np.ones((footprint_size, footprint_size)), labels=binary_cleaned)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labeled_watershed = segmentation.watershed(-distance, markers, mask=binary_cleaned)
    n_watershed = labeled_watershed.max()
    
    # Measure
    props = measure.regionprops(labeled_watershed)
    sizes_major = [prop.major_axis_length * pixel_size for prop in props 
                   if hasattr(prop, 'major_axis_length')]
    mean_major = np.mean(sizes_major)
    diff = mean_major - manual_ref
    diff_percent = (diff / manual_ref) * 100
    
    print(f"\n민감도 {footprint_size}x{footprint_size}:")
    print(f"  검출 입자: {n_watershed}개")
    print(f"  Major Axis 평균: {mean_major:.3f} nm")
    print(f"  수동 측정 대비: {diff:+.3f} nm ({diff_percent:+.1f}%)")

# Use best footprint (7)
print("\n" + "="*70)
print("📊 최적 설정 (footprint=7) 상세 결과")
print("="*70)

distance = ndimage.distance_transform_edt(binary_cleaned)
coords = peak_local_max(distance, footprint=np.ones((7, 7)), labels=binary_cleaned)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)
labeled_watershed = segmentation.watershed(-distance, markers, mask=binary_cleaned)
n_watershed = labeled_watershed.max()

print(f"\n✅ Watershed 후: {n_watershed}개 입자 검출")
print(f"🎯 응집체에서 {n_watershed - n_normal}개 추가 분리됨\n")

# Measure particles (WITH watershed)
props = measure.regionprops(labeled_watershed)

sizes_equiv = []
sizes_feret = []
sizes_major = []

for prop in props:
    # Equivalent diameter
    equiv_diameter = np.sqrt(4 * prop.area / np.pi) * pixel_size
    sizes_equiv.append(equiv_diameter)
    
    # Feret min (use minor_axis as approximation)
    feret_min = prop.minor_axis_length * pixel_size if hasattr(prop, 'minor_axis_length') else equiv_diameter
    sizes_feret.append(feret_min)
    
    # Major axis
    major_axis = prop.major_axis_length * pixel_size if hasattr(prop, 'major_axis_length') else equiv_diameter
    sizes_major.append(major_axis)

# Statistics
print("="*70)
print("📊 측정 결과 비교 (수동 측정 참조값: 2.09 nm)")
print("="*70)

methods = [
    ("Equivalent Diameter", sizes_equiv),
    ("Feret Min Diameter", sizes_feret),
    ("Major Axis Length", sizes_major)
]

for method_name, sizes in methods:
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    diff = mean_size - manual_ref
    diff_percent = (diff / manual_ref) * 100
    
    print(f"\n{method_name}:")
    print(f"  평균: {mean_size:.3f} nm")
    print(f"  중앙값: {median_size:.3f} nm")
    print(f"  범위: {np.min(sizes):.3f} ~ {np.max(sizes):.3f} nm")
    print(f"  수동 측정 대비: {diff:+.3f} nm ({diff_percent:+.1f}%)")
    
    # Color code: green if close, yellow if moderate, red if far
    if abs(diff_percent) < 10:
        status = "✅ 매우 근접"
    elif abs(diff_percent) < 20:
        status = "⚠️  약간 차이"
    else:
        status = "❌ 큰 차이"
    print(f"  평가: {status}")

print("\n" + "="*70)
print("💡 권장 사항:")
print("="*70)
best_method = min(methods, key=lambda x: abs(np.mean(x[1]) - manual_ref))
print(f"✅ 수동 측정에 가장 가까운 방법: {best_method[0]}")
print(f"   평균 차이: {abs(np.mean(best_method[1]) - manual_ref):.3f} nm")
print("\n🎯 Watershed 분할 + " + best_method[0] + " 사용을 권장합니다!")
