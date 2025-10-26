#!/usr/bin/env python3
"""
27개 샘플 입자만 분석
- particle_samples.csv에서 27개 샘플 좌표 로드
- 각 샘플의 크기, 원형도, 강도 등을 측정
- 통계 분석 및 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening
from scipy.optimize import curve_fit
from skimage import exposure, measure, morphology
from skimage.morphology import disk

print("=" * 70)
print("🎯 27개 샘플 입자 분석")
print("=" * 70)

# EMD 파일 로드
emd_file = "Camera Ceta 1204 380 kx.emd"
print(f"\n📂 파일 로딩: {emd_file}")
s = hs.load(emd_file)
data = s.data.astype(float)

# 픽셀 크기
pixel_size = s.axes_manager[0].scale
print(f"📐 픽셀 크기: {pixel_size:.4f} nm/pixel")
print(f"📊 이미지 크기: {data.shape}")

# 샘플 좌표 로드
samples_file = "particle_samples.csv"
print(f"\n📂 샘플 로딩: {samples_file}")
samples = pd.read_csv(samples_file)
print(f"✅ 로드된 샘플: {len(samples)}개")

# 이미지 전처리
print("\n🎨 전처리 중...")
smoothed = gaussian_filter(data, sigma=2.0)
normalized = (smoothed - np.percentile(smoothed, 1)) / (np.percentile(smoothed, 99) - np.percentile(smoothed, 1))
normalized = np.clip(normalized, 0, 1)

# CLAHE 적용
smoothed_uint8 = (normalized * 255).astype(np.uint8)
enhanced = exposure.equalize_adapthist(smoothed_uint8, clip_limit=0.01, kernel_size=128)
normalized = enhanced.astype(float)
print("✅ 전처리 완료")

# 임계값 계산
sample_mean = samples['center_intensity'].mean()
global_mean = np.mean(normalized)
threshold_dark = (sample_mean + global_mean) / 2

print(f"\n📊 임계값 계산:")
print(f"  샘플 평균: {sample_mean:.4f}")
print(f"  전체 평균: {global_mean:.4f}")
print(f"  임계값: {threshold_dark:.4f}")

# 이진화 및 라벨링
binary = normalized < threshold_dark
binary = binary_fill_holes(binary)
binary = binary_opening(binary, disk(2))
labeled = measure.label(binary, connectivity=2)

print(f"\n🔍 검출된 어두운 영역: {labeled.max()}개")

# 각 샘플에 대해 분석
results = []
window_size = 50  # 각 샘플 주변 분석 영역

print("\n📊 샘플별 분석 중...")
for idx, row in samples.iterrows():
    x, y = int(row['x']), int(row['y'])
    
    # 샘플 좌표에서 가장 가까운 검출된 입자 찾기
    label_at_pos = labeled[y, x]
    
    # 해당 위치에 검출된 입자가 없으면 주변 탐색
    if label_at_pos == 0:
        # 10픽셀 반경 내에서 검출된 입자 찾기
        search_radius = 10
        y_min = max(0, y - search_radius)
        y_max = min(labeled.shape[0], y + search_radius + 1)
        x_min = max(0, x - search_radius)
        x_max = min(labeled.shape[1], x + search_radius + 1)
        
        region = labeled[y_min:y_max, x_min:x_max]
        unique_labels = np.unique(region)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) > 0:
            # 가장 가까운 레이블 찾기
            min_dist = float('inf')
            best_label = 0
            for lbl in unique_labels:
                mask_temp = (labeled == lbl).astype(np.uint8)
                props = measure.regionprops(mask_temp)
                if props:
                    cy, cx = props[0].centroid
                    dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = lbl
            label_at_pos = best_label
    
    if label_at_pos > 0:
        # 입자 속성 계산
        mask = (labeled == label_at_pos).astype(np.uint8)
        props = measure.regionprops(mask, intensity_image=normalized)[0]
        
        # 면적 및 직경 (nm 단위)
        area_pixels = props.area
        area_nm2 = area_pixels * (pixel_size ** 2)
        equiv_diameter_nm = props.equivalent_diameter * pixel_size
        
        # 원형도 계산
        perimeter = props.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter ** 2)
        else:
            circularity = 0
        
        # 강도 정보
        mean_intensity = props.mean_intensity
        min_intensity = props.min_intensity
        max_intensity = props.max_intensity
        
        # 중심 좌표
        centroid_y, centroid_x = props.centroid
        
        results.append({
            'sample_id': idx + 1,
            'original_x': x,
            'original_y': y,
            'detected': True,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'distance_px': np.sqrt((centroid_x - x)**2 + (centroid_y - y)**2),
            'area_nm2': area_nm2,
            'equiv_diameter_nm': equiv_diameter_nm,
            'circularity': circularity,
            'mean_intensity': mean_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity
        })
        
        status = "✅"
    else:
        results.append({
            'sample_id': idx + 1,
            'original_x': x,
            'original_y': y,
            'detected': False,
            'centroid_x': np.nan,
            'centroid_y': np.nan,
            'distance_px': np.nan,
            'area_nm2': np.nan,
            'equiv_diameter_nm': np.nan,
            'circularity': np.nan,
            'mean_intensity': np.nan,
            'min_intensity': np.nan,
            'max_intensity': np.nan
        })
        status = "❌"
    
    print(f"  샘플 {idx+1:2d} ({x:4d}, {y:4d}): {status}")

# 결과를 데이터프레임으로 변환
df_results = pd.DataFrame(results)

# 통계 계산 (검출된 샘플만)
detected = df_results[df_results['detected'] == True]
num_detected = len(detected)
num_total = len(df_results)

print(f"\n📊 검출 결과:")
print(f"  검출된 샘플: {num_detected}/{num_total} ({100*num_detected/num_total:.1f}%)")
print(f"  미검출 샘플: {num_total - num_detected}개")

if num_detected > 0:
    print(f"\n📏 크기 분석 (검출된 {num_detected}개 샘플):")
    print(f"  평균 면적: {detected['area_nm2'].mean():.4f} ± {detected['area_nm2'].std():.4f} nm²")
    print(f"  평균 직경: {detected['equiv_diameter_nm'].mean():.4f} ± {detected['equiv_diameter_nm'].std():.4f} nm")
    print(f"  직경 범위: {detected['equiv_diameter_nm'].min():.4f} ~ {detected['equiv_diameter_nm'].max():.4f} nm")
    print(f"  평균 원형도: {detected['circularity'].mean():.3f} ± {detected['circularity'].std():.3f}")
    print(f"  평균 중심 거리: {detected['distance_px'].mean():.2f} ± {detected['distance_px'].std():.2f} 픽셀")

# 결과 저장
output_csv = "sample_particles_analysis.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n💾 결과 저장: {output_csv}")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 원본 이미지 + 샘플 위치
ax = axes[0, 0]
ax.imshow(normalized, cmap='gray', origin='lower')
for _, row in df_results.iterrows():
    if row['detected']:
        ax.plot(row['original_x'], row['original_y'], 'go', markersize=8, markerfacecolor='none', markeredgewidth=2)
        ax.plot(row['centroid_x'], row['centroid_y'], 'g+', markersize=10, markeredgewidth=2)
    else:
        ax.plot(row['original_x'], row['original_y'], 'rx', markersize=10, markeredgewidth=2)
ax.set_title(f'Sample Locations\n(Green O: Detected, Red X: Not Detected)', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

# 2. 강도 히스토그램
ax = axes[0, 1]
ax.hist(normalized.ravel(), bins=100, alpha=0.7, color='gray', label='Full Image')
ax.axvline(threshold_dark, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold_dark:.3f}')
ax.axvline(sample_mean, color='g', linestyle='--', linewidth=2, label=f'Sample Mean: {sample_mean:.3f}')
ax.set_xlabel('Normalized Intensity')
ax.set_ylabel('Frequency')
ax.set_title('Intensity Distribution', fontsize=12)
ax.legend()
ax.set_yscale('log')

# 3. 크기 분포 with Gaussian fitting
if num_detected > 0:
    ax = axes[0, 2]
    
    # 히스토그램 그리기
    counts, bins, patches = ax.hist(detected['equiv_diameter_nm'], bins=20, alpha=0.7, 
                                     color='blue', edgecolor='black', label='Data')
    
    # Gaussian fitting
    def gaussian(x, amp, mean, std):
        return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    # 히스토그램 중심값
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 초기 추정값
    mean_est = detected['equiv_diameter_nm'].mean()
    std_est = detected['equiv_diameter_nm'].std()
    amp_est = counts.max()
    
    try:
        # Curve fitting
        popt, pcov = curve_fit(gaussian, bin_centers, counts, 
                              p0=[amp_est, mean_est, std_est],
                              maxfev=10000)
        
        # Fitting 결과 그리기
        x_fit = np.linspace(bins[0], bins[-1], 200)
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2.5, 
               label=f'Gaussian Fit\nmean={popt[1]:.3f} nm\nstd={abs(popt[2]):.3f} nm')
        
        print(f"\n📈 Gaussian Fitting 결과:")
        print(f"  mean: {popt[1]:.4f} nm")
        print(f"  std: {abs(popt[2]):.4f} nm")
        print(f"  FWHM: {2.355 * abs(popt[2]):.4f} nm")
        
    except Exception as e:
        print(f"\n⚠️ Gaussian fitting 실패: {e}")
        ax.axvline(mean_est, color='r', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_est:.3f} nm')
    
    ax.set_xlabel('Equivalent Diameter (nm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Size Distribution (n={num_detected})', fontsize=12)
    ax.legend()
else:
    axes[0, 2].text(0.5, 0.5, 'No particles detected', ha='center', va='center', transform=axes[0, 2].transAxes)

# 4. 원형도 분포
if num_detected > 0:
    ax = axes[1, 0]
    ax.hist(detected['circularity'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(detected['circularity'].mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {detected["circularity"].mean():.3f}')
    ax.set_xlabel('Circularity')
    ax.set_ylabel('Count')
    ax.set_title(f'Circularity Distribution (n={num_detected})', fontsize=12)
    ax.legend()
else:
    axes[1, 0].text(0.5, 0.5, 'No particles detected', ha='center', va='center', transform=axes[1, 0].transAxes)

# 5. 크기 vs 원형도
if num_detected > 0:
    ax = axes[1, 1]
    scatter = ax.scatter(detected['equiv_diameter_nm'], detected['circularity'], 
                        c=detected['mean_intensity'], cmap='viridis', s=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Equivalent Diameter (nm)')
    ax.set_ylabel('Circularity')
    ax.set_title('Size vs Circularity', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Mean Intensity')
else:
    axes[1, 1].text(0.5, 0.5, 'No particles detected', ha='center', va='center', transform=axes[1, 1].transAxes)

# 6. 중심 거리 분포
if num_detected > 0:
    ax = axes[1, 2]
    ax.hist(detected['distance_px'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(detected['distance_px'].mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {detected["distance_px"].mean():.2f} px')
    ax.set_xlabel('Distance from Original Position (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(f'Centroid Distance (n={num_detected})', fontsize=12)
    ax.legend()
else:
    axes[1, 2].text(0.5, 0.5, 'No particles detected', ha='center', va='center', transform=axes[1, 2].transAxes)

plt.tight_layout()

# 저장
output_image = "sample_particles_analysis.png"
fig.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"💾 시각화 저장: {output_image}")

print("\n" + "=" * 70)
print("✅ 분석 완료!")
print("=" * 70)

plt.show()
