#!/usr/bin/env python
"""115kx 파일의 입자 검출 문제 디버깅"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage
from matplotlib.patches import Circle

print("🔍 115kx 파일 입자 검출 문제 디버깅\n")

# 파일 로드
filename = "Camera Ceta 1111 115 kx 0002.emd"
signal = hs.load(filename)
if isinstance(signal, list):
    signal = signal[0]

data = signal.data.astype(float)
pixel_size_nm = 0.11377

print(f"📊 이미지 정보:")
print(f"   크기: {data.shape[0]} × {data.shape[1]} 픽셀")
print(f"   실제 크기: {data.shape[0] * pixel_size_nm:.1f} × {data.shape[1] * pixel_size_nm:.1f} nm")
print(f"   픽셀 크기: {pixel_size_nm:.4f} nm/pixel")
print(f"   데이터 범위: {data.min():.1f} ~ {data.max():.1f}")
print(f"   데이터 평균: {data.mean():.1f}\n")

print(f"💡 예상 2nm 입자:")
print(f"   지름: {2.0 / pixel_size_nm:.1f} 픽셀")
print(f"   면적: {np.pi * (1.0 / pixel_size_nm) ** 2:.0f} 픽셀²\n")

# 전처리
vmin, vmax = np.percentile(data, [2, 98])
data_normalized = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)

# 가우시안 블러 적용
blurred = ndimage.gaussian_filter(data_normalized, sigma=1.0)

# 이미지 반전 (어두운 입자)
inverted = 1.0 - blurred

# 여러 임계값 + 최소 크기 조합 테스트
thresholds = [0.65, 0.7, 0.75, 0.8]
min_areas = [30, 50, 100, 150]  # 픽셀²

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('임계값 & 최소 크기 조합 테스트 (어두운 입자 검출)', fontsize=16, fontweight='bold')

best_result = None
best_mean = 0

for i, thresh in enumerate(thresholds):
    for j, min_area in enumerate(min_areas):
        ax = axes[i, j]
        
        # 이진화
        binary = inverted > thresh
        
        # 노이즈 제거
        cleaned = morphology.remove_small_objects(binary, min_size=min_area)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_area//2)
        
        # 라벨링
        labels = measure.label(cleaned)
        props = measure.regionprops(labels)
        
        # 크기 계산
        sizes = []
        for prop in props:
            area_nm2 = prop.area * (pixel_size_nm ** 2)
            equiv_diameter = 2 * np.sqrt(area_nm2 / np.pi)
            if 0.5 <= equiv_diameter <= 5.0:
                sizes.append(equiv_diameter)
        
        mean_size = np.mean(sizes) if sizes else 0
        
        # 최적 결과 추적 (1.5~2.5nm 평균을 목표)
        if 1.5 <= mean_size <= 2.5:
            if abs(mean_size - 2.0) < abs(best_mean - 2.0):
                best_mean = mean_size
                best_result = {
                    'threshold': thresh,
                    'min_area': min_area,
                    'n_particles': len(sizes),
                    'mean': mean_size,
                    'median': np.median(sizes) if sizes else 0
                }
        
        # 시각화
        ax.imshow(data_normalized, cmap='gray', origin='lower')
        
        # 입자 윤곽선
        for prop in props[:50]:  # 최대 50개만 표시
            y, x = prop.centroid
            r = np.sqrt(prop.area / np.pi)
            circle = Circle((x, y), r, fill=False, color='red', linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
        
        title = f'Th:{thresh:.2f} Min:{min_area}px²\n'
        title += f'N={len(sizes)}, Avg={mean_size:.2f}nm'
        
        # 색상 코딩
        if 1.5 <= mean_size <= 2.5:
            ax.set_title(title, fontsize=10, fontweight='bold', color='green')
        elif mean_size < 1.0 or mean_size > 3.0:
            ax.set_title(title, fontsize=10, color='red')
        else:
            ax.set_title(title, fontsize=10, color='orange')
        
        ax.axis('off')

plt.tight_layout()
plt.savefig('debug_threshold_minsize_grid.png', dpi=150, bbox_inches='tight')
print("✅ 그리드 비교 이미지 저장: debug_threshold_minsize_grid.png\n")

# 최적 결과 표시
if best_result:
    print("="*70)
    print("🎯 최적 설정 발견!")
    print("="*70)
    print(f"   임계값: {best_result['threshold']}")
    print(f"   최소 면적: {best_result['min_area']} 픽셀²")
    print(f"   입자 수: {best_result['n_particles']}")
    print(f"   평균 크기: {best_result['mean']:.3f} nm")
    print(f"   중앙값: {best_result['median']:.3f} nm")
    print("="*70)
    
    # 최적 설정으로 상세 시각화
    binary = inverted > best_result['threshold']
    cleaned = morphology.remove_small_objects(binary, min_size=best_result['min_area'])
    labels = measure.label(cleaned)
    props = measure.regionprops(labels)
    
    sizes = []
    for prop in props:
        area_nm2 = prop.area * (pixel_size_nm ** 2)
        equiv_diameter = 2 * np.sqrt(area_nm2 / np.pi)
        if 0.5 <= equiv_diameter <= 5.0:
            sizes.append(equiv_diameter)
    
    # 상세 시각화
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'최적 설정 상세 분석 (Th={best_result["threshold"]}, Min={best_result["min_area"]}px²)', 
                  fontsize=14, fontweight='bold')
    
    # 1. 원본
    axes2[0].imshow(data_normalized, cmap='gray', origin='lower')
    axes2[0].set_title('Original (Normalized)', fontsize=12)
    axes2[0].axis('off')
    
    # 2. 이진 마스크
    axes2[1].imshow(cleaned, cmap='binary', origin='lower')
    axes2[1].set_title(f'Binary Mask\n({len(props)} regions)', fontsize=12)
    axes2[1].axis('off')
    
    # 3. 검출 결과
    axes2[2].imshow(data_normalized, cmap='gray', origin='lower')
    for prop in props:
        y, x = prop.centroid
        r = np.sqrt(prop.area / np.pi)
        circle = Circle((x, y), r, fill=False, color='red', linewidth=1.5, alpha=0.8)
        axes2[2].add_patch(circle)
    axes2[2].set_title(f'Detected Particles\nMean={np.mean(sizes):.2f}nm', fontsize=12)
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_best_result.png', dpi=150, bbox_inches='tight')
    print("\n✅ 최적 결과 이미지 저장: debug_best_result.png")
    
    # 히스토그램
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(sizes, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.2f} nm')
    ax3.axvline(np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.2f} nm')
    ax3.set_xlabel('Particle Size (nm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title(f'Size Distribution (N={len(sizes)})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('debug_best_histogram.png', dpi=150, bbox_inches='tight')
    print("✅ 히스토그램 저장: debug_best_histogram.png\n")
    
else:
    print("\n⚠️  1.5~2.5nm 범위의 적절한 결과를 찾지 못했습니다.")
    print("   → 이미지에 2nm 크기의 입자가 없거나")
    print("   → 더 넓은 파라미터 범위를 테스트해야 할 수 있습니다.")
