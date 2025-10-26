"""
밝기 변화(그라디언트) 기반 나노입자 검출
- 입자의 음영 패턴(구형 특성)을 이용
- Lattice fringe보다는 전체적인 밝기 분포에 집중
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, feature, exposure
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
print("🔬 밝기 변화 기반 나노입자 검출 (음영 패턴 분석)")
print("=" * 80)
print(f"픽셀 크기: {pixel_size_x:.4f} {x_units}/픽셀")

# 1. 이미지 전처리
print("\n1️⃣ 이미지 전처리...")
# 정규화
vmin_init, vmax_init = np.percentile(signal.data, [2, 98])
normalized = np.clip(signal.data, vmin_init, vmax_init)
normalized = (normalized - vmin_init) / (vmax_init - vmin_init)

# 적절한 스무딩으로 노이즈 제거 (lattice fringe는 어느 정도 유지)
smoothed = filters.gaussian(normalized, sigma=1.5)
print("   ✓ 가우시안 필터 적용 (sigma=1.5)")

# 2. 그라디언트 기반 접근
print("\n2️⃣ 그라디언트 분석 (밝기 변화 패턴)...")

# Sobel 필터로 그라디언트 크기 계산
grad_x = ndimage.sobel(smoothed, axis=1)
grad_y = ndimage.sobel(smoothed, axis=0)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
print("   ✓ 그라디언트 크기 계산 완료")

# 그라디언트가 강한 부분 = 입자 경계
# 구형 입자는 중심에서 가장자리로 갈수록 밝기가 변함
gradient_threshold = filters.threshold_otsu(gradient_magnitude)
edges = gradient_magnitude > gradient_threshold
print(f"   ✓ 그라디언트 임계값: {gradient_threshold:.4f}")

# 3. Laplacian of Gaussian (LoG) - 구형 blob 검출에 효과적
print("\n3️⃣ LoG (Laplacian of Gaussian) blob 검출...")
print("   ⚠️ 어두운 부분(입자)을 검출하기 위해 이미지 반전")

# 이미지 반전 - 어두운 부분(입자)을 밝게 만들어서 검출
inverted = 1.0 - smoothed

# 여러 sigma 값으로 blob 검출 (다양한 크기의 입자)
# 더 넓은 크기 범위 검출
min_sigma = 10  # 약 0.335nm * 10 = 3.35nm 직경
max_sigma = 50  # 약 0.335nm * 50 = 16.75nm 직경
num_sigma = 15  # 더 많은 sigma 단계

blobs_log = feature.blob_log(inverted, 
                             min_sigma=min_sigma, 
                             max_sigma=max_sigma,
                             num_sigma=num_sigma,
                             threshold=0.05)  # 더 높은 임계값으로 확실한 입자만

# blob 크기 계산 (반지름)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

print(f"   ✓ LoG blob 검출: {len(blobs_log)}개 후보")

# 4. 추가 필터링
print("\n4️⃣ blob 필터링...")

# 크기 기준 필터링
filtered_blobs = []
for blob in blobs_log:
    y, x, r = blob
    diameter_px = 2 * r
    diameter_nm = diameter_px * pixel_size_x
    
    # 크기 필터: 1.5nm ~ 20nm (더 넓은 범위)
    if 1.5 <= diameter_nm <= 20:
        filtered_blobs.append(blob)

print(f"   초기 blob: {len(blobs_log)}개")
print(f"   필터링 후: {len(filtered_blobs)}개")

# 5. 데이터 수집
particle_data = []
for i, blob in enumerate(filtered_blobs, 1):
    y, x, r = blob
    diameter_px = 2 * r
    diameter_nm = diameter_px * pixel_size_x
    area_nm2 = np.pi * (diameter_nm / 2) ** 2
    
    particle_data.append({
        'ID': i,
        'X': x,
        'Y': y,
        f'Diameter_{x_units}': diameter_nm,
        f'Radius_{x_units}': diameter_nm / 2,
        f'Area_{x_units}2': area_nm2,
        'Sigma': r / np.sqrt(2)  # 원래 sigma 값
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
    
    bins = [0, 1.5, 2, 2.5, 3, 4, 5, 10, 20, 100]
    bin_labels = ['<1.5', '1.5-2', '2-2.5', '2.5-3', '3-4', '4-5', '5-10', '10-20', '>20']
    df['Size_Category'] = pd.cut(df[f'Diameter_{x_units}'], bins=bins, labels=bin_labels)
    
    print(f"\n📈 크기별 분포:")
    size_dist = df['Size_Category'].value_counts().sort_index()
    for category, count in size_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {category:>6} nm: {count:3d}개 ({percentage:5.1f}%)")
    
    # 상위 20개
    print(f"\n🏆 가장 큰 입자 Top 20:")
    print("-" * 80)
    top20 = df.nlargest(min(20, len(df)), f'Diameter_{x_units}')
    for idx, row in top20.iterrows():
        print(f"{row['ID']:3d}. 위치: ({row['X']:6.1f}, {row['Y']:6.1f}), "
              f"크기: {row[f'Diameter_{x_units}']:6.2f} {x_units}")

# CSV 저장
csv_filename = f'gradient_based_detection_{len(df)}particles.csv'
df.to_csv(csv_filename, index=False)
print(f"\n💾 데이터 저장: {csv_filename}")

# 7. 시각화
print(f"\n🎨 시각화 생성 중...")

# 전체 비교
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# (0,0) 원본
axes[0,0].imshow(signal.data, cmap='gray', origin='lower')
axes[0,0].set_title('Original TEM Image', fontsize=14, fontweight='bold')
axes[0,0].axis('off')

# (0,1) 스무딩
axes[0,1].imshow(smoothed, cmap='gray', origin='lower')
axes[0,1].set_title('After Smoothing (sigma=1.5)', fontsize=14, fontweight='bold')
axes[0,1].axis('off')

# (0,2) 그라디언트 크기
im = axes[0,2].imshow(gradient_magnitude, cmap='hot', origin='lower')
axes[0,2].set_title('Gradient Magnitude (Edge Strength)', fontsize=14, fontweight='bold')
axes[0,2].axis('off')
plt.colorbar(im, ax=axes[0,2], fraction=0.046)

# (1,0) 그라디언트 임계값
axes[1,0].imshow(edges, cmap='binary', origin='lower')
axes[1,0].set_title(f'Gradient Edges (Threshold={gradient_threshold:.4f})', 
                   fontsize=14, fontweight='bold')
axes[1,0].axis('off')

# (1,1) 반전된 이미지 (어두운 입자가 밝게)
axes[1,1].imshow(inverted, cmap='gray', origin='lower')
axes[1,1].set_title('Inverted Image (Dark particles → Bright)', 
                   fontsize=14, fontweight='bold')
axes[1,1].axis('off')

# (1,2) 최종 검출 결과 (필터링 후)
axes[1,2].imshow(signal.data, cmap='gray', origin='lower')
if len(df) > 0:
    for _, row in df.iterrows():
        x, y = row['X'], row['Y']
        radius = row[f'Radius_{x_units}'] / pixel_size_x
        circle = Circle((x, y), radius, fill=False, color='red', linewidth=2, alpha=0.9)
        axes[1,2].add_patch(circle)
        if row['ID'] <= 30:
            axes[1,2].text(x, y, str(int(row['ID'])), ha='center', va='center',
                          color='yellow', fontsize=8, fontweight='bold',
                          bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.6))

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

plt.suptitle('Gradient-Based Particle Detection (Brightness Pattern Analysis)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('gradient_based_detection_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 저장: gradient_based_detection_comparison.png")

# 8. 개별 입자 확대 (상위 20개)
if len(df) > 0:
    n_show = min(20, len(df))
    top_particles = df.nlargest(n_show, f'Diameter_{x_units}')
    
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_particles.iterrows()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        x_center, y_center = int(row['X']), int(row['Y'])
        zoom_size = 100
        
        y_start = max(0, y_center - zoom_size)
        y_end = min(signal.data.shape[0], y_center + zoom_size)
        x_start = max(0, x_center - zoom_size)
        x_end = min(signal.data.shape[1], x_center + zoom_size)
        
        zoom_img = signal.data[y_start:y_end, x_start:x_end]
        
        zoom_vmin, zoom_vmax = np.percentile(zoom_img, [2, 98])
        ax.imshow(zoom_img, cmap='gray', origin='lower', vmin=zoom_vmin, vmax=zoom_vmax)
        
        # 입자 원 표시
        local_x = x_center - x_start
        local_y = y_center - y_start
        radius = row[f'Radius_{x_units}'] / pixel_size_x
        circle = Circle((local_x, local_y), radius, fill=False, color='red', linewidth=2.5)
        ax.add_patch(circle)
        
        # 중심점
        ax.plot(local_x, local_y, 'y+', markersize=15, markeredgewidth=2.5)
        
        ax.set_title(f'Particle #{row["ID"]:d}\n'
                    f'Size: {row[f"Diameter_{x_units}"]:.2f} {x_units}',
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 스케일바
        scale_nm = 5
        scale_px = scale_nm / pixel_size_x
        scale_y_pos = zoom_img.shape[0] - 12
        ax.plot([12, 12 + scale_px], [scale_y_pos, scale_y_pos], 'w-', linewidth=3)
        ax.text(12 + scale_px/2, scale_y_pos - 8, f'{scale_nm} nm',
               ha='center', va='top', color='white', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # 빈 subplot 숨기기
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {n_show} Particles - Detailed View', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_based_top_particles.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: gradient_based_top_particles.png")

# 9. 크기 분포 (가우시안 피팅 포함)
if len(df) > 0:
    from scipy.optimize import curve_fit
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sizes = df[f'Diameter_{x_units}']
    
    n_bins = min(20, max(10, len(df)//3))
    counts, bin_edges, patches = ax.hist(sizes, bins=n_bins, alpha=0.8, color='steelblue', 
                                         edgecolor='black', linewidth=1.5, label='Data')
    
    # 가우시안 피팅
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    def gaussian(x, amp, mean, std):
        return amp * np.exp(-(x - mean)**2 / (2 * std**2))
    
    # 초기 추정값
    amp_init = counts.max()
    mean_init = sizes.mean()
    std_init = sizes.std()
    
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, counts, 
                               p0=[amp_init, mean_init, std_init],
                               maxfev=10000)
        amp_fit, mean_fit, std_fit = popt
        
        # 피팅 곡선 그리기
        x_fit = np.linspace(sizes.min(), sizes.max(), 200)
        y_fit = gaussian(x_fit, amp_fit, mean_fit, std_fit)
        ax.plot(x_fit, y_fit, 'r-', linewidth=3, 
                label=f'Gaussian Fit\nμ={mean_fit:.2f} {x_units}, σ={std_fit:.2f} {x_units}')
        
        print(f"\n📈 가우시안 피팅 결과:")
        print(f"   평균 (μ): {mean_fit:.2f} {x_units}")
        print(f"   표준편차 (σ): {std_fit:.2f} {x_units}")
        print(f"   진폭: {amp_fit:.2f}")
    except Exception as e:
        print(f"\n⚠️ 가우시안 피팅 실패: {e}")
    
    ax.axvline(sizes.mean(), color='orange', linestyle='--', linewidth=2.5,
              label=f'Sample Mean: {sizes.mean():.2f} {x_units}')
    ax.axvline(sizes.median(), color='green', linestyle='--', linewidth=2.5,
              label=f'Median: {sizes.median():.2f} {x_units}')
    
    ax.set_xlabel(f'Particle Diameter ({x_units})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title(f'Particle Size Distribution with Gaussian Fit (n={len(df)})', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_based_size_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 저장: gradient_based_size_distribution.png")

# 10. 크기 카테고리별 입자 샘플 이미지
if len(df) > 0:
    print(f"\n🎨 크기 카테고리별 입자 샘플 이미지 생성 중...")
    
    # 각 카테고리에서 최대 5개씩 샘플링
    size_categories = df['Size_Category'].unique()
    size_categories = sorted([cat for cat in size_categories if pd.notna(cat)])
    
    # 카테고리 개수에 맞춰 서브플롯 구성
    n_categories = len(size_categories)
    if n_categories > 0:
        samples_per_category = 5
        
        fig, axes = plt.subplots(n_categories, samples_per_category, 
                                figsize=(5*samples_per_category, 5*n_categories))
        
        if n_categories == 1:
            axes = axes.reshape(1, -1)
        
        for cat_idx, category in enumerate(size_categories):
            # 해당 카테고리의 입자들
            cat_particles = df[df['Size_Category'] == category]
            
            # 최대 5개 샘플링 (크기 순으로 정렬)
            samples = cat_particles.nlargest(min(samples_per_category, len(cat_particles)), 
                                             f'Diameter_{x_units}')
            
            for sample_idx in range(samples_per_category):
                ax = axes[cat_idx, sample_idx]
                
                if sample_idx < len(samples):
                    row = samples.iloc[sample_idx]
                    
                    x_center, y_center = int(row['X']), int(row['Y'])
                    zoom_size = 80
                    
                    y_start = max(0, y_center - zoom_size)
                    y_end = min(signal.data.shape[0], y_center + zoom_size)
                    x_start = max(0, x_center - zoom_size)
                    x_end = min(signal.data.shape[1], x_center + zoom_size)
                    
                    zoom_img = signal.data[y_start:y_end, x_start:x_end]
                    
                    zoom_vmin, zoom_vmax = np.percentile(zoom_img, [2, 98])
                    ax.imshow(zoom_img, cmap='gray', origin='lower', vmin=zoom_vmin, vmax=zoom_vmax)
                    
                    # 입자 원 표시
                    local_x = x_center - x_start
                    local_y = y_center - y_start
                    radius = row[f'Radius_{x_units}'] / pixel_size_x
                    circle = Circle((local_x, local_y), radius, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)
                    
                    # 중심점
                    ax.plot(local_x, local_y, 'y+', markersize=12, markeredgewidth=2)
                    
                    ax.set_title(f'ID {row["ID"]:d}: {row[f"Diameter_{x_units}"]:.2f} {x_units}',
                                fontsize=10, fontweight='bold')
                    ax.axis('off')
                    
                    # 스케일바
                    scale_nm = 5
                    scale_px = scale_nm / pixel_size_x
                    scale_y_pos = zoom_img.shape[0] - 10
                    ax.plot([10, 10 + scale_px], [scale_y_pos, scale_y_pos], 'w-', linewidth=2.5)
                    ax.text(10 + scale_px/2, scale_y_pos - 6, f'{scale_nm} nm',
                           ha='center', va='top', color='white', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                else:
                    ax.axis('off')
            
            # 첫 번째 subplot에 카테고리 레이블 추가
            axes[cat_idx, 0].text(-0.15, 0.5, f'{category} nm', 
                                 transform=axes[cat_idx, 0].transAxes,
                                 fontsize=14, fontweight='bold', rotation=90,
                                 va='center', ha='right')
        
        plt.suptitle('Particle Samples by Size Category', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gradient_based_particles_by_category.png', dpi=300, bbox_inches='tight')
        print(f"✅ 저장: gradient_based_particles_by_category.png")

print("\n" + "=" * 80)
print("✅ 그라디언트 기반 입자 검출 완료!")
print("=" * 80)
print(f"\n💡 방법:")
print(f"   ✓ 이미지 반전: 어두운 부분(입자)을 밝게 변환")
print(f"   ✓ Sobel 필터로 밝기 변화(그라디언트) 분석")
print(f"   ✓ LoG (Laplacian of Gaussian) blob 검출")
print(f"   ✓ 구형 입자의 음영 패턴에 집중")
print(f"   ✓ Lattice fringe에 덜 민감")
print(f"   ✓ 총 {len(df)}개 입자 검출")
if len(df) > 0:
    print(f"   ✓ 평균 크기: {df[f'Diameter_{x_units}'].mean():.2f} nm")
    print(f"   ✓ 중앙값 크기: {df[f'Diameter_{x_units}'].median():.2f} nm")
print(f"\n📋 생성된 파일:")
print(f"   - {csv_filename}")
print(f"   - gradient_based_detection_comparison.png")
print(f"   - gradient_based_top_particles.png")
print(f"   - gradient_based_size_distribution.png")
print(f"   - gradient_based_particles_by_category.png")
