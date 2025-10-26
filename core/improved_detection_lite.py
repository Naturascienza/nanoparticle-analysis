"""
경량화된 개선 입자 검출 알고리즘
- 메모리 효율적인 전처리
- LoG 블롭 검출
- 기존 방법 대비 향상된 정확도
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from skimage import filters, feature, morphology, exposure
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚀 경량화된 입자 검출 알고리즘")
print("="*70)

# 1. 파일 로드
print("\n📁 EMD 파일 로딩 중...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")

if isinstance(signal, list):
    signal = signal[0]

print(f"📊 데이터 형태: {signal.data.shape}")

# 픽셀 크기
try:
    x_axis = signal.axes_manager.signal_axes[0]
    pixel_size_x = x_axis.scale if hasattr(x_axis, 'scale') else 1.0
    x_units = x_axis.units if hasattr(x_axis, 'units') else 'pixels'
except:
    pixel_size_x = 1.0
    x_units = 'pixels'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")

# 2. 효율적인 전처리
print("\n🎨 이미지 전처리...")

# 원본 데이터
original_data = signal.data.copy().astype(np.float32)  # float32로 메모리 절약

# 2.1 백그라운드 제거 (간단한 방법)
print("  - 백그라운드 제거 (가우시안 필터)...")
from scipy.ndimage import gaussian_filter
background = gaussian_filter(original_data, sigma=20)  # 큰 스케일 블러
bg_removed = original_data - background
bg_removed = np.clip(bg_removed, 0, None)  # 음수 제거

# 2.2 정규화
print("  - 정규화...")
p2, p98 = np.percentile(bg_removed, [2, 98])
normalized = np.clip((bg_removed - p2) / (p98 - p2), 0, 1)

# 2.3 대비 향상 (간단한 CLAHE)
print("  - CLAHE 대비 향상...")
clahe = exposure.equalize_adapthist(normalized, clip_limit=0.03)

# 2.4 약간의 가우시안 블러 (노이즈 제거)
print("  - 노이즈 제거...")
processed_data = gaussian_filter(clahe, sigma=1.0)

print("  ✅ 전처리 완료")

# 3. 사용자 입력
print("\n" + "="*70)
print("🔍 입자 검출 설정")
print("="*70)

# 검출 범위
print(f"\n검출할 입자 크기 범위 ({x_units}):")
min_size = float(input(f"  최소 직경 [기본: 1.0]: ") or "1.0")
max_size = float(input(f"  최대 직경 [기본: 5.0]: ") or "5.0")

# LoG 임계값
print(f"\n검출 민감도 (낮을수록 더 많이 검출):")
threshold = float(input("  임계값 (0.001-0.1) [기본: 0.01]: ") or "0.01")

# 겹침 허용
overlap = float(input("  겹침 허용 비율 (0-1) [기본: 0.5]: ") or "0.5")

print(f"\n✅ 설정 완료:")
print(f"  - 검출 범위: {min_size}-{max_size} {x_units}")
print(f"  - 임계값: {threshold}")
print(f"  - 겹침 허용: {overlap * 100}%")

# 4. LoG 블롭 검출
print("\n" + "="*70)
print("🎯 LoG 블롭 검출 실행 중...")
print("="*70)

# 이미지 반전 (어두운 입자)
inverted = 1.0 - processed_data

# 반지름 범위
min_radius_px = (min_size / 2) / pixel_size_x
max_radius_px = (max_size / 2) / pixel_size_x

min_sigma = min_radius_px / np.sqrt(2)
max_sigma = max_radius_px / np.sqrt(2)

print(f"  반지름 범위: {min_radius_px:.2f}-{max_radius_px:.2f} 픽셀")
print(f"  시그마 범위: {min_sigma:.2f}-{max_sigma:.2f}")

try:
    blobs = feature.blob_log(
        inverted,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=10,
        threshold=threshold,
        overlap=overlap
    )
    
    # 반지름 보정
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    n_particles = len(blobs)
    print(f"\n✅ 검출 완료: {n_particles}개 입자")
    
except Exception as e:
    print(f"\n❌ 검출 실패: {e}")
    blobs = np.array([])
    n_particles = 0

# 5. 결과 처리
if n_particles > 0:
    print("\n📊 데이터 수집 중...")
    
    particle_data = []
    for idx, (y, x, r) in enumerate(blobs):
        diameter_nm = 2 * r * pixel_size_x
        area_nm2 = np.pi * (r * pixel_size_x) ** 2
        
        particle_data.append({
            'Particle_ID': idx + 1,
            'Centroid_X': x,
            'Centroid_Y': y,
            'Radius_pixels': r,
            f'Diameter_{x_units}': diameter_nm,
            f'Area_{x_units}2': area_nm2
        })
    
    df = pd.DataFrame(particle_data)
    
    # 통계
    sizes = df[f'Diameter_{x_units}']
    print(f"\n📊 크기 통계:")
    print(f"  평균: {sizes.mean():.3f} {x_units}")
    print(f"  표준편차: {sizes.std():.3f} {x_units}")
    print(f"  중앙값: {sizes.median():.3f} {x_units}")
    print(f"  최소-최대: {sizes.min():.3f}-{sizes.max():.3f} {x_units}")
    
    # 6. 시각화
    print("\n📈 시각화 중...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'LoG Detection Results (n={n_particles})', fontsize=16, fontweight='bold')
    
    # 6.1 원본
    vmin, vmax = np.percentile(original_data, [2, 98])
    axes[0].imshow(original_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Image')
    
    # 6.2 전처리
    axes[1].imshow(processed_data, cmap='gray', origin='lower')
    axes[1].set_title('Preprocessed')
    
    # 6.3 검출 결과
    axes[2].imshow(processed_data, cmap='gray', origin='lower')
    
    # 크기별 색상
    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap_colors = plt.cm.viridis
    
    for idx, (y, x, r) in enumerate(blobs):
        color = cmap_colors(norm(sizes.iloc[idx]))
        circle = Circle((x, y), r, fill=False, color=color, linewidth=1.5, alpha=0.8)
        axes[2].add_patch(circle)
        
        if idx < 20:
            axes[2].text(x, y, str(idx+1), ha='center', va='center',
                        fontsize=7, color='yellow', fontweight='bold')
    
    axes[2].set_title(f'Detected (n={n_particles})')
    
    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2])
    cbar.set_label(f'Diameter ({x_units})')
    
    plt.tight_layout()
    
    # 7. 히스토그램
    fig2, ax = plt.subplots(figsize=(10, 7))
    
    n_bins = min(30, max(10, n_particles//10))
    counts, edges, _ = ax.hist(sizes, bins=n_bins, alpha=0.7, 
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
        
        ax.plot(x_fit, y_fit, 'r-', linewidth=2,
               label=f'Gaussian Fit\nMean: {mean:.3f} {x_units}\nStd: {std:.3f} {x_units}')
        ax.axvline(mean, color='red', linestyle='--', alpha=0.7)
        
        print(f"\n📊 가우시안 피팅:")
        print(f"  평균: {mean:.3f} {x_units}")
        print(f"  표준편차: {std:.3f} {x_units}")
    except:
        pass
    
    ax.axvline(sizes.median(), color='green', linestyle='--', linewidth=2,
              label=f'Median: {sizes.median():.3f} {x_units}')
    
    ax.set_xlabel(f'Diameter ({x_units})', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title('Particle Size Distribution', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 8. 저장
    print("\n💾 결과 저장 중...")
    
    suffix = f"_log_{n_particles}p_thresh{threshold}"
    
    fig.savefig(f'detection_result{suffix}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'size_histogram{suffix}.png', dpi=300, bbox_inches='tight')
    df.to_csv(f'particle_data{suffix}.csv', index=False)
    
    print(f"  ✅ detection_result{suffix}.png")
    print(f"  ✅ size_histogram{suffix}.png")
    print(f"  ✅ particle_data{suffix}.csv")
    
    print("\n" + "="*70)
    print("✅ 검출 완료!")
    print("="*70)
    print(f"\n📊 최종 결과:")
    print(f"  입자 수: {n_particles}개")
    print(f"  평균 크기: {sizes.mean():.3f} {x_units}")
    print(f"  중앙값: {sizes.median():.3f} {x_units}")
    
    # 크기별 분포
    small = len(sizes[sizes < 2.0])
    medium = len(sizes[(sizes >= 2.0) & (sizes < 3.0)])
    large = len(sizes[sizes >= 3.0])
    
    print(f"\n📊 크기 분포:")
    print(f"  <2.0 {x_units}: {small}개 ({small/n_particles*100:.1f}%)")
    print(f"  2.0-3.0 {x_units}: {medium}개 ({medium/n_particles*100:.1f}%)")
    print(f"  ≥3.0 {x_units}: {large}개 ({large/n_particles*100:.1f}%)")
    
else:
    print("\n❌ 입자가 검출되지 않았습니다.")
    print("\n💡 개선 방안:")
    print("  1. 임계값을 낮추세요 (예: 0.005)")
    print("  2. 검출 범위를 넓히세요")
    print("  3. 겹침 허용을 높이세요 (예: 0.7)")

plt.show()

print("\n" + "="*70)
print("프로그램 종료")
print("="*70)
