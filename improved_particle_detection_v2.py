"""
개선된 입자 검출 알고리즘 v2.0
- LoG (Laplacian of Gaussian) 기반 정밀 검출
- 다중 스케일 검출 (여러 크기의 입자 동시 검출)
- 향상된 전처리 및 노이즈 제거
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from skimage import filters, feature, morphology, measure, exposure
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚀 개선된 입자 검출 알고리즘 v2.0")
print("="*70)

# 1. 파일 로드
print("\n📁 EMD 파일 로딩 중...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")

if isinstance(signal, list):
    signal = signal[0]

print(f"📊 데이터 형태: {signal.data.shape}")
print(f"📏 데이터 타입: {signal.data.dtype}")

# 픽셀 크기 정보
try:
    x_axis = signal.axes_manager.signal_axes[0]
    y_axis = signal.axes_manager.signal_axes[1]
    pixel_size_x = x_axis.scale if hasattr(x_axis, 'scale') else 1.0
    pixel_size_y = y_axis.scale if hasattr(y_axis, 'scale') else 1.0
    x_units = x_axis.units if hasattr(x_axis, 'units') else 'pixels'
    y_units = y_axis.units if hasattr(y_axis, 'units') else 'pixels'
except:
    pixel_size_x = pixel_size_y = 1.0
    x_units = y_units = 'pixels'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")

# 2. 고급 전처리
print("\n" + "="*70)
print("🎨 고급 이미지 전처리")
print("="*70)

# 원본 데이터
original_data = signal.data.copy()

# 2.1 백그라운드 제거 (Rolling Ball)
print("1️⃣ Rolling Ball 백그라운드 제거...")
from skimage.morphology import ball, white_tophat

# 백그라운드 구조 요소 크기 (입자보다 크게)
radius = int(10 / pixel_size_x)  # 10nm 정도
selem = morphology.disk(radius)
background = morphology.white_tophat(original_data, selem)
bg_removed = original_data - background

print(f"   ✅ 백그라운드 제거 완료 (radius={radius} pixels)")

# 2.2 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
print("2️⃣ CLAHE 대비 향상...")
clahe = exposure.equalize_adapthist(bg_removed, clip_limit=0.03)
print("   ✅ CLAHE 완료")

# 2.3 노이즈 제거 (Non-local Means Denoising)
print("3️⃣ 노이즈 제거 (Bilateral Filter)...")
from skimage.restoration import denoise_bilateral
denoised = denoise_bilateral(clahe, sigma_color=0.05, sigma_spatial=2)
print("   ✅ 노이즈 제거 완료")

# 2.4 정규화
processed_data = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
print("   ✅ 정규화 완료 [0-1]")

# 3. LoG (Laplacian of Gaussian) 기반 다중 스케일 검출
print("\n" + "="*70)
print("🔍 LoG 기반 다중 스케일 입자 검출")
print("="*70)

# 검출할 입자 크기 범위 (지름, nm 단위)
min_diameter_nm = 1.0
max_diameter_nm = 5.0

# 픽셀 단위로 변환
min_radius_pixels = (min_diameter_nm / 2) / pixel_size_x
max_radius_pixels = (max_diameter_nm / 2) / pixel_size_x

print(f"📏 검출 범위: {min_diameter_nm}-{max_diameter_nm} {x_units} (지름)")
print(f"   픽셀 반지름: {min_radius_pixels:.2f}-{max_radius_pixels:.2f} pixels")

# 이미지 반전 (어두운 입자 검출)
inverted_data = 1.0 - processed_data

# LoG 블롭 검출
print("\n🎯 LoG 블롭 검출 실행 중...")
print("   (여러 스케일에서 동시 검출 - 시간이 걸릴 수 있습니다)")

# blob_log 매개변수
min_sigma = min_radius_pixels / np.sqrt(2)
max_sigma = max_radius_pixels / np.sqrt(2)
num_sigma = 10  # 스케일 단계 수
threshold = 0.01  # 낮을수록 더 많은 입자 검출 (기본: 0.2)
overlap = 0.5  # 겹침 허용 (0.5 = 50%)

try:
    blobs_log = feature.blob_log(
        inverted_data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap
    )
    
    # 반지름 계산 (LoG의 경우 sigma * sqrt(2))
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
    
    n_blobs_log = len(blobs_log)
    print(f"   ✅ LoG 검출 완료: {n_blobs_log}개 입자 발견")
    
except Exception as e:
    print(f"   ❌ LoG 검출 실패: {e}")
    blobs_log = np.array([])
    n_blobs_log = 0

# 4. DoG (Difference of Gaussian) 검출 (보조)
print("\n🎯 DoG 블롭 검출 실행 중...")

try:
    blobs_dog = feature.blob_dog(
        inverted_data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        overlap=overlap
    )
    
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    n_blobs_dog = len(blobs_dog)
    print(f"   ✅ DoG 검출 완료: {n_blobs_dog}개 입자 발견")
    
except Exception as e:
    print(f"   ❌ DoG 검출 실패: {e}")
    blobs_dog = np.array([])
    n_blobs_dog = 0

# 5. 하이브리드 검출: LoG + DoG 결합
print("\n🔀 하이브리드 검출: LoG + DoG 결합...")

all_blobs = []
if n_blobs_log > 0:
    all_blobs.append(blobs_log)
if n_blobs_dog > 0:
    all_blobs.append(blobs_dog)

if len(all_blobs) > 0:
    combined_blobs = np.vstack(all_blobs)
    
    # 중복 제거 (매우 가까운 블롭 병합)
    from scipy.spatial.distance import pdist, squareform
    
    if len(combined_blobs) > 1:
        # 거리 행렬 계산
        coords = combined_blobs[:, :2]
        distances = squareform(pdist(coords))
        
        # 중복 임계값 (픽셀)
        duplicate_threshold = 2.0 / pixel_size_x  # 2nm 이내면 중복으로 간주
        
        # 중복 제거
        to_keep = []
        removed = set()
        
        for i in range(len(combined_blobs)):
            if i in removed:
                continue
            
            to_keep.append(i)
            
            # 가까운 블롭 찾기
            close_blobs = np.where(distances[i] < duplicate_threshold)[0]
            for j in close_blobs:
                if j != i and j not in removed:
                    removed.add(j)
        
        unique_blobs = combined_blobs[to_keep]
        print(f"   ✅ 중복 제거: {len(combined_blobs)}개 → {len(unique_blobs)}개")
    else:
        unique_blobs = combined_blobs
        print(f"   ✅ 중복 없음: {len(unique_blobs)}개")
    
    n_particles = len(unique_blobs)
else:
    unique_blobs = np.array([])
    n_particles = 0
    print("   ❌ 검출된 입자 없음")

# 6. 입자 데이터 수집
print("\n" + "="*70)
print("📊 입자 데이터 수집 및 분석")
print("="*70)

particle_data = []

if n_particles > 0:
    for idx, blob in enumerate(unique_blobs):
        y, x, radius = blob
        
        # 픽셀 단위 직경
        diameter_pixels = 2 * radius
        
        # 실제 단위 변환
        diameter_real = diameter_pixels * pixel_size_x
        area_real = np.pi * (radius * pixel_size_x) ** 2
        
        # 입자 정보 저장
        particle_info = {
            'Particle_ID': idx + 1,
            'Centroid_X': x,
            'Centroid_Y': y,
            'Radius_pixels': radius,
            f'Diameter_{x_units}': diameter_real,
            f'Area_{x_units}2': area_real,
            'Detection_Method': 'LoG+DoG',
            'Detection_Score': 1.0  # blob 검출은 자체적으로 임계값 적용
        }
        
        particle_data.append(particle_info)
    
    # 데이터프레임 생성
    df_particles = pd.DataFrame(particle_data)
    
    # 통계 요약
    size_stats = df_particles[f'Diameter_{x_units}'].describe()
    
    print(f"\n📊 입자 크기 통계 ({x_units}):")
    print(f"   총 입자 수: {len(df_particles)}")
    print(f"   평균 지름: {size_stats['mean']:.3f} {x_units}")
    print(f"   표준편차: {size_stats['std']:.3f} {x_units}")
    print(f"   최솟값: {size_stats['min']:.3f} {x_units}")
    print(f"   최댓값: {size_stats['max']:.3f} {x_units}")
    print(f"   중앙값: {size_stats['50%']:.3f} {x_units}")
    
    # 7. 결과 시각화
    print("\n📈 결과 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Particle Detection Results (LoG+DoG)', 
                 fontsize=16, fontweight='bold')
    
    # 7.1 원본 이미지
    vmin, vmax = np.percentile(original_data, [2, 98])
    axes[0, 0].imshow(original_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    
    # 7.2 전처리된 이미지
    axes[0, 1].imshow(processed_data, cmap='gray', origin='lower')
    axes[0, 1].set_title('Preprocessed Image\n(Background Removed + CLAHE + Denoised)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    
    # 7.3 검출 결과 오버레이
    axes[1, 0].imshow(processed_data, cmap='gray', origin='lower')
    
    # 입자 원 그리기 (크기에 따라 색상 구분)
    sizes = df_particles[f'Diameter_{x_units}'].values
    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap = plt.cm.viridis
    
    for idx, blob in enumerate(unique_blobs):
        y, x, radius = blob
        diameter = sizes[idx]
        color = cmap(norm(diameter))
        
        circle = Circle((x, y), radius, fill=False, color=color, linewidth=2, alpha=0.8)
        axes[1, 0].add_patch(circle)
        
        # 처음 20개만 ID 표시
        if idx < 20:
            axes[1, 0].text(x, y, str(idx+1), ha='center', va='center',
                          fontsize=8, color='yellow', fontweight='bold')
    
    axes[1, 0].set_title(f'Detected Particles (n={n_particles})')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    
    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 0])
    cbar.set_label(f'Diameter ({x_units})')
    
    # 7.4 크기 분포 히스토그램
    n_bins = min(30, max(10, len(sizes)//10))
    counts, bin_edges, patches = axes[1, 1].hist(
        sizes, bins=n_bins, alpha=0.7, color='blue', 
        edgecolor='black', density=True
    )
    
    # 가우시안 피팅
    try:
        def gaussian(x, amp, mean, std):
            return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        amp_init = np.max(counts)
        mean_init = np.mean(sizes)
        std_init = np.std(sizes)
        
        popt, _ = curve_fit(
            gaussian, bin_centers, counts,
            p0=[amp_init, mean_init, std_init],
            maxfev=2000
        )
        
        amp_fit, mean_fit, std_fit = popt
        x_fit = np.linspace(sizes.min(), sizes.max(), 300)
        y_fit = gaussian(x_fit, amp_fit, mean_fit, std_fit)
        
        axes[1, 1].plot(x_fit, y_fit, 'r-', linewidth=2,
                       label=f'Gaussian Fit\nMean: {mean_fit:.3f} {x_units}\nStd: {std_fit:.3f} {x_units}')
        axes[1, 1].axvline(mean_fit, color='red', linestyle='--', alpha=0.7)
        
        print(f"\n📊 가우시안 피팅 결과:")
        print(f"   평균: {mean_fit:.3f} {x_units}")
        print(f"   표준편차: {std_fit:.3f} {x_units}")
        
    except Exception as e:
        print(f"   ⚠️ 가우시안 피팅 실패: {e}")
    
    # 중앙값 표시
    median_value = np.median(sizes)
    axes[1, 1].axvline(median_value, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_value:.3f} {x_units}')
    
    axes[1, 1].set_xlabel(f'Particle Diameter ({x_units})', fontweight='bold')
    axes[1, 1].set_ylabel('Probability Density', fontweight='bold')
    axes[1, 1].set_title(f'Size Distribution (n={n_particles})', fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    result_filename = f'advanced_detection_{n_particles}particles.png'
    plt.savefig(result_filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 결과 이미지 저장: '{result_filename}'")
    
    # CSV 저장
    csv_filename = f'advanced_detection_{n_particles}particles.csv'
    df_particles.to_csv(csv_filename, index=False)
    print(f"💾 데이터 저장: '{csv_filename}'")
    
    # 요약 통계 저장
    summary_filename = f'advanced_detection_{n_particles}particles_summary.txt'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("고급 입자 검출 결과 요약\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"검출 방법: LoG + DoG 하이브리드\n")
        f.write(f"총 입자 수: {n_particles}개\n\n")
        f.write(f"크기 통계 ({x_units}):\n")
        f.write(f"  - 평균: {size_stats['mean']:.3f}\n")
        f.write(f"  - 표준편차: {size_stats['std']:.3f}\n")
        f.write(f"  - 최솟값: {size_stats['min']:.3f}\n")
        f.write(f"  - 최댓값: {size_stats['max']:.3f}\n")
        f.write(f"  - 중앙값: {size_stats['50%']:.3f}\n")
        f.write(f"  - Q1 (25%): {size_stats['25%']:.3f}\n")
        f.write(f"  - Q3 (75%): {size_stats['75%']:.3f}\n\n")
        f.write(f"검출 매개변수:\n")
        f.write(f"  - 최소 직경: {min_diameter_nm} {x_units}\n")
        f.write(f"  - 최대 직경: {max_diameter_nm} {x_units}\n")
        f.write(f"  - LoG 임계값: {threshold}\n")
        f.write(f"  - 겹침 허용: {overlap * 100}%\n")
        f.write(f"  - 스케일 단계: {num_sigma}\n")
    
    print(f"💾 요약 저장: '{summary_filename}'")
    
    print("\n" + "="*70)
    print("✅ 고급 입자 검출 완료!")
    print("="*70)
    print(f"\n📊 주요 결과:")
    print(f"   - 검출된 입자: {n_particles}개")
    print(f"   - 평균 크기: {size_stats['mean']:.3f} {x_units}")
    print(f"   - 중앙값: {size_stats['50%']:.3f} {x_units}")
    
else:
    print("\n❌ 입자가 검출되지 않았습니다.")
    print("💡 제안:")
    print("   1. threshold 값을 낮추세요 (현재: {threshold})")
    print("   2. 검출 범위를 조정하세요")
    print("   3. 전처리 매개변수를 조정하세요")

plt.show()
