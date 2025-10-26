"""
시각적 확인 기반 입자 검출
- 단계별 시각화로 사용자가 직접 확인
- 수동 임계값 조정 가능
- 실제 1개 입자 정확히 검출 목표
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from skimage import filters, segmentation, measure, morphology
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔍 시각적 확인 기반 정밀 검출")
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

# ROI 설정
print("\n🎯 ROI 설정")
center_x = int(input(f"ROI 중심 X [1024]: ") or "1024")
center_y = int(input(f"ROI 중심 Y [1024]: ") or "1024")
roi_size = int(input("ROI 크기 [250]: ") or "250")

half = roi_size // 2
x_start = max(0, center_x - half)
y_start = max(0, center_y - half)
x_end = min(signal.data.shape[1], center_x + half)
y_end = min(signal.data.shape[0], center_y + half)

roi_coords = (x_start, y_start, x_end, y_end)
analysis_data = signal.data[y_start:y_end, x_start:x_end].copy()

print(f"✅ ROI: ({x_start},{y_start}) → ({x_end},{y_end})")
print(f"   크기: {analysis_data.shape}")

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

# 이미지 반전 (어두운 입자 → 밝게)
inverted = 1.0 - smoothed

print("✅ 전처리 완료")

# 대화형 임계값 설정
print("\n" + "="*70)
print("🎯 임계값 설정 (대화형)")
print("="*70)

# 히스토그램 표시
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('임계값 설정을 위한 시각적 확인', fontsize=14, fontweight='bold')

# 원본
axes[0, 0].imshow(analysis_data, cmap='gray', origin='lower')
axes[0, 0].set_title('원본 (ROI)')
axes[0, 0].axis('off')

# 전처리
axes[0, 1].imshow(inverted, cmap='gray', origin='lower')
axes[0, 1].set_title('전처리 후 (어두운 입자 → 밝게)')
axes[0, 1].axis('off')

# 히스토그램
axes[1, 0].hist(inverted.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[1, 0].set_xlabel('강도')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].set_title('강도 히스토그램')
axes[1, 0].grid(True, alpha=0.3)

# 자동 임계값들
thresh_otsu = filters.threshold_otsu(inverted)
thresh_li = filters.threshold_li(inverted)
thresh_90 = np.percentile(inverted, 90)
thresh_85 = np.percentile(inverted, 85)
thresh_80 = np.percentile(inverted, 80)

axes[1, 0].axvline(thresh_otsu, color='red', linestyle='--', linewidth=2, label=f'Otsu: {thresh_otsu:.3f}')
axes[1, 0].axvline(thresh_li, color='green', linestyle='--', linewidth=2, label=f'Li: {thresh_li:.3f}')
axes[1, 0].axvline(thresh_85, color='orange', linestyle='--', linewidth=2, label=f'85%ile: {thresh_85:.3f}')
axes[1, 0].legend()

# 빈 공간 (나중에 이진화 결과 표시)
axes[1, 1].text(0.5, 0.5, '임계값 선택 후\n이진화 결과 표시', 
               ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

print(f"\n📊 자동 임계값 참고:")
print(f"  1. Otsu: {thresh_otsu:.4f}")
print(f"  2. Li: {thresh_li:.4f}")
print(f"  3. 90 percentile: {thresh_90:.4f}")
print(f"  4. 85 percentile: {thresh_85:.4f}")
print(f"  5. 80 percentile: {thresh_80:.4f}")

# 사용자 선택
print(f"\n임계값을 선택하세요:")
print("1~5: 위 자동 임계값 사용")
print("0: 수동 입력")

choice = input("선택 [0-5]: ").strip()

if choice == "1":
    threshold = thresh_otsu
    method = "Otsu"
elif choice == "2":
    threshold = thresh_li
    method = "Li"
elif choice == "3":
    threshold = thresh_90
    method = "90%ile"
elif choice == "4":
    threshold = thresh_85
    method = "85%ile"
elif choice == "5":
    threshold = thresh_80
    method = "80%ile"
else:
    threshold = float(input("임계값 입력 (0-1): "))
    method = "Manual"

print(f"\n✅ 선택: {method}, 임계값 = {threshold:.4f}")

# 이진화
binary = inverted > threshold

# 모폴로지 정리
print("\n🔧 모폴로지 정리...")

# 최소 크기 (매우 작게 - 1개 입자 놓치지 않도록)
min_area_pixels = 50  # 약 0.056 nm²
print(f"  최소 면적: {min_area_pixels} 픽셀²")

binary_cleaned = morphology.remove_small_objects(binary, min_size=min_area_pixels)
binary_cleaned = morphology.binary_opening(binary_cleaned, morphology.disk(1))

# 라벨링
labeled = measure.label(binary_cleaned)
n_regions = labeled.max()

print(f"  ✅ 검출된 영역: {n_regions}개")

# 이진화 결과 시각화 업데이트
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(f'이진화 결과 (임계값={threshold:.4f}, 영역={n_regions}개)', fontsize=14, fontweight='bold')

axes2[0].imshow(analysis_data, cmap='gray', origin='lower')
axes2[0].set_title('원본')
axes2[0].axis('off')

axes2[1].imshow(inverted, cmap='gray', origin='lower')
axes2[1].set_title('전처리 (반전)')
axes2[1].axis('off')

# 이진화 + 레이블 오버레이
axes2[2].imshow(analysis_data, cmap='gray', origin='lower')
# 이진 마스크를 빨간색으로 오버레이
mask_overlay = np.zeros((*binary_cleaned.shape, 4))
mask_overlay[binary_cleaned] = [1, 0, 0, 0.3]  # 빨간색, 투명도 0.3
axes2[2].imshow(mask_overlay, origin='lower')
axes2[2].set_title(f'검출 영역 (빨간색)')
axes2[2].axis('off')

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# 입자 분석
if n_regions > 0:
    props = measure.regionprops(labeled, intensity_image=smoothed)
    
    print(f"\n📊 검출된 영역 상세 분석:")
    print("="*70)
    
    particle_data = []
    
    for i, prop in enumerate(props, 1):
        area_pixels = prop.area
        equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        equiv_diameter_nm = equiv_diameter_pixels * pixel_size_x
        
        # 강도
        mean_intensity = prop.mean_intensity
        max_intensity = prop.max_intensity
        
        # 위치
        centroid_y, centroid_x = prop.centroid
        
        # 형태
        if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
            if prop.minor_axis_length > 0:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            else:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0
        
        perimeter = prop.perimeter if hasattr(prop, 'perimeter') else 0
        circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
        
        print(f"\n영역 #{i}:")
        print(f"  위치: ({centroid_x:.1f}, {centroid_y:.1f})")
        print(f"  면적: {area_pixels} 픽셀² ({area_pixels * pixel_size_x**2:.4f} {x_units}²)")
        print(f"  직경: {equiv_diameter_nm:.3f} {x_units}")
        print(f"  평균 강도: {mean_intensity:.3f}")
        print(f"  최대 강도: {max_intensity:.3f}")
        print(f"  종횡비: {aspect_ratio:.2f}")
        print(f"  원형도: {circularity:.3f}")
        
        # 이 영역이 실제 입자인지 사용자 확인
        is_particle = input(f"  ❓ 이 영역이 실제 입자입니까? [Y/n]: ").strip().lower()
        
        if is_particle in ['', 'y', 'yes']:
            particle_data.append({
                'Particle_ID': len(particle_data) + 1,
                'Centroid_X': centroid_x + roi_coords[0],
                'Centroid_Y': centroid_y + roi_coords[1],
                f'Diameter_{x_units}': equiv_diameter_nm,
                f'Area_{x_units}2': area_pixels * pixel_size_x**2,
                'Mean_Intensity': mean_intensity,
                'Aspect_Ratio': aspect_ratio,
                'Circularity': circularity
            })
            print(f"    ✅ 입자로 인정")
        else:
            print(f"    ❌ 제외")
    
    n_particles = len(particle_data)
    
    print("\n" + "="*70)
    print(f"🎉 최종 확인된 입자: {n_particles}개")
    print("="*70)
    
    if n_particles > 0:
        df = pd.DataFrame(particle_data)
        
        # 최종 시각화
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
        fig3.suptitle(f'최종 검출 결과 (확인된 입자: {n_particles}개)', fontsize=14, fontweight='bold')
        
        # 원본 + 검출
        vmin, vmax = np.percentile(analysis_data, [2, 98])
        axes3[0].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes3[0].set_title('원본 ROI')
        axes3[0].axis('off')
        
        axes3[1].imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        
        # 확인된 입자 표시
        for _, p in df.iterrows():
            x = p['Centroid_X'] - roi_coords[0]
            y = p['Centroid_Y'] - roi_coords[1]
            r = p[f'Diameter_{x_units}'] / (2 * pixel_size_x)
            
            circle = Circle((x, y), r, fill=False, color='red', linewidth=2.5)
            axes3[1].add_patch(circle)
            
            axes3[1].text(x, y, f"#{p['Particle_ID']}", ha='center', va='center',
                         fontsize=12, color='yellow', fontweight='bold',
                         bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7))
            
            axes3[1].text(x, y + r + 10, f"{p[f'Diameter_{x_units}']:.2f}{x_units}",
                         ha='center', va='top', fontsize=10, color='yellow', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        axes3[1].set_title(f'검출된 입자 ({n_particles}개)')
        axes3[1].axis('off')
        
        plt.tight_layout()
        
        # 저장
        suffix = f"_verified_{n_particles}p"
        fig3.savefig(f'detection_result{suffix}.png', dpi=300, bbox_inches='tight')
        df.to_csv(f'particle_data{suffix}.csv', index=False)
        
        print(f"\n💾 저장:")
        print(f"  detection_result{suffix}.png")
        print(f"  particle_data{suffix}.csv")
        
        # 통계
        sizes = df[f'Diameter_{x_units}']
        print(f"\n📊 입자 통계:")
        print(f"  개수: {n_particles}개")
        if n_particles == 1:
            print(f"  크기: {sizes.iloc[0]:.3f} {x_units}")
        else:
            print(f"  평균: {sizes.mean():.3f} {x_units}")
            print(f"  범위: {sizes.min():.3f} - {sizes.max():.3f} {x_units}")
        
        plt.show()
        
    else:
        print("❌ 확인된 입자가 없습니다.")

else:
    print("\n❌ 검출된 영역이 없습니다.")
    print("💡 임계값을 낮추거나 조정해보세요.")

print("\n" + "="*70)
print("프로그램 종료")
print("="*70)
