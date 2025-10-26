import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

# EMD 파일 로드
signal = hs.load("Camera_Ceta_0956_115_kx_0002.emd")
if isinstance(signal, list):
    signal = signal[0]

# 슬라이스 2 선택
slice_data = signal.data[2].copy()
print(f"📊 슬라이스 2 원본 데이터 범위: {np.min(slice_data):.1f} ~ {np.max(slice_data):.1f}")

# 4가지 대비 개선 방법 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('이미지 대비 개선 방법 비교 - 슬라이스 2', fontsize=16, fontweight='bold')

# 1. 원본 (기본)
data1 = slice_data.copy()
vmin1, vmax1 = np.min(data1), np.max(data1)
im1 = axes[0,0].imshow(data1, cmap='gray', origin='lower', vmin=vmin1, vmax=vmax1)
axes[0,0].set_title(f'1. 원본 범위\n({vmin1:.1f} ~ {vmax1:.1f})')
plt.colorbar(im1, ax=axes[0,0], shrink=0.6)

# 2. 백분위 기반 (2%-98%)
data2 = slice_data.copy()
vmin2, vmax2 = np.percentile(data2, [2, 98])
im2 = axes[0,1].imshow(data2, cmap='gray', origin='lower', vmin=vmin2, vmax=vmax2)
axes[0,1].set_title(f'2. 백분위 기반 (2%-98%)\n({vmin2:.1f} ~ {vmax2:.1f})')
plt.colorbar(im2, ax=axes[0,1], shrink=0.6)

# 3. 표준편차 기반 (평균±3σ)
data3 = slice_data.copy()
mean_val = np.mean(data3)
std_val = np.std(data3)
vmin3, vmax3 = mean_val - 3*std_val, mean_val + 3*std_val
im3 = axes[1,0].imshow(data3, cmap='gray', origin='lower', vmin=vmin3, vmax=vmax3)
axes[1,0].set_title(f'3. 표준편차 기반 (평균±3σ)\n({vmin3:.1f} ~ {vmax3:.1f})')
plt.colorbar(im3, ax=axes[1,0], shrink=0.6)

# 4. 히스토그램 균등화
data4 = exposure.equalize_hist(slice_data)
vmin4, vmax4 = np.min(data4), np.max(data4)
im4 = axes[1,1].imshow(data4, cmap='gray', origin='lower', vmin=vmin4, vmax=vmax4)
axes[1,1].set_title(f'4. 히스토그램 균등화\n({vmin4:.3f} ~ {vmax4:.3f})')
plt.colorbar(im4, ax=axes[1,1], shrink=0.6)

# 각 이미지에 스케일바 추가 (50nm)
pixel_size = 0.228  # nm/pixel
scale_length = 50   # nm
bar_length_pixels = scale_length / pixel_size

for ax in axes.flatten():
    height, width = 1024, 1024
    margin = 0.05
    bar_height = height * 0.02
    x_pos = width - bar_length_pixels - width * margin
    y_pos = height * margin
    
    # 스케일바
    from matplotlib.patches import Rectangle
    bar = Rectangle((x_pos, y_pos), bar_length_pixels, bar_height, 
                   facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(bar)
    
    # 텍스트
    ax.text(x_pos + bar_length_pixels/2, y_pos + height * 0.04, '50 nm',
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('contrast_comparison_slice2.png', dpi=300, bbox_inches='tight')
print("💾 대비 개선 비교 이미지 저장: 'contrast_comparison_slice2.png'")

print("✅ 모든 대비 개선 방법 비교 완료!")
print("📋 추천:")
print("   • 일반적인 경우: 백분위 기반 (2%-98%)")
print("   • 세부 구조 강조: 히스토그램 균등화")
print("   • 아웃라이어 제거: 표준편차 기반")