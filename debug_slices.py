import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

# EMD 파일 로드
print("📁 EMD 파일 로딩 중...")
signal = hs.load("Camera_Ceta_0956_115_kx_0002.emd")

if isinstance(signal, list):
    signal = signal[0]

print(f"📊 전체 데이터 형태: {signal.data.shape}")
print(f"📈 전체 데이터 범위: {np.min(signal.data):.2f} ~ {np.max(signal.data):.2f}")

# 각 슬라이스별 상세 분석
for i in range(signal.data.shape[0]):
    slice_data = signal.data[i]
    print(f"\n🔍 슬라이스 {i} 분석:")
    print(f"   형태: {slice_data.shape}")
    print(f"   범위: {np.min(slice_data):.2f} ~ {np.max(slice_data):.2f}")
    print(f"   평균: {np.mean(slice_data):.2f}")
    print(f"   표준편차: {np.std(slice_data):.2f}")
    print(f"   0인 픽셀 수: {np.sum(slice_data == 0)}")
    print(f"   고유값 개수: {len(np.unique(slice_data))}")

# 모든 슬라이스를 한번에 시각화
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    slice_data = signal.data[i]
    
    # 다양한 방법으로 시각화 시도
    im = axes[i].imshow(slice_data, cmap='gray', origin='lower')
    axes[i].set_title(f'Slice {i}\nRange: {np.min(slice_data):.1f}-{np.max(slice_data):.1f}')
    plt.colorbar(im, ax=axes[i], shrink=0.6)

plt.tight_layout()
plt.savefig('all_slices_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n💾 모든 슬라이스 비교 이미지 저장: 'all_slices_comparison.png'")

# 각 슬라이스를 개별적으로 다양한 컬러맵으로 시도
print(f"\n🎨 다양한 컬러맵으로 슬라이스 2 테스트...")
slice_2 = signal.data[2]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colormaps = ['gray', 'viridis', 'plasma', 'hot', 'jet', 'turbo']
for i, cmap in enumerate(colormaps):
    im = axes[i].imshow(slice_2, cmap=cmap, origin='lower')
    axes[i].set_title(f'Colormap: {cmap}')
    plt.colorbar(im, ax=axes[i], shrink=0.6)

plt.tight_layout()
plt.savefig('slice2_colormaps.png', dpi=300, bbox_inches='tight')
print(f"💾 슬라이스 2 컬러맵 비교 저장: 'slice2_colormaps.png'")

print("✅ 디버깅 분석 완료!")