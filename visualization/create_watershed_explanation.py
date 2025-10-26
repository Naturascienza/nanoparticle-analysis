"""
Watershed 분할 Before/After 설명 이미지 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("🎨 Watershed 분할 설명 이미지 생성 중...")

# 1. 샘플 이미지 생성 (응집된 입자 시뮬레이션)
print("  1. 응집된 입자 샘플 이미지 생성...")
img_size = 400
image = np.zeros((img_size, img_size))

# 응집된 입자 그룹 생성
particle_groups = [
    # 2개 입자 응집 (중심 간격이 짧음)
    [(100, 100, 25, 0.9), (130, 100, 22, 0.85)],
    # 3개 입자 응집
    [(250, 120, 20, 0.9), (275, 130, 18, 0.85), (265, 100, 16, 0.8)],
    # 4개 입자 응집 (십자 형태)
    [(120, 250, 15, 0.9), (145, 250, 14, 0.85), 
     (132, 235, 13, 0.8), (132, 265, 12, 0.75)],
    # 큰 입자 + 작은 입자 응집
    [(300, 280, 30, 0.95), (340, 270, 15, 0.8)],
    # 개별 입자 (비교용)
    [(80, 320, 18, 0.85)],
    [(320, 50, 20, 0.9)],
]

# 입자 그리기
for group in particle_groups:
    for x, y, r, intensity in group:
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= r**2
        image[mask] = np.maximum(image[mask], intensity)

# 노이즈 및 백그라운드 추가
noise = np.random.normal(0, 0.03, image.shape)
image = image + noise
background = gaussian_filter(np.random.random((img_size, img_size)) * 0.2, sigma=30)
image = image + background
image = np.clip(image, 0, 1)
image = (image - image.min()) / (image.max() - image.min())

print("  2. Watershed 분할 적용 전/후 처리...")

# Before: 단순 임계값 처리
threshold = 0.5
binary_simple = image > threshold

# 형태학적 연산으로 노이즈 제거
binary_simple = morphology.remove_small_objects(binary_simple, min_size=50)
binary_simple = morphology.closing(binary_simple, morphology.disk(3))

# Before 레이블링 (응집된 입자는 하나로 계산됨)
labels_before = measure.label(binary_simple)
regions_before = measure.regionprops(labels_before)

print(f"     Before: {len(regions_before)}개 영역 검출")

# After: Watershed 분할
binary_watershed = image > threshold
binary_watershed = morphology.remove_small_objects(binary_watershed, min_size=50)
binary_watershed = morphology.closing(binary_watershed, morphology.disk(3))

# 거리 변환
distance = ndimage.distance_transform_edt(binary_watershed)

# 로컬 최대값 찾기 (입자 중심)
coords = peak_local_max(
    distance,
    footprint=np.ones((15, 15)),  # 분리 민감도
    labels=binary_watershed,
    min_distance=10
)

# 마커 생성
mask_markers = np.zeros(distance.shape, dtype=bool)
mask_markers[tuple(coords.T)] = True
markers, _ = ndimage.label(mask_markers)

# Watershed 적용
labels_after = segmentation.watershed(-distance, markers, mask=binary_watershed)
regions_after = measure.regionprops(labels_after)

print(f"     After: {len(regions_after)}개 입자로 분리")
print(f"     응집체에서 {len(regions_after) - len(regions_before)}개 추가 분리")

# 3. 시각화
print("  3. Before/After 비교 이미지 생성...")

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)

# ===== 상단: Before (단순 임계값) =====
# (a) 원본 이미지
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image, cmap='gray', origin='lower')
ax1.set_title('(a) 원본 TEM 이미지\n(응집된 입자 포함)', fontsize=12, fontweight='bold')
ax1.axis('off')

# (b) 이진화
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(binary_simple, cmap='gray', origin='lower')
ax2.set_title('(b) 단순 임계값 처리\n(이진화)', fontsize=12, fontweight='bold')
ax2.axis('off')

# (c) Before 검출 결과
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(image, cmap='gray', origin='lower', alpha=0.7)
# 컬러맵으로 레이블 표시
label_overlay = np.ma.masked_where(labels_before == 0, labels_before)
ax3.imshow(label_overlay, cmap='tab20', origin='lower', alpha=0.6)

# 경계선 그리기
for region in regions_before:
    y0, x0 = region.centroid
    circle = Circle((x0, y0), 3, color='red', fill=True, zorder=10)
    ax3.add_patch(circle)
    ax3.text(x0, y0-15, f'{region.label}', ha='center', va='top',
            color='yellow', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

ax3.set_title(f'(c) Before: {len(regions_before)}개 영역 검출\n(응집체 = 1개로 계산)', 
             fontsize=12, fontweight='bold', color='red')
ax3.axis('off')

# (d) 문제점 확대
ax4 = fig.add_subplot(gs[0, 3])
# 가장 큰 응집체 영역 확대
largest_region = max(regions_before, key=lambda r: r.area)
minr, minc, maxr, maxc = largest_region.bbox
padding = 30
minr, minc = max(0, minr-padding), max(0, minc-padding)
maxr, maxc = min(img_size, maxr+padding), min(img_size, maxc+padding)

ax4.imshow(image[minr:maxr, minc:maxc], cmap='gray', origin='lower', alpha=0.7)
label_crop = labels_before[minr:maxr, minc:maxc]
label_overlay = np.ma.masked_where(label_crop == 0, label_crop)
ax4.imshow(label_overlay, cmap='tab20', origin='lower', alpha=0.6)

ax4.set_title('(d) 문제점: 응집된 입자\n여러 개가 1개로 측정됨', 
             fontsize=12, fontweight='bold', color='darkred')
ax4.axis('off')

# 문제점 화살표 및 텍스트
ax4.annotate('', xy=(0.7, 0.5), xytext=(0.3, 0.5),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='<->', color='red', lw=3))
ax4.text(0.5, 0.6, '실제로는\n여러 입자', ha='center', va='bottom',
        transform=ax4.transAxes, fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# ===== 하단: After (Watershed) =====
# (e) 거리 변환
ax5 = fig.add_subplot(gs[1, 0])
im = ax5.imshow(distance, cmap='hot', origin='lower')
# 로컬 최대값 표시
ax5.plot(coords[:, 1], coords[:, 0], 'g*', markersize=10, markeredgecolor='lime', markeredgewidth=1.5)
ax5.set_title('(e) 거리 변환\n(녹색★ = 입자 중심)', fontsize=12, fontweight='bold')
ax5.axis('off')
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label='거리')

# (f) Watershed 마커
ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(binary_watershed, cmap='gray', origin='lower', alpha=0.5)
marker_overlay = np.ma.masked_where(markers == 0, markers)
ax6.imshow(marker_overlay, cmap='tab20', origin='lower', alpha=0.8)
ax6.set_title('(f) Watershed 마커\n(입자별 시드 포인트)', fontsize=12, fontweight='bold')
ax6.axis('off')

# (g) After 검출 결과
ax7 = fig.add_subplot(gs[1, 2])
ax7.imshow(image, cmap='gray', origin='lower', alpha=0.7)
# 컬러맵으로 레이블 표시
label_overlay = np.ma.masked_where(labels_after == 0, labels_after)
ax7.imshow(label_overlay, cmap='tab20', origin='lower', alpha=0.6)

# 경계선 그리기
for region in regions_after:
    y0, x0 = region.centroid
    circle = Circle((x0, y0), 2, color='lime', fill=True, zorder=10)
    ax7.add_patch(circle)

ax7.set_title(f'(g) After: {len(regions_after)}개 입자로 분리\n(응집체 → 개별 입자)', 
             fontsize=12, fontweight='bold', color='green')
ax7.axis('off')

# (h) 개선 효과 확대
ax8 = fig.add_subplot(gs[1, 3])
ax8.imshow(image[minr:maxr, minc:maxc], cmap='gray', origin='lower', alpha=0.7)
label_crop = labels_after[minr:maxr, minc:maxc]
label_overlay = np.ma.masked_where(label_crop == 0, label_crop)
ax8.imshow(label_overlay, cmap='tab20', origin='lower', alpha=0.6)

# 분리된 입자 개수 표시
unique_labels = np.unique(label_crop)
unique_labels = unique_labels[unique_labels > 0]

for label in unique_labels:
    mask = label_crop == label
    if np.sum(mask) > 10:
        y_coords, x_coords = np.where(mask)
        y_center, x_center = y_coords.mean(), x_coords.mean()
        ax8.text(x_center, y_center, str(label), ha='center', va='center',
                color='white', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='green', alpha=0.8))

ax8.set_title(f'(h) 개선: {len(unique_labels)}개로 정확히 분리\n(크기 측정 정확도 향상)', 
             fontsize=12, fontweight='bold', color='darkgreen')
ax8.axis('off')

# 전체 제목
fig.suptitle('Watershed 알고리즘: 응집 입자 자동 분리\nBefore (단순 임계값) vs After (Watershed 분할)', 
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# 저장
filename = 'Watershed_before_after.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 저장 완료: {filename}")

print("\n" + "="*70)
print("📊 추가: Watershed 작동 원리 단계별 설명")
print("="*70)

# 4. Watershed 작동 원리 상세 설명
fig2 = plt.figure(figsize=(18, 6))

# (a) 원본
ax1 = fig2.add_subplot(1, 5, 1)
ax1.imshow(image[minr:maxr, minc:maxc], cmap='gray', origin='lower')
ax1.set_title('Step 1\n원본 이미지', fontsize=12, fontweight='bold')
ax1.axis('off')

# (b) 이진화
ax2 = fig2.add_subplot(1, 5, 2)
ax2.imshow(binary_watershed[minr:maxr, minc:maxc], cmap='gray', origin='lower')
ax2.set_title('Step 2\n이진화', fontsize=12, fontweight='bold')
ax2.axis('off')

# (c) 거리 변환
ax3 = fig2.add_subplot(1, 5, 3)
im = ax3.imshow(distance[minr:maxr, minc:maxc], cmap='hot', origin='lower')
ax3.set_title('Step 3\n거리 변환', fontsize=12, fontweight='bold')
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# (d) 마커 (시드)
ax4 = fig2.add_subplot(1, 5, 4)
ax4.imshow(binary_watershed[minr:maxr, minc:maxc], cmap='gray', origin='lower', alpha=0.5)
marker_crop = markers[minr:maxr, minc:maxc]
marker_overlay = np.ma.masked_where(marker_crop == 0, marker_crop)
ax4.imshow(marker_overlay, cmap='tab20', origin='lower', alpha=0.9)
ax4.set_title('Step 4\n마커 생성', fontsize=12, fontweight='bold')
ax4.axis('off')

# (e) Watershed 결과
ax5 = fig2.add_subplot(1, 5, 5)
ax5.imshow(image[minr:maxr, minc:maxc], cmap='gray', origin='lower', alpha=0.7)
label_crop = labels_after[minr:maxr, minc:maxc]
label_overlay = np.ma.masked_where(label_crop == 0, label_crop)
ax5.imshow(label_overlay, cmap='tab20', origin='lower', alpha=0.7)

# 경계선 강조
from skimage.segmentation import find_boundaries
boundaries = find_boundaries(label_crop, mode='outer')
ax5.contour(boundaries, colors='lime', linewidths=2)

ax5.set_title('Step 5\nWatershed 분할', fontsize=12, fontweight='bold')
ax5.axis('off')

fig2.suptitle('Watershed 알고리즘 작동 원리 (단계별)', fontsize=16, fontweight='bold')
plt.tight_layout()

filename2 = 'Watershed_step_by_step.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename2}")

print("\n" + "="*70)
print("📊 추가: 통계 비교 차트")
print("="*70)

# 5. 통계 비교
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) 검출 개수 비교
ax = axes[0]
methods = ['Before\n(단순 임계값)', 'After\n(Watershed)']
counts = [len(regions_before), len(regions_after)]
colors = ['red', 'green']
bars = ax.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}개',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('검출된 입자 개수', fontsize=12, fontweight='bold')
ax.set_title('(a) 검출 개수 비교', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts) * 1.2)
ax.grid(axis='y', alpha=0.3)

# 개선율 표시
improvement = ((counts[1] - counts[0]) / counts[0]) * 100
ax.text(0.5, 0.95, f'개선율: +{improvement:.1f}%',
       ha='center', va='top', transform=ax.transAxes,
       fontsize=12, fontweight='bold', color='blue',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# (b) 평균 크기 비교
ax = axes[1]
areas_before = [r.area for r in regions_before]
areas_after = [r.area for r in regions_after]

mean_before = np.mean(areas_before)
mean_after = np.mean(areas_after)

bars = ax.bar(methods, [mean_before, mean_after], color=colors, alpha=0.7, 
             edgecolor='black', linewidth=2)

for bar, mean in zip(bars, [mean_before, mean_after]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.0f} px²',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('평균 입자 면적 (픽셀²)', fontsize=12, fontweight='bold')
ax.set_title('(b) 평균 크기 비교', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(mean_before, mean_after) * 1.2)
ax.grid(axis='y', alpha=0.3)

size_change = ((mean_after - mean_before) / mean_before) * 100
ax.text(0.5, 0.95, f'크기 변화: {size_change:+.1f}%\n(응집체 분리 효과)',
       ha='center', va='top', transform=ax.transAxes,
       fontsize=11, fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# (c) 면적 분포 히스토그램
ax = axes[2]
bins = np.linspace(0, max(max(areas_before), max(areas_after)), 15)

ax.hist(areas_before, bins=bins, alpha=0.6, label='Before', color='red', edgecolor='black')
ax.hist(areas_after, bins=bins, alpha=0.6, label='After', color='green', edgecolor='black')

ax.axvline(mean_before, color='red', linestyle='--', linewidth=2, 
          label=f'Before 평균: {mean_before:.0f}')
ax.axvline(mean_after, color='green', linestyle='--', linewidth=2,
          label=f'After 평균: {mean_after:.0f}')

ax.set_xlabel('입자 면적 (픽셀²)', fontsize=12, fontweight='bold')
ax.set_ylabel('개수', fontsize=12, fontweight='bold')
ax.set_title('(c) 크기 분포 비교', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig3.suptitle('Watershed 분할 효과 정량 분석', fontsize=16, fontweight='bold')
plt.tight_layout()

filename3 = 'Watershed_statistics.png'
plt.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename3}")

print("\n" + "="*70)
print("✅ 모든 Watershed 설명 이미지 생성 완료!")
print("="*70)
print(f"\n📁 생성된 파일:")
print(f"  1. {filename} - Before/After 비교 (PDF 메인)")
print(f"  2. {filename2} - 단계별 작동 원리")
print(f"  3. {filename3} - 통계 분석")

plt.show()
