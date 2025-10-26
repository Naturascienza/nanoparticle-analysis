"""
신뢰성 분석 시각화 (Figure 4)
파이 차트 + 산점도
"""

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure, filters, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 신뢰성 분석 시각화 (Figure 4)")
print("="*70)

# 1. EMD 파일 로드
print("\n📁 EMD 파일 로딩...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")
if isinstance(signal, list):
    signal = signal[0]

pixel_size = signal.axes_manager[0].scale
x_units = signal.axes_manager[0].units

print(f"✅ 파일 로드 완료")

# 2. 전체 이미지 분석 (더 큰 ROI)
original_data = signal.data.astype(np.float32)
h, w = original_data.shape
center_y, center_x = h // 2, w // 2

# 더 큰 ROI (500x500)
roi_size = 500
y1 = max(0, center_y - roi_size // 2)
y2 = min(h, center_y + roi_size // 2)
x1 = max(0, center_x - roi_size // 2)
x2 = min(w, center_x + roi_size // 2)

roi_data = original_data[y1:y2, x1:x2]

print(f"\n📐 ROI 영역: {roi_data.shape}")

# 3. 전처리
print("\n🎨 이미지 전처리...")
background = gaussian_filter(roi_data, sigma=20)
processed = roi_data - background
processed = np.clip(processed, 0, None)

p2, p98 = np.percentile(processed, [2, 98])
processed = np.clip((processed - p2) / (p98 - p2), 0, 1)

processed = 1.0 - processed
processed = exposure.equalize_adapthist(processed, clip_limit=0.03)
processed = gaussian_filter(processed, sigma=1.0)

# 4. 입자 검출
print("\n🌊 입자 검출...")
threshold = filters.threshold_otsu(processed)
binary = processed > threshold
binary = morphology.remove_small_objects(binary, min_size=10)
binary = morphology.closing(binary, morphology.disk(2))

distance = ndimage.distance_transform_edt(binary)
coords = peak_local_max(distance, footprint=np.ones((7, 7)), 
                        labels=binary, min_distance=5)

mask_markers = np.zeros(distance.shape, dtype=bool)
mask_markers[tuple(coords.T)] = True
markers, _ = ndimage.label(mask_markers)

labels = watershed(-distance, markers, mask=binary)
regions = measure.regionprops(labels, intensity_image=processed)

print(f"   검출된 입자: {len(regions)}개")

# 5. 신뢰성 점수 및 속성 계산
print("\n📊 신뢰성 분석...")

areas_nm2 = []
aspect_ratios = []
circularities = []
diameters_nm = []
reliability_scores = []
reliability_categories = []

for region in regions:
    # 원형도
    if region.perimeter > 0:
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
    else:
        circularity = 0
    
    # 종횡비
    if region.minor_axis_length > 0:
        aspect_ratio = region.major_axis_length / region.minor_axis_length
    else:
        aspect_ratio = 999
    
    # 크기
    area_nm2 = region.area * (pixel_size ** 2)
    diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)
    
    # 크기 점수
    size_score = np.exp(-((diameter_nm - 3.0)**2) / (2 * 1.0**2))
    
    # 강도 대비
    mean_intensity = region.mean_intensity
    background_intensity = np.mean(processed[binary == 0])
    if background_intensity > 0:
        contrast = (mean_intensity - background_intensity) / background_intensity
    else:
        contrast = 0
    
    # 종합 신뢰성 점수
    reliability_score = (
        circularity * 0.4 + 
        size_score * 0.3 + 
        min(contrast, 1.0) * 0.2 +
        (1.0 if aspect_ratio < 3.0 else 0.0) * 0.1
    )
    
    # 카테고리 분류
    if reliability_score >= 0.7:
        category = 'High'
    elif reliability_score >= 0.5:
        category = 'Mid'
    else:
        category = 'Low'
    
    areas_nm2.append(area_nm2)
    aspect_ratios.append(min(aspect_ratio, 4.0))  # 상한 4.0
    circularities.append(circularity)
    diameters_nm.append(diameter_nm)
    reliability_scores.append(reliability_score)
    reliability_categories.append(category)

# 통계
total_particles = len(regions)
high_count = reliability_categories.count('High')
mid_count = reliability_categories.count('Mid')
low_count = reliability_categories.count('Low')

high_pct = (high_count / total_particles) * 100
mid_pct = (mid_count / total_particles) * 100
low_pct = (low_count / total_particles) * 100

print(f"   🟢 High: {high_count}개 ({high_pct:.1f}%)")
print(f"   🟡 Mid:  {mid_count}개 ({mid_pct:.1f}%)")
print(f"   🔴 Low:  {low_count}개 ({low_pct:.1f}%)")

# 6. 시각화
print("\n🎨 신뢰성 분석 그래프 생성...")

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

# ===== 왼쪽: 파이 차트 =====
ax1 = fig.add_subplot(gs[0, 0])

counts = [high_count, mid_count, low_count]
labels = [f'🟢 High\n{high_pct:.1f}%', 
         f'🟡 Mid\n{mid_pct:.1f}%', 
         f'🔴 Low\n{low_pct:.1f}%']
colors = ['#66BB6A', '#FDD835', '#EF5350']
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax1.pie(
    counts, 
    labels=labels, 
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode,
    textprops={'fontsize': 13, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 2}
)

# autopct 텍스트 스타일
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

ax1.set_title('신뢰도별 분포', fontsize=16, fontweight='bold', pad=20)

# 총 개수 표시
ax1.text(0, -1.4, f'총 {total_particles:,}개 입자', 
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                 edgecolor='black', linewidth=2))

# 범례 추가
legend_text = """
🟢 High (≥0.7)
  완벽한 구형, 이상적 크기
  
🟡 Mid (0.5-0.7)
  양호한 품질
  
🔴 Low (<0.5)
  재검토 필요
"""

ax1.text(1.5, 0, legend_text, 
        ha='left', va='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='black', linewidth=1.5))

# ===== 오른쪽: 산점도 =====
ax2 = fig.add_subplot(gs[0, 1])

# 색상 매핑
colors_scatter = []
for cat in reliability_categories:
    if cat == 'High':
        colors_scatter.append('#66BB6A')
    elif cat == 'Mid':
        colors_scatter.append('#FDD835')
    else:
        colors_scatter.append('#EF5350')

# 산점도
scatter = ax2.scatter(
    areas_nm2, 
    aspect_ratios,
    c=colors_scatter,
    s=50,
    alpha=0.6,
    edgecolors='black',
    linewidths=0.5
)

ax2.set_xlabel(f'입자 면적 (nm²)', fontsize=13, fontweight='bold')
ax2.set_ylabel('종횡비 (Aspect Ratio)', fontsize=13, fontweight='bold')
ax2.set_title('면적 vs 종횡비', fontsize=16, fontweight='bold', pad=15)

# 그리드
ax2.grid(True, alpha=0.3, linestyle='--')

# 기준선 표시
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, 
           label='완벽한 원형 (1.0)')
ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='허용 상한 (3.0)')

# 이상적 영역 강조 (작고 둥근)
from matplotlib.patches import Rectangle
ideal_zone = Rectangle((0, 0.8), 0.5, 0.4, 
                       facecolor='lightgreen', alpha=0.2, 
                       edgecolor='green', linewidth=2, linestyle='--')
ax2.add_patch(ideal_zone)
ax2.text(0.25, 1.0, '이상적\n영역', ha='center', va='center',
        fontsize=10, fontweight='bold', color='darkgreen')

ax2.legend(loc='upper right', fontsize=10)

# 축 범위 조정
ax2.set_xlim(0, max(areas_nm2) * 1.1)
ax2.set_ylim(0.8, 4.0)

# 전체 제목
fig.suptitle('입자 신뢰성 분석 (Reliability Analysis)', 
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# 저장
filename = 'reliability_analysis_figure4.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 저장 완료: {filename}")

# 7. 추가: 상세 통계 그래프
print("\n📊 추가: 상세 통계 분석...")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('신뢰성 분석 상세 통계', fontsize=16, fontweight='bold')

# (a) 신뢰도 점수 분포
ax = axes[0, 0]
ax.hist(reliability_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(0.7, color='green', linestyle='--', linewidth=2, label='High 기준')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Mid 기준')
ax.set_xlabel('신뢰도 점수', fontsize=11, fontweight='bold')
ax.set_ylabel('입자 개수', fontsize=11, fontweight='bold')
ax.set_title('(a) 신뢰도 점수 분포', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (b) 원형도 분포
ax = axes[0, 1]
ax.hist(circularities, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='최소 기준')
ax.set_xlabel('원형도 (Circularity)', fontsize=11, fontweight='bold')
ax.set_ylabel('입자 개수', fontsize=11, fontweight='bold')
ax.set_title('(b) 원형도 분포', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (c) 종횡비 분포
ax = axes[1, 0]
ax.hist(aspect_ratios, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='완벽한 원형')
ax.axvline(3.0, color='red', linestyle='--', linewidth=2, label='허용 상한')
ax.set_xlabel('종횡비 (Aspect Ratio)', fontsize=11, fontweight='bold')
ax.set_ylabel('입자 개수', fontsize=11, fontweight='bold')
ax.set_title('(c) 종횡비 분포', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (d) 크기 vs 신뢰도
ax = axes[1, 1]
scatter = ax.scatter(diameters_nm, reliability_scores, 
                    c=colors_scatter, s=30, alpha=0.6,
                    edgecolors='black', linewidths=0.5)
ax.axhline(0.7, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel(f'입자 직경 ({x_units})', fontsize=11, fontweight='bold')
ax.set_ylabel('신뢰도 점수', fontsize=11, fontweight='bold')
ax.set_title('(d) 크기 vs 신뢰도', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()

filename2 = 'reliability_detailed_statistics.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename2}")

# 8. 통계 요약
print("\n" + "="*70)
print("📊 신뢰성 분석 통계 요약")
print("="*70)

print(f"\n전체 입자: {total_particles:,}개")
print(f"\n신뢰도별 분포:")
print(f"  🟢 High (≥0.7): {high_count}개 ({high_pct:.1f}%)")
print(f"  🟡 Mid (0.5-0.7): {mid_count}개 ({mid_pct:.1f}%)")
print(f"  🔴 Low (<0.5): {low_count}개 ({low_pct:.1f}%)")

print(f"\n평균 값:")
print(f"  신뢰도 점수: {np.mean(reliability_scores):.3f}")
print(f"  원형도: {np.mean(circularities):.3f}")
print(f"  종횡비: {np.mean(aspect_ratios):.2f}")
print(f"  입자 크기: {np.mean(diameters_nm):.2f} {x_units}")

print(f"\n품질 기준:")
print(f"  ✅ 고품질 비율: {high_pct:.1f}% (목표: >60%)")
print(f"  ✅ 양호 이상: {high_pct + mid_pct:.1f}% (High + Mid)")

print("\n" + "="*70)
print("✅ 모든 시각화 완료!")
print("="*70)
print(f"\n📁 생성된 파일:")
print(f"  1. {filename} - 메인 그래프 (PDF 삽입용)")
print(f"  2. {filename2} - 상세 통계")

plt.show()
