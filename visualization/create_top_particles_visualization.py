"""
상위 입자 확대 (Top 20) 시각화
신뢰도 점수 기준으로 상위 20개 입자를 확대하여 표시
"""

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure, filters, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 상위 입자 확대 (Top 20) 시각화")
print("="*70)

# 1. EMD 파일 로드
print("\n📁 EMD 파일 로딩...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")
if isinstance(signal, list):
    signal = signal[0]

pixel_size = signal.axes_manager[0].scale
x_units = signal.axes_manager[0].units

print(f"✅ 파일 로드 완료")
print(f"   이미지 크기: {signal.data.shape}")
print(f"   픽셀 크기: {pixel_size:.4f} {x_units}/pixel")

# 2. ROI 선택
original_data = signal.data.astype(np.float32)
h, w = original_data.shape
center_y, center_x = h // 2, w // 2

roi_size = 300
y1 = max(0, center_y - roi_size // 2)
y2 = min(h, center_y + roi_size // 2)
x1 = max(0, center_x - roi_size // 2)
x2 = min(w, center_x + roi_size // 2)

roi_data = original_data[y1:y2, x1:x2]

print(f"\n📐 ROI 영역: ({y1}, {x1}) → ({y2}, {x2})")

# 3. 전처리
print("\n🎨 이미지 전처리...")
background = gaussian_filter(roi_data, sigma=20)
processed = roi_data - background
processed = np.clip(processed, 0, None)

p2, p98 = np.percentile(processed, [2, 98])
processed = np.clip((processed - p2) / (p98 - p2), 0, 1)

processed = 1.0 - processed  # 반전
processed = exposure.equalize_adapthist(processed, clip_limit=0.03)
processed = gaussian_filter(processed, sigma=1.0)

# 4. 입자 검출 (Watershed)
print("\n🌊 입자 검출 (Watershed)...")
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

# 5. 신뢰성 점수 계산
print("\n📊 신뢰성 점수 계산...")

particles_data = []

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
    
    # 크기 (nm)
    area_nm2 = region.area * (pixel_size ** 2)
    diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)
    
    # 크기 점수 (1-5 nm이 이상적)
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
    
    particles_data.append({
        'label': region.label,
        'centroid': region.centroid,
        'bbox': region.bbox,
        'diameter_nm': diameter_nm,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'reliability_score': reliability_score,
        'mean_intensity': mean_intensity
    })

# 신뢰도 순 정렬
particles_data.sort(key=lambda x: x['reliability_score'], reverse=True)

print(f"   평균 신뢰도: {np.mean([p['reliability_score'] for p in particles_data]):.3f}")

# 6. 상위 20개 입자 시각화
print("\n🎨 상위 20개 입자 확대 시각화...")

# Top 20 선택
top_20 = particles_data[:min(20, len(particles_data))]

# 5×4 그리드
fig, axes = plt.subplots(4, 5, figsize=(18, 14))
fig.suptitle('상위 입자 확대 (Top 20)\n신뢰도 순 정렬', 
            fontsize=18, fontweight='bold', y=0.98)

for idx, (ax, particle) in enumerate(zip(axes.flat, top_20)):
    # 입자 영역 추출
    minr, minc, maxr, maxc = particle['bbox']
    
    # 패딩 추가
    pad = 15
    minr_pad = max(0, minr - pad)
    maxr_pad = min(roi_data.shape[0], maxr + pad)
    minc_pad = max(0, minc - pad)
    maxc_pad = min(roi_data.shape[1], maxc + pad)
    
    # 이미지 크롭
    crop_img = roi_data[minr_pad:maxr_pad, minc_pad:maxc_pad]
    
    # 표시
    ax.imshow(crop_img, cmap='gray', origin='lower')
    
    # 입자 경계 표시
    crop_label = labels[minr_pad:maxr_pad, minc_pad:maxc_pad]
    mask = crop_label == particle['label']
    
    # 경계선
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'lime', linewidth=2)
    
    # 정보 텍스트
    info_text = f"#{idx+1}\n"
    info_text += f"크기: {particle['diameter_nm']:.2f} {x_units}\n"
    info_text += f"신뢰도: {particle['reliability_score']:.3f}\n"
    info_text += f"원형도: {particle['circularity']:.3f}"
    
    ax.text(0.05, 0.95, info_text, 
           transform=ax.transAxes, ha='left', va='top',
           fontsize=8, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    # 순위 배지
    rank_color = 'gold' if idx < 3 else 'silver' if idx < 10 else 'lightblue'
    ax.text(0.95, 0.95, f'#{idx+1}',
           transform=ax.transAxes, ha='right', va='top',
           fontsize=12, fontweight='bold', color='black',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor=rank_color, 
                    edgecolor='black', linewidth=2))
    
    ax.axis('off')
    
    # 스케일바 (첫 번째만)
    if idx == 0:
        scale_length_nm = 2  # nm
        scale_length_pixels = scale_length_nm / pixel_size
        scale_bar_y = crop_img.shape[0] - 5
        scale_bar_x_start = 5
        scale_bar_x_end = scale_bar_x_start + scale_length_pixels
        
        ax.plot([scale_bar_x_start, scale_bar_x_end], 
               [scale_bar_y, scale_bar_y], 
               'w-', linewidth=3)
        ax.text((scale_bar_x_start + scale_bar_x_end)/2, scale_bar_y - 3, 
               f'{scale_length_nm} {x_units}', 
               ha='center', va='top', color='white', 
               fontsize=7, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

# 빈 축 숨기기
for idx in range(len(top_20), 20):
    axes.flat[idx].axis('off')

plt.tight_layout()

# 저장
filename = 'top_20_particles_enlarged.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 저장 완료: {filename}")

# 7. 추가: 상위 10개만 크게
print("\n📊 추가: 상위 10개 입자 (더 크게)")

fig2, axes2 = plt.subplots(2, 5, figsize=(18, 8))
fig2.suptitle('Top 10 입자 확대 (신뢰도 기준)', 
             fontsize=16, fontweight='bold')

top_10 = particles_data[:min(10, len(particles_data))]

for idx, (ax, particle) in enumerate(zip(axes2.flat, top_10)):
    minr, minc, maxr, maxc = particle['bbox']
    
    pad = 20
    minr_pad = max(0, minr - pad)
    maxr_pad = min(roi_data.shape[0], maxr + pad)
    minc_pad = max(0, minc - pad)
    maxc_pad = min(roi_data.shape[1], maxc + pad)
    
    crop_img = roi_data[minr_pad:maxr_pad, minc_pad:maxc_pad]
    ax.imshow(crop_img, cmap='gray', origin='lower')
    
    crop_label = labels[minr_pad:maxr_pad, minc_pad:maxc_pad]
    mask = crop_label == particle['label']
    
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'lime', linewidth=3)
    
    # 제목
    title = f"#{idx+1}: {particle['diameter_nm']:.2f} {x_units}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # 정보
    info_text = f"신뢰도: {particle['reliability_score']:.3f}\n"
    info_text += f"원형도: {particle['circularity']:.3f}\n"
    info_text += f"종횡비: {particle['aspect_ratio']:.2f}"
    
    ax.text(0.05, 0.05, info_text,
           transform=ax.transAxes, ha='left', va='bottom',
           fontsize=9, color='white',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.axis('off')

plt.tight_layout()

filename2 = 'top_10_particles_enlarged_large.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename2}")

# 8. 통계 요약
print("\n" + "="*70)
print("📊 Top 20 입자 통계")
print("="*70)

print(f"\n신뢰도 점수:")
print(f"  최고: {top_20[0]['reliability_score']:.3f}")
print(f"  최저: {top_20[-1]['reliability_score']:.3f}")
print(f"  평균: {np.mean([p['reliability_score'] for p in top_20]):.3f}")

print(f"\n입자 크기:")
sizes = [p['diameter_nm'] for p in top_20]
print(f"  최대: {max(sizes):.2f} {x_units}")
print(f"  최소: {min(sizes):.2f} {x_units}")
print(f"  평균: {np.mean(sizes):.2f} {x_units}")

print(f"\n원형도:")
circs = [p['circularity'] for p in top_20]
print(f"  최고: {max(circs):.3f}")
print(f"  최저: {min(circs):.3f}")
print(f"  평균: {np.mean(circs):.3f}")

print("\n" + "="*70)
print("✅ 모든 시각화 완료!")
print("="*70)
print(f"\n📁 생성된 파일:")
print(f"  1. {filename} - Top 20 입자 (5×4 그리드)")
print(f"  2. {filename2} - Top 10 입자 (2×5 그리드, 더 크게)")

plt.show()
