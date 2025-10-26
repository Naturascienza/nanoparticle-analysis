#!/usr/bin/env python3
"""
27개 샘플 입자가 detect_all_particles.py 결과에 포함되는지 검증
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter
from skimage import measure

print("="*70)
print("🎯 샘플 입자 검출 검증 (detect_all_particles.py 설정 사용)")
print("="*70)

# 1. 샘플 좌표 로드
print(f"\n📁 샘플 좌표 로딩...")
df_samples = pd.read_csv('particle_samples.csv')
sample_coords = list(zip(df_samples['x'].astype(int), df_samples['y'].astype(int)))
print(f"✅ {len(sample_coords)}개 샘플 좌표 로드 완료")

# 2. 검출된 입자 데이터 로드
print(f"\n📁 검출된 입자 데이터 로딩...")
df_detected = pd.read_csv('all_particles_data.csv')
print(f"✅ {len(df_detected)}개 검출된 입자 데이터 로드 완료")

# 3. EMD 파일 로드 및 전처리 (detect_all_particles.py와 동일)
print(f"\n📁 EMD 파일 로딩...")
emd_file = "Camera Ceta 1204 380 kx.emd"
s = hs.load(emd_file)
data = s.data
pixel_size = s.axes_manager[0].scale
print(f"📐 픽셀 크기: {pixel_size:.4f} nm/pixel")
print(f"📊 이미지 크기: {data.shape}")

# 전처리
print(f"\n🎨 전처리 중 (detect_all_particles.py와 동일)...")
smoothed = gaussian_filter(data, sigma=1.0)
p1, p99 = np.percentile(smoothed, [1, 99])
normalized = np.clip((smoothed - p1) / (p99 - p1), 0, 1)
print(f"✅ 전처리 완료")

# 임계값 및 검출 (detect_all_particles.py와 동일)
print(f"\n📊 임계값 계산 (detect_all_particles.py와 동일):")
sample_mean = 0.2438
global_mean = np.mean(normalized)
threshold_dark = (sample_mean + global_mean) / 2
print(f"  임계값: {threshold_dark:.4f}")

# 검출
binary = normalized < threshold_dark
labeled = measure.label(binary, connectivity=2)
regions = measure.regionprops(labeled, intensity_image=normalized)

# 필터링 (detect_all_particles.py와 동일)
min_area = 30  # 샘플 우선
min_circularity = 0.10  # 샘플 우선

valid_particles = []
for region in regions:
    area_px = region.area
    if area_px < min_area:
        continue
    
    perimeter = region.perimeter
    if perimeter == 0:
        continue
    circularity = (4 * np.pi * area_px) / (perimeter ** 2)
    
    if circularity >= min_circularity:
        valid_particles.append(region)

print(f"✅ 필터링 후: {len(valid_particles)}개 입자")

# 4. 각 샘플이 검출된 입자에 포함되는지 확인
print(f"\n🔍 샘플 좌표 검증 중...")
print(f"{'ID':<6}{'좌표':<20}{'샘플강도':<12}{'검출':<10}{'가장 가까운 입자':<20}")
print("-" * 70)

detected_samples = []
missed_samples = []

for idx, (x, y) in enumerate(sample_coords, 1):
    sample_intensity = normalized[y, x]
    
    # 가장 가까운 검출된 입자 찾기
    min_dist = float('inf')
    nearest_particle = None
    
    for particle in valid_particles:
        cy, cx = particle.centroid
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_particle = particle
    
    # 10 픽셀 이내면 검출된 것으로 간주
    is_detected = min_dist < 10
    
    if is_detected:
        detected_samples.append(idx)
        cy, cx = nearest_particle.centroid
        status = f"✅ ({cx:.1f},{cy:.1f}) d={min_dist:.1f}px"
    else:
        missed_samples.append(idx)
        if nearest_particle:
            cy, cx = nearest_particle.centroid
            status = f"❌ 가장 가까운: ({cx:.1f},{cy:.1f}) d={min_dist:.1f}px"
        else:
            status = "❌ 근처에 검출 없음"
    
    print(f"{idx:<6}({x},{y}){'':<10}{sample_intensity:.4f}{'':<6}{status}")

# 5. 통계
print(f"\n{'='*70}")
print(f"📊 검증 결과:")
print(f"  샘플 총 개수: {len(sample_coords)}개")
print(f"  검출된 샘플: {len(detected_samples)}개")
print(f"  누락된 샘플: {len(missed_samples)}개")
print(f"  검출율: {len(detected_samples)/len(sample_coords)*100:.1f}%")
print(f"  검출된 총 입자: {len(valid_particles)}개")
print(f"  비율: {len(detected_samples)} 샘플 / {len(valid_particles)} 전체 입자")

if missed_samples:
    print(f"\n❌ 누락된 샘플 ID: {missed_samples}")
    print(f"\n💡 제안:")
    print(f"  1. 원형도 필터를 낮추기 (현재: {min_circularity})")
    print(f"  2. 최소 면적을 줄이기 (현재: {min_area} 픽셀)")
    print(f"  3. 임계값을 높이기 (현재: {threshold_dark:.4f})")

# 6. 시각화
print(f"\n📈 시각화 중...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 원본 이미지 + 샘플 좌표
ax1 = axes[0]
ax1.imshow(normalized, cmap='gray', origin='lower')
for idx, (x, y) in enumerate(sample_coords, 1):
    if idx in detected_samples:
        circle = Circle((x, y), 20, color='lime', fill=False, linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y-25, f'{idx}', color='lime', fontsize=8, ha='center', weight='bold')
    else:
        ax1.plot(x, y, 'rx', markersize=15, markeredgewidth=3)
        ax1.text(x, y-25, f'{idx}', color='red', fontsize=8, ha='center', weight='bold')
ax1.set_title(f'샘플 좌표 검증\n녹색 원: 검출됨 ({len(detected_samples)}개) | 빨간 X: 누락 ({len(missed_samples)}개)', 
              fontsize=12)
ax1.axis('off')

# 검출된 입자 + 샘플 위치
ax2 = axes[1]
ax2.imshow(normalized, cmap='gray', origin='lower')
# 검출된 모든 입자 표시
for particle in valid_particles:
    cy, cx = particle.centroid
    circle = Circle((cx, cy), 20, color='cyan', fill=False, linewidth=1, alpha=0.5)
    ax2.add_patch(circle)

# 샘플 위치 표시
for idx, (x, y) in enumerate(sample_coords, 1):
    if idx in detected_samples:
        ax2.plot(x, y, 'g+', markersize=10, markeredgewidth=2)
    else:
        ax2.plot(x, y, 'rx', markersize=15, markeredgewidth=3)

ax2.set_title(f'전체 검출 결과\n청록 원: 검출된 입자 ({len(valid_particles)}개)\n녹색 +: 샘플(검출됨) | 빨간 X: 샘플(누락)', 
              fontsize=12)
ax2.axis('off')

plt.tight_layout()
output_filename = f'sample_verification_{len(detected_samples)}of{len(sample_coords)}_in_{len(valid_particles)}total.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"💾 저장 완료: {output_filename}")

plt.show()

print(f"\n{'='*70}")
print(f"✅ 검증 완료!")
print(f"{'='*70}")
