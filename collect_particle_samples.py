#!/usr/bin/env python3
"""
전체 이미지에서 실제로 보이는 입자들을 클릭으로 수집하는 도구
사용자가 10개 정도의 입자를 클릭하면, 그 특성을 분석하여 최적의 검출 파라미터를 찾습니다.
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure, exposure
import pandas as pd

print("="*70)
print("🎯 입자 샘플 수집 도구")
print("="*70)
print("사용법: 실제로 보이는 입자들을 50개 정도 클릭해주세요.")
print("       기존 27개에서 추가로 23개를 더 수집합니다.")
print("       클릭 후 콘솔에서 'q'를 입력하면 분석을 시작합니다.")
print("="*70)

# EMD 파일 로드
emd_file = "Camera Ceta 1204 380 kx.emd"
signal = hs.load(emd_file)
data = signal.data

# 픽셀 크기
try:
    x_axis = signal.axes_manager.signal_axes[0]
    pixel_size_x = x_axis.scale
    x_units = x_axis.units
except:
    pixel_size_x = 1.0
    x_units = 'pixels'

print(f"\n📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")
print(f"📊 이미지 크기: {data.shape}")

# 전처리
print("\n🎨 전처리 중...")
# 노이즈 제거를 위해 더 강한 Gaussian smoothing 적용
smoothed = gaussian_filter(data, sigma=2.0)

# 원본 이미지에 살짝만 처리
from skimage import img_as_ubyte

# 노이즈 제거만 (smoothing은 이미 적용됨)
p1, p99 = np.percentile(smoothed, [1, 99])
normalized = (smoothed - p1) / (p99 - p1)
normalized = np.clip(normalized, 0, 1)

# 시각화용: 원본에 gamma만 살짝 조정 (밝기 조절)
display_image = exposure.adjust_gamma(normalized, gamma=1.2)

print("✅ 전처리 완료 (밝기 조정)")

# 기존 샘플 로드 시도
import os
clicked_points = []
particle_data = []
existing_samples_file = "particle_samples.csv"

if os.path.exists(existing_samples_file):
    print(f"\n📂 기존 샘플 로드: {existing_samples_file}")
    existing_df = pd.read_csv(existing_samples_file)
    for _, row in existing_df.iterrows():
        clicked_points.append((int(row['x']), int(row['y'])))
        particle_data.append({
            'id': int(row['id']),
            'x': int(row['x']),
            'y': int(row['y']),
            'center_intensity': row['center_intensity'],
            'local_mean': row['local_mean'],
            'local_std': row['local_std'],
            'local_min': row['local_min'],
            'contrast': row['contrast']
        })
    print(f"✅ 기존 샘플 {len(clicked_points)}개 로드됨")
    print(f"💡 목표: 50개 (추가로 {50 - len(clicked_points)}개 필요)")
else:
    print(f"\n💡 새로운 샘플 수집 시작 (목표: 50개)")

def onclick(event):
    """마우스 클릭 이벤트 핸들러"""
    if event.inaxes is not None and event.button == 1:  # 왼쪽 버튼
        x, y = int(event.xdata), int(event.ydata)
        
        # 이미지 범위 내인지 확인
        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
            clicked_points.append((x, y))
            
            # 클릭 지점 분석
            intensity = normalized[y, x]
            
            # 주변 영역 분석 (15x15)
            window_size = 15
            y_min = max(0, y - window_size)
            y_max = min(data.shape[0], y + window_size)
            x_min = max(0, x - window_size)
            x_max = min(data.shape[1], x + window_size)
            
            local_patch = normalized[y_min:y_max, x_min:x_max]
            local_mean = np.mean(local_patch)
            local_std = np.std(local_patch)
            local_min = np.min(local_patch)
            
            particle_info = {
                'id': len(clicked_points),
                'x': x,
                'y': y,
                'center_intensity': intensity,
                'local_mean': local_mean,
                'local_std': local_std,
                'local_min': local_min,
                'contrast': intensity - local_mean
            }
            particle_data.append(particle_info)
            
            # 클릭 위치에 마커 표시 - 작고 명확하게
            ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
            circle = plt.Circle((x, y), 20, fill=False, color='red', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y-25, f'{len(clicked_points)}', 
                   color='white', fontsize=10, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.9))
            
            print(f"\n입자 #{len(clicked_points)} 클릭됨:")
            print(f"  위치: ({x}, {y})")
            print(f"  중심 강도: {intensity:.4f}")
            print(f"  주변 평균: {local_mean:.4f}")
            print(f"  대비: {intensity - local_mean:+.4f}")
            
            fig.canvas.draw()
            
            if len(clicked_points) >= 50:  # 목표 50개
                print(f"\n✅ {len(clicked_points)}개 수집 완료! 창을 닫고 분석을 시작합니다...")

# 시각화 - 원본스타일
fig = plt.figure(figsize=(18, 18))
ax = plt.subplot(111)
# 원본 그대로, vmin/vmax만 조정
ax.imshow(display_image, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.set_title(f'어두운 점들이 입자입니다 (현재: {len(clicked_points)}/50개)\n좌측 클릭: 입자 선택 | 창 닫기: 저장', 
             fontsize=16, fontweight='bold', pad=15)
ax.axis('off')

# 기존 샘플 표시 (있는 경우) - 작고 투명하게
if len(clicked_points) > 0:
    print(f"\n📍 기존 샘플 {len(clicked_points)}개를 이미지에 표시합니다...")
    for i, (x, y) in enumerate(clicked_points, 1):
        ax.plot(x, y, 'c+', markersize=12, markeredgewidth=1.5, alpha=0.6)
        circle = plt.Circle((x, y), 15, fill=False, color='cyan', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

# 첫 번째 입자 위치 표시 (참고용)
ax.plot(997, 1118, 'go', markersize=12, markeredgewidth=2, fillstyle='none')
ax.text(997, 1118-30, 'Reference', color='lime', fontsize=10, ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# 클릭 이벤트 연결
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()

# 분석
if len(clicked_points) == 0:
    print("\n❌ 클릭된 입자가 없습니다.")
    exit()

print(f"\n" + "="*70)
print(f"📊 수집된 입자 분석 ({len(clicked_points)}개)")
print("="*70)

df = pd.DataFrame(particle_data)

print(f"\n🔍 중심 강도 통계:")
print(f"  평균: {df['center_intensity'].mean():.4f}")
print(f"  표준편차: {df['center_intensity'].std():.4f}")
print(f"  범위: {df['center_intensity'].min():.4f} ~ {df['center_intensity'].max():.4f}")

print(f"\n🔍 대비 통계:")
print(f"  평균: {df['contrast'].mean():.4f}")
print(f"  표준편차: {df['contrast'].std():.4f}")
print(f"  범위: {df['contrast'].min():.4f} ~ {df['contrast'].max():.4f}")

print(f"\n🔍 전체 이미지와 비교:")
global_mean = np.mean(normalized)
global_std = np.std(normalized)
print(f"  전체 평균: {global_mean:.4f}")
print(f"  전체 표준편차: {global_std:.4f}")

# 추천 임계값 계산
# 방법 1: 입자 중심 강도의 최대값
max_particle_intensity = df['center_intensity'].max()
print(f"\n💡 추천 임계값 (방법 1 - 입자 최대 강도):")
print(f"   {max_particle_intensity:.4f}")

# 방법 2: 입자 평균 + 1 표준편차
mean_plus_std = df['center_intensity'].mean() + df['center_intensity'].std()
print(f"\n💡 추천 임계값 (방법 2 - 입자 평균 + 1σ):")
print(f"   {mean_plus_std:.4f}")

# 방법 3: 전체 평균과 입자 평균의 중간
recommended_threshold = (global_mean + df['center_intensity'].mean()) / 2
print(f"\n💡 추천 임계값 (방법 3 - 전체와 입자의 중간):")
print(f"   {recommended_threshold:.4f}")

# CSV 저장
df.to_csv('particle_samples.csv', index=False)
print(f"\n💾 저장:")
print(f"  particle_samples.csv ({len(clicked_points)}개 입자)")

# 주변 영역 크기 추정
print(f"\n🔍 입자 크기 추정을 위한 상세 분석...")
for i, point in enumerate(clicked_points[:5], 1):  # 처음 5개만 상세 분석
    x, y = point
    
    # 30x30 영역에서 어두운 영역 찾기
    window = 30
    y_min = max(0, y - window)
    y_max = min(data.shape[0], y + window)
    x_min = max(0, x - window)
    x_max = min(data.shape[1], x + window)
    
    local_region = normalized[y_min:y_max, x_min:x_max]
    
    # 적응형 임계값
    local_threshold = (normalized[y, x] + np.mean(local_region)) / 2
    binary_local = local_region < local_threshold
    
    # 중심에 가장 가까운 영역 찾기
    labeled = measure.label(binary_local)
    center_y_local = y - y_min
    center_x_local = x - x_min
    
    if 0 <= center_y_local < labeled.shape[0] and 0 <= center_x_local < labeled.shape[1]:
        target_label = labeled[center_y_local, center_x_local]
        
        if target_label > 0:
            props = measure.regionprops(labeled)
            for prop in props:
                if prop.label == target_label:
                    area_nm2 = prop.area * pixel_size_x**2
                    equiv_diameter = np.sqrt(4 * prop.area / np.pi) * pixel_size_x
                    print(f"  입자 #{i}: 면적={area_nm2:.4f} {x_units}², 직경≈{equiv_diameter:.3f} {x_units}")
                    break

print(f"\n" + "="*70)
print("✅ 분석 완료!")
print("="*70)
print("\n💡 다음 단계:")
print("  1. 추천된 임계값 중 하나를 선택하세요")
print("  2. 해당 임계값으로 전체 이미지 검출을 실행하세요")
