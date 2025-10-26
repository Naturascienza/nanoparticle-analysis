#!/usr/bin/env python3
"""
샘플 좌표 기반 입자 검출 검증
particle_samples.csv의 27개 클릭 좌표를 사용하여 검출 알고리즘 검증
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from skimage import filters, measure, morphology
from scipy import ndimage

print("="*70)
print("🎯 샘플 좌표 기반 입자 검출 검증")
print("="*70)

# 1. 샘플 좌표 로드
print("\n📁 샘플 좌표 로딩...")
try:
    df_samples = pd.read_csv('particle_samples.csv')
    print(f"✅ {len(df_samples)}개 샘플 좌표 로드 완료")
    print(f"   좌표 범위: X({df_samples['x'].min()}-{df_samples['x'].max()}), "
          f"Y({df_samples['y'].min()}-{df_samples['y'].max()})")
except FileNotFoundError:
    print("❌ particle_samples.csv 파일을 찾을 수 없습니다.")
    exit()

# 2. EMD 파일 로드
print("\n📁 EMD 파일 로딩...")
signal = hs.load("Camera Ceta 1204 380 kx.emd")
data = signal.data

# 픽셀 크기
try:
    x_axis = signal.axes_manager.signal_axes[0]
    pixel_size_x = x_axis.scale
    x_units = x_axis.units
except:
    pixel_size_x = 0.0335
    x_units = 'nm'

print(f"📐 픽셀 크기: {pixel_size_x:.4f} {x_units}/pixel")
print(f"📊 이미지 크기: {data.shape}")

# 3. 전처리 (가우시안 필터 + 백분위)
print("\n🎨 전처리 중...")
from scipy.ndimage import gaussian_filter

smoothed = gaussian_filter(data, sigma=2.0)  # 노이즈 감소
vmin, vmax = np.percentile(smoothed, [1, 99])
normalized = (smoothed - vmin) / (vmax - vmin)
normalized = np.clip(normalized, 0, 1)
print("✅ 전처리 완료 (Gaussian σ=2.0 + Percentile normalization)")

# 4. 임계값 설정 방법 선택
print("\n🎯 임계값 방법 선택:")
print("1. Otsu 자동 임계값")
print("2. 수동 임계값 입력")
print("3. 샘플 기반 adaptive threshold")

threshold_method = input("방법 선택 [1-3, 기본값: 3]: ").strip()

if threshold_method == "1":
    # 이미지 반전 후 Otsu
    inverted = 1.0 - normalized
    threshold = filters.threshold_otsu(inverted)
    print(f"✅ Otsu 임계값: {threshold:.4f}")
elif threshold_method == "2":
    threshold = float(input("임계값 입력 (0-1): "))
    print(f"✅ 수동 임계값: {threshold:.4f}")
else:
    # 샘플 기반 (기본값)
    # 샘플의 평균 강도를 참조
    sample_mean_intensity = df_samples['center_intensity'].mean()
    global_mean = np.mean(normalized)
    threshold = (sample_mean_intensity + global_mean) / 2
    print(f"✅ 샘플 기반 임계값: {threshold:.4f}")
    print(f"   (샘플 평균: {sample_mean_intensity:.4f}, 전체 평균: {global_mean:.4f})")

# 5. 이진화 및 입자 검출
print("\n🔍 입자 검출 중...")

# 어두운 입자 검출 (이미지 반전)
inverted_data = 1.0 - normalized
binary_mask = inverted_data > threshold

# 노이즈 제거
min_particle_size = 50  # 픽셀²
binary_cleaned = morphology.remove_small_objects(binary_mask, min_size=min_particle_size)
binary_cleaned = morphology.binary_opening(binary_cleaned, morphology.disk(2))

# 라벨링
labeled = measure.label(binary_cleaned)
n_detected = labeled.max()

print(f"✅ 총 {n_detected}개 영역 검출됨")

# 6. 샘플 좌표 검증
print("\n🔍 샘플 좌표 검증...")
print(f"{'ID':<5} {'좌표':<15} {'강도':<10} {'검출':<10} {'라벨':<10}")
print("-" * 60)

detection_results = []
for idx, row in df_samples.iterrows():
    x, y = int(row['x']), int(row['y'])
    intensity = row['center_intensity']
    
    # 해당 좌표에서 라벨 확인
    label_at_pos = labeled[y, x]
    detected = label_at_pos > 0
    
    detection_results.append({
        'id': row['id'],
        'x': x,
        'y': y,
        'intensity': intensity,
        'detected': detected,
        'label': label_at_pos
    })
    
    status = "✅" if detected else "❌"
    print(f"{row['id']:<5} ({x},{y}){'':<5} {intensity:<10.4f} {status:<10} {label_at_pos:<10}")

df_results = pd.DataFrame(detection_results)

# 검출 통계
n_detected_samples = df_results['detected'].sum()
detection_rate = n_detected_samples / len(df_results) * 100

print("\n📊 검출 통계:")
print(f"   샘플 총 개수: {len(df_results)}개")
print(f"   검출된 샘플: {n_detected_samples}개")
print(f"   검출 실패: {len(df_results) - n_detected_samples}개")
print(f"   검출율: {detection_rate:.1f}%")

# 7. 시각화
print("\n📈 시각화 중...")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 원본 + 샘플 위치
vmin_display, vmax_display = np.percentile(data, [2, 98])
axes[0].imshow(data, cmap='gray', origin='lower', vmin=vmin_display, vmax=vmax_display)

for idx, row in df_results.iterrows():
    x, y = row['x'], row['y']
    if row['detected']:
        # 검출된 샘플: 녹색 원
        circle = Circle((x, y), 20, fill=False, color='lime', linewidth=3)
        axes[0].add_patch(circle)
        axes[0].text(x, y-30, f"#{row['id']}", color='lime', fontsize=10, 
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    else:
        # 검출 실패: 빨간 X
        axes[0].plot(x, y, 'rx', markersize=15, markeredgewidth=4)
        axes[0].text(x, y-30, f"#{row['id']}", color='red', fontsize=10, 
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

axes[0].set_title(f'샘플 좌표 검증 결과\n검출: {n_detected_samples}/{len(df_results)} ({detection_rate:.1f}%)', 
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('X (pixels)', fontsize=12)
axes[0].set_ylabel('Y (pixels)', fontsize=12)

# 이진화 결과
axes[1].imshow(binary_cleaned, cmap='gray', origin='lower')
for idx, row in df_results.iterrows():
    x, y = row['x'], row['y']
    color = 'lime' if row['detected'] else 'red'
    marker = 'o' if row['detected'] else 'x'
    axes[1].plot(x, y, marker=marker, color=color, markersize=10, markeredgewidth=2)

axes[1].set_title(f'이진화 결과 (Threshold: {threshold:.4f})\n총 {n_detected}개 영역', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('X (pixels)', fontsize=12)
axes[1].set_ylabel('Y (pixels)', fontsize=12)

plt.tight_layout()

# 저장
output_filename = f'validation_result_{n_detected_samples}of{len(df_results)}_detected.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n💾 검증 결과 저장: {output_filename}")

# 검출 실패한 샘플 분석
if n_detected_samples < len(df_results):
    print("\n❌ 검출 실패한 샘플 분석:")
    failed = df_results[~df_results['detected']]
    print(f"   실패한 샘플: {list(failed['id'].values)}")
    print(f"   실패 샘플의 평균 강도: {failed['intensity'].mean():.4f}")
    print(f"   성공 샘플의 평균 강도: {df_results[df_results['detected']]['intensity'].mean():.4f}")
    print(f"   현재 임계값: {threshold:.4f}")

print("\n" + "="*70)
print("✅ 검증 완료!")
print("="*70)

plt.show()
