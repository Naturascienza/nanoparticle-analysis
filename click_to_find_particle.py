"""
클릭 기반 입자 위치 확인
- 이미지를 표시하고 클릭하면 해당 위치의 좌표와 강도를 출력
- 실제 입자 위치를 정확히 찾을 수 있음
"""

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🖱️  클릭하여 입자 위치 확인")
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

# ROI
center_x, center_y, roi_size = 1024, 1024, 250
half = roi_size // 2
x_start, y_start = center_x - half, center_y - half
x_end, y_end = center_x + half, center_y + half

analysis_data = signal.data[y_start:y_end, x_start:x_end].copy()

print(f"✅ ROI: ({x_start},{y_start}) → ({x_end},{y_end})")
print(f"   크기: {analysis_data.shape} (height × width)")

# 전처리
background = gaussian_filter(analysis_data, sigma=20)
bg_removed = analysis_data - background
bg_removed = np.clip(bg_removed, 0, None)
p1, p99 = np.percentile(bg_removed, [1, 99])
normalized = np.clip((bg_removed - p1) / (p99 - p1), 0, 1)

print("\n" + "="*70)
print("📌 사용 방법:")
print("  1. 이미지에서 입자를 클릭하세요")
print("  2. 좌표와 강도 정보가 출력됩니다")
print("  3. 여러 위치를 클릭하여 비교할 수 있습니다")
print("  4. 창을 닫으면 종료됩니다")
print("="*70)

# 클릭 이벤트 저장
clicks = []

def onclick(event):
    if event.inaxes is None:
        return
    
    x_click = int(event.xdata)
    y_click = int(event.ydata)
    
    # 범위 확인
    if 0 <= y_click < analysis_data.shape[0] and 0 <= x_click < analysis_data.shape[1]:
        # 강도 확인
        intensity_orig = analysis_data[y_click, x_click]
        intensity_norm = normalized[y_click, x_click]
        
        # 주변 평균 (3x3)
        y_min = max(0, y_click - 1)
        y_max = min(analysis_data.shape[0], y_click + 2)
        x_min = max(0, x_click - 1)
        x_max = min(analysis_data.shape[1], x_click + 2)
        
        patch = normalized[y_min:y_max, x_min:x_max]
        local_mean = np.mean(patch)
        
        # 전체 이미지 좌표
        global_x = x_click + x_start
        global_y = y_click + y_start
        
        print(f"\n🖱️  클릭 #{len(clicks)+1}:")
        print(f"  ROI 좌표: ({x_click}, {y_click})")
        print(f"  전체 좌표: ({global_x}, {global_y})")
        print(f"  원본 강도: {intensity_orig:.4f}")
        print(f"  정규화 강도: {intensity_norm:.4f}")
        print(f"  주변 평균: {local_mean:.4f}")
        print(f"  대비: {local_mean - intensity_norm:+.4f} (양수면 주변이 더 밝음, 즉 입자)")
        
        # 입자 판별
        if intensity_norm < local_mean - 0.05:
            print(f"  ✅ 입자일 가능성 높음 (주변보다 어두움)")
        elif intensity_norm > local_mean + 0.05:
            print(f"  ❌ 입자 아님 (주변보다 밝음)")
        else:
            print(f"  ⚠️  배경과 비슷")
        
        clicks.append({
            'roi_x': x_click,
            'roi_y': y_click,
            'global_x': global_x,
            'global_y': global_y,
            'intensity': intensity_norm,
            'local_mean': local_mean
        })
        
        # 클릭 위치 표시
        ax.plot(x_click, y_click, 'r+', markersize=15, markeredgewidth=2)
        ax.text(x_click, y_click - 10, f"#{len(clicks)}", 
               ha='center', va='bottom', fontsize=10, color='yellow', fontweight='bold',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8))
        fig.canvas.draw()

# 시각화
fig, ax = plt.subplots(figsize=(12, 12))
fig.suptitle('입자 위치를 클릭하세요 (좌표 확인)', fontsize=14, fontweight='bold')

vmin, vmax = np.percentile(analysis_data, [2, 98])
ax.imshow(analysis_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.set_title(f'ROI 영역 ({roi_size}×{roi_size} pixels)')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.grid(True, alpha=0.3, linestyle='--')

# 축 눈금 더 세밀하게
ax.set_xticks(np.arange(0, roi_size+1, 25))
ax.set_yticks(np.arange(0, roi_size+1, 25))

# 이벤트 연결
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()

# 종료 후 요약
if clicks:
    print("\n" + "="*70)
    print(f"📊 클릭 위치 요약 ({len(clicks)}개):")
    print("="*70)
    
    for i, click in enumerate(clicks, 1):
        print(f"\n위치 #{i}:")
        print(f"  ROI 좌표: ({click['roi_x']}, {click['roi_y']})")
        print(f"  전체 좌표: ({click['global_x']}, {click['global_y']})")
        print(f"  강도: {click['intensity']:.4f}")
        print(f"  대비: {click['local_mean'] - click['intensity']:+.4f}")
    
    # 가장 어두운 위치 찾기
    darkest_idx = np.argmin([c['intensity'] for c in clicks])
    darkest = clicks[darkest_idx]
    
    print(f"\n🎯 가장 어두운 위치 (입자 가능성 최대):")
    print(f"  위치 #{darkest_idx + 1}")
    print(f"  ROI 좌표: ({darkest['roi_x']}, {darkest['roi_y']})")
    print(f"  전체 좌표: ({darkest['global_x']}, {darkest['global_y']})")

print("\n프로그램 종료")
