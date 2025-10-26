import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 1️⃣ 파일 로드
print("📁 EMD 파일 로딩 중...")
signal = hs.load("Camera_Ceta_0958_27000_x.emd")

# 2️⃣ 데이터 정보 출력
if isinstance(signal, list):
    signal = signal[0]  # 첫 번째 신호 사용
    print(f"📋 신호 리스트에서 첫 번째 신호를 선택했습니다.")

print(f"📊 데이터 형태: {signal.data.shape}")
print(f"📈 데이터 타입: {signal.data.dtype}")
print(f"📏 데이터 크기: {signal.data.size} 픽셀")
print(f"📐 데이터 차원: {len(signal.data.shape)}D")

# 3️⃣ 메타데이터 출력
print(f"📝 메타데이터:")
if hasattr(signal.metadata, 'General'):
    for key, value in signal.metadata.General.as_dictionary().items():
        print(f"   {key}: {value}")

# 4️⃣ 기본 통계 정보
print(f"📊 통계 정보:")
print(f"   최솟값: {np.min(signal.data):.2f}")
print(f"   최댓값: {np.max(signal.data):.2f}")
print(f"   평균값: {np.mean(signal.data):.2f}")
print(f"   표준편차: {np.std(signal.data):.2f}")

# 5️⃣ 이미지 시각화 및 스케일바 추가
fig, ax = plt.subplots(figsize=(12, 10))

# 이미지 표시
im = ax.imshow(signal.data, cmap='gray', origin='lower')

# 픽셀 크기 정보 가져오기 (메타데이터에서)
try:
    # EMD 파일에서 픽셀 크기 정보 추출 시도
    if hasattr(signal.axes_manager, 'signal_axes'):
        x_axis = signal.axes_manager.signal_axes[0]
        y_axis = signal.axes_manager.signal_axes[1]
        pixel_size_x = x_axis.scale if hasattr(x_axis, 'scale') else 1.0
        pixel_size_y = y_axis.scale if hasattr(y_axis, 'scale') else 1.0
        x_units = x_axis.units if hasattr(x_axis, 'units') else 'pixels'
        y_units = y_axis.units if hasattr(y_axis, 'units') else 'pixels'
    else:
        # 기본값 사용
        pixel_size_x = pixel_size_y = 1.0
        x_units = y_units = 'pixels'
        
    print(f"📐 픽셀 크기: X={pixel_size_x} {x_units}, Y={pixel_size_y} {y_units}")
    
except Exception as e:
    print(f"⚠️  픽셀 크기 정보를 가져올 수 없습니다: {e}")
    pixel_size_x = pixel_size_y = 1.0
    x_units = y_units = 'pixels'

# 스케일바 추가
def add_scale_bar(ax, image_shape, pixel_size, units='pixels', bar_length_ratio=0.2):
    """이미지에 스케일바 추가"""
    height, width = image_shape
    
    # 스케일바 길이 계산 (이미지 너비의 20%)
    bar_length_pixels = int(width * bar_length_ratio)
    bar_length_real = bar_length_pixels * pixel_size
    
    # 적절한 스케일바 길이로 반올림
    if bar_length_real >= 1000:
        bar_length_real = round(bar_length_real / 1000) * 1000
        bar_text = f"{int(bar_length_real/1000)} k{units}"
    elif bar_length_real >= 100:
        bar_length_real = round(bar_length_real / 100) * 100
        bar_text = f"{int(bar_length_real)} {units}"
    elif bar_length_real >= 10:
        bar_length_real = round(bar_length_real / 10) * 10
        bar_text = f"{int(bar_length_real)} {units}"
    else:
        bar_text = f"{bar_length_real:.1f} {units}"
    
    # 실제 픽셀로 다시 변환
    bar_length_pixels = bar_length_real / pixel_size
    
    # 스케일바 위치 (우하단)
    x_pos = width - bar_length_pixels - width * 0.05
    y_pos = height * 0.05
    
    # 스케일바 그리기
    bar = Rectangle((x_pos, y_pos), bar_length_pixels, height * 0.02, 
                   facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(bar)
    
    # 스케일바 텍스트
    ax.text(x_pos + bar_length_pixels/2, y_pos + height * 0.04, bar_text,
           ha='center', va='bottom', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    return bar_text

# 스케일바 추가
scale_info = add_scale_bar(ax, signal.data.shape, pixel_size_x, x_units)
print(f"📏 스케일바 추가: {scale_info}")

# 축 레이블 및 제목 설정 (영어로)
ax.set_xlabel(f'X Position ({x_units})', fontsize=14)
ax.set_ylabel(f'Y Position ({y_units})', fontsize=14)
ax.set_title('EMD Image Analysis - Camera Ceta', fontsize=16, fontweight='bold')

# 컬러바 추가
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Intensity', fontsize=12)

# 축 눈금 설정
if pixel_size_x != 1.0:
    # 실제 단위로 축 눈금 표시
    x_ticks = np.linspace(0, signal.data.shape[1], 6)  # 6개 눈금
    y_ticks = np.linspace(0, signal.data.shape[0], 6)  # 6개 눈금
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'{tick*pixel_size_x:.1f}' for tick in x_ticks])
    ax.set_yticklabels([f'{tick*pixel_size_y:.1f}' for tick in y_ticks])

# 그리드 추가
ax.grid(True, alpha=0.3)

# 레이아웃 조정
plt.tight_layout()

# 이미지 저장
plt.savefig('emd_analysis_with_scalebar.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("💾 스케일바가 포함된 이미지가 'emd_analysis_with_scalebar.png'로 저장되었습니다.")

# 기존 파일도 업데이트
plt.savefig('emd_analysis_result.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')

print("✅ EMD 파일 분석이 완료되었습니다!")
