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

# 5️⃣ 사용자 입력 받기
print("\n" + "="*50)
print("🔧 스케일바 설정")
print("="*50)

# 스케일바 길이 입력
print(f"📐 현재 픽셀 크기: {pixel_size_x:.3f} {x_units}/pixel")
print(f"📏 이미지 크기: {signal.data.shape[1] * pixel_size_x:.1f} × {signal.data.shape[0] * pixel_size_y:.1f} {x_units}")

while True:
    try:
        scale_length_input = input(f"\n스케일바 길이를 입력하세요 ({x_units}) [기본값: 200]: ").strip()
        if scale_length_input == "":
            scale_length = 200.0
        else:
            scale_length = float(scale_length_input)
        break
    except ValueError:
        print("❌ 올바른 숫자를 입력해주세요.")

# 스케일바 위치 선택
print("\n스케일바 위치를 선택하세요:")
print("1. 좌측 하단 (Left Bottom)")
print("2. 우측 하단 (Right Bottom)")

while True:
    try:
        position_input = input("위치 선택 [1-2, 기본값: 2]: ").strip()
        if position_input == "" or position_input == "2":
            scale_position = "right_bottom"
            print("✅ 우측 하단으로 설정되었습니다.")
            break
        elif position_input == "1":
            scale_position = "left_bottom"
            print("✅ 좌측 하단으로 설정되었습니다.")
            break
        else:
            print("❌ 1 또는 2를 입력해주세요.")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        exit()

print(f"📏 설정된 스케일바: {scale_length} {x_units}, 위치: {scale_position}")

# 6️⃣ 이미지 시각화 및 스케일바 추가
fig, ax = plt.subplots(figsize=(12, 10))

# 이미지 표시
im = ax.imshow(signal.data, cmap='gray', origin='lower')

# 스케일바 추가 함수
def add_scale_bar(ax, image_shape, pixel_size, units='pixels', scale_length=200, position='right_bottom'):
    """이미지에 스케일바 추가"""
    height, width = image_shape
    
    # 실제 픽셀로 변환
    bar_length_pixels = scale_length / pixel_size
    bar_text = f"{scale_length} {units}"
    
    # 스케일바 위치 계산
    margin = 0.05  # 가장자리 여백 (5%)
    bar_height = height * 0.02  # 스케일바 높이
    
    if position == 'left_bottom':
        # 좌측 하단
        x_pos = width * margin
        y_pos = height * margin
        text_x = x_pos + bar_length_pixels/2
    else:  # right_bottom
        # 우측 하단
        x_pos = width - bar_length_pixels - width * margin
        y_pos = height * margin
        text_x = x_pos + bar_length_pixels/2
    
    # 스케일바 그리기
    bar = Rectangle((x_pos, y_pos), bar_length_pixels, bar_height, 
                   facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(bar)
    
    # 스케일바 텍스트
    ax.text(text_x, y_pos + height * 0.04, bar_text,
           ha='center', va='bottom', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    return bar_text

# 스케일바 추가
scale_info = add_scale_bar(ax, signal.data.shape, pixel_size_x, x_units, scale_length, scale_position)
print(f"📏 스케일바 추가 완료: {scale_info}")

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
position_str = "left_bottom" if scale_position == "left_bottom" else "right_bottom"
filename = f'emd_analysis_{position_str}_{scale_length}{x_units}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"💾 설정된 스케일바가 포함된 이미지가 '{filename}'로 저장되었습니다.")

# 기본 파일도 업데이트
plt.savefig('emd_analysis_result.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')

print("✅ EMD 파일 분석이 완료되었습니다!")
print(f"📏 최종 설정: 스케일바 {scale_length} {x_units}, 위치: {position_str}")
