import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle
from matplotlib.offsetbox import AnchoredText
from skimage import filters, segmentation, measure, morphology, exposure
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm
import seaborn as sns

# 한글 폰트 설정 
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("📦 입자 분석을 위한 라이브러리 로드 완료!")

# 1️⃣ 파일 로드
print("📁 EMD 파일 로딩 중...")
signal = hs.load("Camera Ceta 1042 27000 x 0001.emd")

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

# 🔬 입자 분석 설정
print("\n" + "="*50)
print("🔬 입자 분석 설정")
print("="*50)

perform_particle_analysis = input("입자 분석을 수행하시겠습니까? [y/N]: ").strip().lower()
if perform_particle_analysis in ['y', 'yes']:
    
    # 이미지 전처리 및 대비 개선
    print("\n🎨 이미지 전처리...")
    print("1. 백분위 기반 (2%-98%)")
    print("2. 히스토그램 균등화")
    print("3. 가우시안 필터 + 백분위")
    
    while True:
        try:
            preprocess_input = input("전처리 방법 선택 [1-3, 기본값: 1]: ").strip()
            if preprocess_input == "" or preprocess_input == "1":
                # 백분위 기반
                vmin, vmax = np.percentile(signal.data, [2, 98])
                processed_data = np.clip(signal.data, vmin, vmax)
                processed_data = (processed_data - vmin) / (vmax - vmin)
                print("✅ 백분위 기반 전처리 완료")
                break
            elif preprocess_input == "2":
                # 히스토그램 균등화
                processed_data = exposure.equalize_hist(signal.data)
                print("✅ 히스토그램 균등화 완료")
                break
            elif preprocess_input == "3":
                # 가우시안 필터 + 백분위
                smoothed = filters.gaussian(signal.data, sigma=1)
                vmin, vmax = np.percentile(smoothed, [5, 95])
                processed_data = np.clip(smoothed, vmin, vmax)
                processed_data = (processed_data - vmin) / (vmax - vmin)
                print("✅ 가우시안 필터 + 백분위 전처리 완료")
                break
            else:
                print("❌ 1-3 사이의 값을 입력해주세요.")
        except KeyboardInterrupt:
            print("\n기본값으로 설정합니다.")
            vmin, vmax = np.percentile(signal.data, [2, 98])
            processed_data = np.clip(signal.data, vmin, vmax)
            processed_data = (processed_data - vmin) / (vmax - vmin)
            break
    
    # 측정 영역 설정
    print("\n📏 측정 영역 설정...")
    print("1. 전체 이미지 분석")
    print("2. 관심 영역(ROI) - 좌표로 지정")
    print("3. 관심 영역(ROI) - 중심점과 크기로 지정")
    
    # ROI 변수 초기화 (첫 진입시에만)
    if 'roi_coords' not in locals():
        roi_coords = None
    
    analysis_data = processed_data.copy()
    
    while True:
        try:
            roi_input = input("측정 범위 선택 [1-3, 기본값: 1]: ").strip()
            if roi_input == "" or roi_input == "1":
                print("✅ 전체 이미지를 분석합니다.")
                roi_coords = None
                break
            elif roi_input == "2":
                print("📐 관심 영역 좌표를 직접 입력하세요 (픽셀 단위):")
                print(f"   이미지 크기: {signal.data.shape[1]} × {signal.data.shape[0]} 픽셀")
                print(f"   좌표계: (0,0)=좌측하단, ({signal.data.shape[1]-1},{signal.data.shape[0]-1})=우측상단")
                
                # ROI 좌표 입력
                while True:
                    try:
                        x_start = int(input("시작 X 좌표 [0]: ") or "0")
                        y_start = int(input("시작 Y 좌표 [0]: ") or "0")
                        x_end = int(input(f"끝 X 좌표 [{signal.data.shape[1]}]: ") or str(signal.data.shape[1]))
                        y_end = int(input(f"끝 Y 좌표 [{signal.data.shape[0]}]: ") or str(signal.data.shape[0]))
                        
                        # 좌표 유효성 검사
                        if (0 <= x_start < x_end <= signal.data.shape[1] and 
                            0 <= y_start < y_end <= signal.data.shape[0]):
                            roi_coords = (x_start, y_start, x_end, y_end)
                            analysis_data = processed_data[y_start:y_end, x_start:x_end]
                            roi_width = x_end - x_start
                            roi_height = y_end - y_start
                            roi_area_nm2 = roi_width * roi_height * (pixel_size_x ** 2)
                            print(f"✅ ROI 설정 완료: ({x_start},{y_start}) → ({x_end},{y_end})")
                            print(f"   ROI 크기: {roi_width} × {roi_height} 픽셀 ({roi_width*pixel_size_x:.1f} × {roi_height*pixel_size_y:.1f} {x_units})")
                            print(f"   ROI 중심: ({x_start + roi_width//2}, {y_start + roi_height//2})")
                            print(f"   ROI 면적: {roi_area_nm2:.1f} {x_units}²")
                            break
                        else:
                            print("❌ 좌표가 유효하지 않습니다. 다시 입력해주세요.")
                    except ValueError:
                        print("❌ 올바른 정수를 입력해주세요.")
                break
            elif roi_input == "3":
                print("📐 중심점과 크기로 ROI를 설정합니다:")
                print(f"   이미지 크기: {signal.data.shape[1]} × {signal.data.shape[0]} 픽셀")
                print(f"   이미지 중심: ({signal.data.shape[1]//2}, {signal.data.shape[0]//2})")
                
                while True:
                    try:
                        center_x = int(input(f"ROI 중심 X 좌표 [{signal.data.shape[1]//2}]: ") or str(signal.data.shape[1]//2))
                        center_y = int(input(f"ROI 중심 Y 좌표 [{signal.data.shape[0]//2}]: ") or str(signal.data.shape[0]//2))
                        roi_size = int(input("ROI 한 변 크기 (픽셀) [800]: ") or "800")
                        
                        # ROI 좌표 계산
                        half_size = roi_size // 2
                        x_start = max(0, center_x - half_size)
                        y_start = max(0, center_y - half_size)
                        x_end = min(signal.data.shape[1], center_x + half_size)
                        y_end = min(signal.data.shape[0], center_y + half_size)
                        
                        # 실제 ROI 크기 재계산 (경계 때문에 달라질 수 있음)
                        actual_width = x_end - x_start
                        actual_height = y_end - y_start
                        
                        roi_coords = (x_start, y_start, x_end, y_end)
                        analysis_data = processed_data[y_start:y_end, x_start:x_end]
                        roi_area_nm2 = actual_width * actual_height * (pixel_size_x ** 2)
                        
                        print(f"✅ ROI 설정 완료: ({x_start},{y_start}) → ({x_end},{y_end})")
                        print(f"   요청 중심: ({center_x}, {center_y})")
                        print(f"   실제 중심: ({x_start + actual_width//2}, {y_start + actual_height//2})")
                        print(f"   ROI 크기: {actual_width} × {actual_height} 픽셀 ({actual_width*pixel_size_x:.1f} × {actual_height*pixel_size_y:.1f} {x_units})")
                        print(f"   ROI 면적: {roi_area_nm2:.1f} {x_units}²")
                        break
                    except ValueError:
                        print("❌ 올바른 정수를 입력해주세요.")
                break
            else:
                print("❌ 1-3 사이의 값을 입력해주세요.")
        except KeyboardInterrupt:
            print("\n전체 이미지로 설정합니다.")
            roi_coords = None
            break
    
    # 입자 크기 범위 필터링 설정
    print("\n📊 입자 크기 범위 필터링...")
    print("분석할 입자 크기 범위를 설정하세요 (등가직경 기준):")
    
    # 최소/최대 크기 설정
    min_size_filter = None
    max_size_filter = None
    
    while True:
        try:
            size_filter_input = input("크기 필터링을 사용하시겠습니까? [y/N]: ").strip().lower()
            if size_filter_input in ['y', 'yes']:
                try:
                    min_size_input = input(f"최소 입자 크기 ({x_units}) [제한 없음]: ").strip()
                    max_size_input = input(f"최대 입자 크기 ({x_units}) [제한 없음]: ").strip()
                    
                    min_size_filter = float(min_size_input) if min_size_input else None
                    max_size_filter = float(max_size_input) if max_size_input else None
                    
                    if min_size_filter is not None or max_size_filter is not None:
                        filter_text = []
                        if min_size_filter is not None:
                            filter_text.append(f"최소: {min_size_filter} {x_units}")
                        if max_size_filter is not None:
                            filter_text.append(f"최대: {max_size_filter} {x_units}")
                        print(f"✅ 크기 필터링 설정: {', '.join(filter_text)}")
                    else:
                        print("✅ 크기 제한 없이 분석합니다.")
                    break
                except ValueError:
                    print("❌ 올바른 숫자를 입력해주세요.")
            else:
                print("✅ 크기 필터링을 사용하지 않습니다.")
                break
        except KeyboardInterrupt:
            print("\n크기 필터링을 사용하지 않습니다.")
            break
    
    # 신뢰성 필터링 설정 (작은 입자의 신뢰성 향상)
    print("\n🔍 입자 신뢰성 필터링...")
    print("작은 입자(1-3nm)의 측정 신뢰성을 높이기 위한 고급 필터를 설정합니다:")
    print("노이즈와 구별되는 실제 입자만 선별하여 분석합니다.")
    
    # 신뢰성 필터 매개변수
    reliability_filter = False
    min_circularity = 0.3  # 원형도 최소값
    max_aspect_ratio = 3.0  # 최대 종횡비
    min_area_perimeter_ratio = 0.1  # 면적/둘레² 비율 최소값
    min_intensity_contrast = 1.2  # 최소 강도 대비
    reliability_score_threshold = 0.6  # 신뢰성 점수 임계값
    
    while True:
        try:
            reliability_input = input("신뢰성 필터링을 사용하시겠습니까? [y/N]: ").strip().lower()
            if reliability_input in ['y', 'yes']:
                reliability_filter = True
                print("\n📋 신뢰성 필터 매개변수 설정:")
                
                # 원형도 임계값
                try:
                    circ_input = input(f"최소 원형도 [0-1, 기본값: {min_circularity}]: ").strip()
                    min_circularity = float(circ_input) if circ_input else min_circularity
                    min_circularity = max(0.0, min(1.0, min_circularity))  # 0-1 범위 제한
                except ValueError:
                    print("기본값을 사용합니다.")
                
                # 최대 종횡비
                try:
                    aspect_input = input(f"최대 종횡비 [기본값: {max_aspect_ratio}]: ").strip()
                    max_aspect_ratio = float(aspect_input) if aspect_input else max_aspect_ratio
                    max_aspect_ratio = max(1.0, max_aspect_ratio)  # 최소 1.0
                except ValueError:
                    print("기본값을 사용합니다.")
                
                # 면적/둘레² 비율 (컴팩트함 측정)
                try:
                    compact_input = input(f"최소 컴팩트함 [0-1, 기본값: {min_area_perimeter_ratio}]: ").strip()
                    min_area_perimeter_ratio = float(compact_input) if compact_input else min_area_perimeter_ratio
                    min_area_perimeter_ratio = max(0.0, min(1.0, min_area_perimeter_ratio))
                except ValueError:
                    print("기본값을 사용합니다.")
                
                # 신뢰성 점수 임계값
                try:
                    score_input = input(f"신뢰성 점수 임계값 [0-1, 기본값: {reliability_score_threshold}]: ").strip()
                    reliability_score_threshold = float(score_input) if score_input else reliability_score_threshold
                    reliability_score_threshold = max(0.0, min(1.0, reliability_score_threshold))
                except ValueError:
                    print("기본값을 사용합니다.")
                
                print(f"✅ 신뢰성 필터 설정:")
                print(f"   - 최소 원형도: {min_circularity:.2f}")
                print(f"   - 최대 종횡비: {max_aspect_ratio:.1f}")
                print(f"   - 최소 컴팩트함: {min_area_perimeter_ratio:.2f}")
                print(f"   - 신뢰성 점수 임계값: {reliability_score_threshold:.2f}")
                break
            else:
                print("✅ 신뢰성 필터링을 사용하지 않습니다.")
                break
        except KeyboardInterrupt:
            print("\n신뢰성 필터링을 사용하지 않습니다.")
            break
    
    # 입자 검출 매개변수 설정
    print("\n🎯 입자 검출 매개변수 설정...")
    
    # 최소 입자 크기
    while True:
        try:
            min_size_input = input(f"최소 입자 크기 (픽셀) [기본값: 10]: ").strip()
            min_particle_size = 10 if min_size_input == "" else int(min_size_input)
            break
        except ValueError:
            print("❌ 올바른 정수를 입력해주세요.")
    
    # 임계값 방법
    print("\n임계값 설정 방법:")
    print("1. Otsu 자동 임계값")
    print("2. 사용자 정의 임계값")
    print("3. Li 자동 임계값")
    
    while True:
        try:
            threshold_input = input("임계값 방법 선택 [1-3, 기본값: 1]: ").strip()
            if threshold_input == "" or threshold_input == "1":
                threshold = filters.threshold_otsu(analysis_data)
                print(f"✅ Otsu 임계값: {threshold:.3f}")
                break
            elif threshold_input == "2":
                manual_threshold = float(input("임계값 입력 (0-1): "))
                threshold = manual_threshold
                print(f"✅ 사용자 정의 임계값: {threshold:.3f}")
                break
            elif threshold_input == "3":
                threshold = filters.threshold_li(analysis_data)
                print(f"✅ Li 임계값: {threshold:.3f}")
                break
            else:
                print("❌ 1-3 사이의 값을 입력해주세요.")
        except (ValueError, KeyboardInterrupt):
            print("\n기본값(Otsu)으로 설정합니다.")
            threshold = filters.threshold_otsu(analysis_data)
            break
    
    print(f"🔍 설정된 분석 매개변수:")
    print(f"   최소 입자 크기: {min_particle_size} 픽셀")
    print(f"   임계값: {threshold:.3f}")

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

# ROI 표시 (설정된 경우)
if roi_coords is not None:
    x_start, y_start, x_end, y_end = roi_coords
    roi_rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                        fill=False, edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(roi_rect)
    # ROI 레이블
    ax.text(x_start + (x_end - x_start)/2, y_start - 20, 'ROI', 
           ha='center', va='top', fontsize=12, fontweight='bold',
           color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    print(f"📦 ROI 영역 표시 완료: ({x_start},{y_start}) → ({x_end},{y_end})")

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

# 신뢰성 점수 계산 함수
def calculate_reliability_score(prop, analysis_data, labeled_particles, particle_id, 
                               min_circularity, max_aspect_ratio, min_area_perimeter_ratio):
    """
    입자의 신뢰성 점수를 계산하는 함수
    여러 형태학적 특성과 강도 특성을 종합하여 0-1 사이의 점수 반환
    """
    scores = []
    weights = []
    
    # 1. 원형도 점수 (0-1, 높을수록 좋음)
    circularity = prop.circularity if hasattr(prop, 'circularity') else 0
    if hasattr(prop, 'perimeter') and prop.perimeter > 0:
        circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
    circularity_score = min(1.0, circularity / min_circularity)
    scores.append(circularity_score)
    weights.append(0.25)
    
    # 2. 종횡비 점수 (역수, 1에 가까울수록 좋음)
    if hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
        else:
            aspect_ratio = max_aspect_ratio
    else:
        aspect_ratio = 1.0
    
    aspect_score = min(1.0, max_aspect_ratio / max(1.0, aspect_ratio))
    scores.append(aspect_score)
    weights.append(0.2)
    
    # 3. 컴팩트함 점수 (면적/둘레²)
    if hasattr(prop, 'perimeter') and prop.perimeter > 0:
        compactness = prop.area / (prop.perimeter ** 2)
        compactness_score = min(1.0, compactness / min_area_perimeter_ratio)
    else:
        compactness_score = 0.5
    scores.append(compactness_score)
    weights.append(0.2)
    
    # 4. 면적 일관성 점수 (실제 면적 vs 경계상자 면적)
    if hasattr(prop, 'bbox'):
        bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
        if bbox_area > 0:
            area_consistency = prop.area / bbox_area
        else:
            area_consistency = 0
    else:
        area_consistency = 0.5
    scores.append(area_consistency)
    weights.append(0.15)
    
    # 5. 강도 대비 점수 (입자 내부 vs 주변 강도)
    try:
        # 입자 마스크 생성
        particle_mask = (labeled_particles == (particle_id + 1))
        
        # 입자 내부 평균 강도
        particle_intensity = np.mean(analysis_data[particle_mask])
        
        # 입자 주변 영역의 강도 (확장된 마스크에서 원본 제외)
        from scipy.ndimage import binary_dilation
        expanded_mask = binary_dilation(particle_mask, iterations=3)
        background_mask = expanded_mask & (~particle_mask)
        
        if np.sum(background_mask) > 0:
            background_intensity = np.mean(analysis_data[background_mask])
            if background_intensity > 0:
                contrast_ratio = particle_intensity / background_intensity
                # 대비가 높을수록 좋음 (1.2 이상이 이상적)
                contrast_score = min(1.0, max(0.0, (contrast_ratio - 1.0) / 0.5))
            else:
                contrast_score = 0.5
        else:
            contrast_score = 0.5
    except:
        contrast_score = 0.5
    
    scores.append(contrast_score)
    weights.append(0.2)
    
    # 가중 평균 계산
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    return weighted_score, {
        'circularity_score': circularity_score,
        'aspect_score': aspect_score,
        'compactness_score': compactness_score,
        'area_consistency_score': area_consistency,
        'contrast_score': contrast_score,
        'overall_score': weighted_score
    }

# 🔬 입자 분석 수행 (만약 선택했다면)
if perform_particle_analysis in ['y', 'yes']:
    print("\n" + "="*50)
    print("🔬 입자 분석 수행 중...")
    print("="*50)
    
    # 분석 영역 정보 출력
    if roi_coords is not None:
        x_start, y_start, x_end, y_end = roi_coords
        print(f"📦 분석 영역: ROI ({x_start},{y_start}) → ({x_end},{y_end})")
        print(f"   크기: {x_end-x_start} × {y_end-y_start} 픽셀")
    else:
        print(f"📦 분석 영역: 전체 이미지 ({signal.data.shape[1]} × {signal.data.shape[0]} 픽셀)")
    
    # 이진화 (분석 영역에 대해)
    binary_mask = analysis_data > threshold
    
    # 모폴로지 연산으로 노이즈 제거
    binary_cleaned = morphology.remove_small_objects(binary_mask, min_size=min_particle_size)
    binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_particle_size//2)
    
    # 입자 라벨링
    labeled_particles = measure.label(binary_cleaned)
    n_particles = labeled_particles.max()
    
    print(f"🔍 검출된 입자 수: {n_particles}개")
    
    if n_particles > 0:
        # 입자 속성 측정
        particle_props = measure.regionprops(labeled_particles)
        
        # 입자 데이터 수집
        particle_data = []
        
        for i, prop in enumerate(particle_props):
            # 등가 직경 계산 (면적 기준)
            area_pixels = prop.area
            area_real = area_pixels * (pixel_size_x ** 2)  # 실제 면적
            equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
            equiv_diameter_real = equiv_diameter_pixels * pixel_size_x
            
            # 크기 필터링 적용
            if min_size_filter is not None and equiv_diameter_real < min_size_filter:
                continue
            if max_size_filter is not None and equiv_diameter_real > max_size_filter:
                continue
            
            # 장축/단축 길이
            major_axis_pixels = prop.major_axis_length if hasattr(prop, 'major_axis_length') else np.sqrt(area_pixels)
            minor_axis_pixels = prop.minor_axis_length if hasattr(prop, 'minor_axis_length') else np.sqrt(area_pixels)
            major_axis_real = major_axis_pixels * pixel_size_x
            minor_axis_real = minor_axis_pixels * pixel_size_x
            
            # 종횡비 및 원형도
            aspect_ratio = major_axis_pixels / minor_axis_pixels if minor_axis_pixels > 0 else 1
            perimeter_pixels = prop.perimeter if hasattr(prop, 'perimeter') else 2 * np.sqrt(np.pi * area_pixels)
            circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
            
            # 신뢰성 필터링 적용 (선택된 경우)
            reliability_score = 1.0  # 기본값
            reliability_details = {}
            
            if reliability_filter:
                reliability_score, reliability_details = calculate_reliability_score(
                    prop, analysis_data, labeled_particles, i,
                    min_circularity, max_aspect_ratio, min_area_perimeter_ratio
                )
                
                # 신뢰성 점수가 임계값 미만인 경우 제외
                if reliability_score < reliability_score_threshold:
                    continue
            
            # ROI 좌표 보정 (ROI가 설정된 경우)
            # prop.centroid는 (y, x) 순서임에 주의
            centroid_x_local = prop.centroid[1]  # ROI 내 로컬 x 좌표
            centroid_y_local = prop.centroid[0]  # ROI 내 로컬 y 좌표
            
            if roi_coords is not None:
                # 전체 이미지 좌표계로 변환
                centroid_x = centroid_x_local + roi_coords[0]  # x_start 추가
                centroid_y = centroid_y_local + roi_coords[1]  # y_start 추가
            else:
                centroid_x = centroid_x_local
                centroid_y = centroid_y_local
            
            # 입자 데이터 저장
            particle_info = {
                'Particle_ID': len(particle_data) + 1,  # 필터링된 순서로 ID 재지정
                'Area_pixels': area_pixels,
                f'Area_{x_units}2': area_real,
                'Equiv_Diameter_pixels': equiv_diameter_pixels,
                f'Equiv_Diameter_{x_units}': equiv_diameter_real,
                f'Major_Axis_{x_units}': major_axis_real,
                f'Minor_Axis_{x_units}': minor_axis_real,
                'Aspect_Ratio': aspect_ratio,
                'Circularity': circularity,
                'Centroid_X': centroid_x,
                'Centroid_Y': centroid_y,
                'Reliability_Score': reliability_score
            }
            
            # 신뢰성 세부 점수도 추가 (신뢰성 필터 사용시)
            if reliability_filter:
                particle_info.update({
                    'Circularity_Score': reliability_details['circularity_score'],
                    'Aspect_Score': reliability_details['aspect_score'],
                    'Compactness_Score': reliability_details['compactness_score'],
                    'Area_Consistency_Score': reliability_details['area_consistency_score'],
                    'Contrast_Score': reliability_details['contrast_score']
                })
            
            particle_data.append(particle_info)
        
        # 필터링 결과 출력
        total_detected = n_particles
        filtered_count = len(particle_data)
        
        print(f"\n📊 필터링 결과:")
        print(f"   전체 검출된 입자: {total_detected}개")
        print(f"   필터링 후 입자: {filtered_count}개")
        if filtered_count < total_detected:
            removed_count = total_detected - filtered_count
            print(f"   제거된 입자: {removed_count}개 ({removed_count/total_detected*100:.1f}%)")
            
        # 적용된 필터 요약
        applied_filters = []
        if min_size_filter is not None or max_size_filter is not None:
            applied_filters.append("크기 필터")
        if reliability_filter:
            applied_filters.append("신뢰성 필터")
        
        if applied_filters:
            print(f"   적용된 필터: {', '.join(applied_filters)}")
            
        # 신뢰성 필터 사용시 평균 신뢰성 점수 출력
        if reliability_filter and len(particle_data) > 0:
            avg_reliability = np.mean([p['Reliability_Score'] for p in particle_data])
            print(f"   평균 신뢰성 점수: {avg_reliability:.3f} (임계값: {reliability_score_threshold:.3f})")
        
        if len(particle_data) == 0:
            print("❌ 필터링 조건을 만족하는 입자가 없습니다.")
        else:
            # 데이터프레임 생성
            df_particles = pd.DataFrame(particle_data)
            
            # 통계 요약
            size_stats = df_particles[f'Equiv_Diameter_{x_units}'].describe()
            
            print(f"\n📊 입자 크기 통계 (등가직경, {x_units}):")
            print(f"   총 입자 수: {len(df_particles)}")
            print(f"   평균: {size_stats['mean']:.3f}")
            print(f"   표준편차: {size_stats['std']:.3f}")
            print(f"   최솟값: {size_stats['min']:.3f}")
            print(f"   최댓값: {size_stats['max']:.3f}")
            print(f"   중앙값: {size_stats['50%']:.3f}")
        
        # 결과 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Particle Analysis Results', fontsize=16, fontweight='bold')
        
        # 표시할 데이터 설정 (ROI 여부에 따라)
        if roi_coords is not None:
            # ROI가 설정된 경우 ROI 영역만 표시
            display_data = analysis_data
            x_offset, y_offset = roi_coords[0], roi_coords[1]
            img_title_suffix = f" (ROI: {roi_coords[0]}-{roi_coords[2]}, {roi_coords[1]}-{roi_coords[3]})"
        else:
            # 전체 이미지 표시
            display_data = signal.data
            x_offset, y_offset = 0, 0
            img_title_suffix = " (Full Image)"
        
        # 1. 원본 이미지 + 검출된 입자 표시
        # 대비 개선을 위한 전처리
        vmin, vmax = np.percentile(display_data, [2, 98])
        axes[0,0].imshow(display_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        
        # 필터링된 입자들만 표시 (최대 50개)
        particles_to_show = min(50, len(df_particles))
        if particles_to_show > 0:
            for idx in range(particles_to_show):
                particle = df_particles.iloc[idx]
                
                # ROI 좌표 처리
                if roi_coords is not None:
                    # 전체 이미지 좌표계에서 ROI 영역으로 다시 변환하여 표시
                    x = particle['Centroid_X'] - roi_coords[0]  # ROI 내 로컬 좌표
                    y = particle['Centroid_Y'] - roi_coords[1]  # ROI 내 로컬 좌표
                    
                    # ROI 범위 내에 있는지 확인
                    roi_width = roi_coords[2] - roi_coords[0]
                    roi_height = roi_coords[3] - roi_coords[1]
                    if 0 <= x <= roi_width and 0 <= y <= roi_height:
                        display_x, display_y = x, y
                    else:
                        continue  # ROI 범위 밖이면 표시하지 않음
                else:
                    # 전체 이미지인 경우
                    display_x = particle['Centroid_X']
                    display_y = particle['Centroid_Y']
                
                # 등가 반지름 계산 (픽셀 단위)
                equiv_radius = particle['Equiv_Diameter_pixels'] / 2
                
                # 원 그리기
                circle = Circle((display_x, display_y), equiv_radius, fill=False, color='red', linewidth=1.5, alpha=0.8)
                axes[0,0].add_patch(circle)
                
                # 입자 ID 표시 (처음 10개만)
                if idx < 10:
                    axes[0,0].text(display_x, display_y, str(particle['Particle_ID']), 
                                  ha='center', va='center', fontsize=8, color='yellow', fontweight='bold')
        
        axes[0,0].set_title(f'Original Image + Detected Particles{img_title_suffix}\n({particles_to_show} particles shown)')
        axes[0,0].set_xlabel('X Position (pixels)')
        axes[0,0].set_ylabel('Y Position (pixels)')
        
        # 2. 이진화된 이미지
        axes[0,1].imshow(binary_cleaned, cmap='binary', origin='lower')
        axes[0,1].set_title(f'Binary Mask (Detected Particles){img_title_suffix}')
        axes[0,1].set_xlabel('X Position (pixels)')
        axes[0,1].set_ylabel('Y Position (pixels)')
        
        # 3. 입자 크기 히스토그램 + 가우시안 피팅
        sizes = df_particles[f'Equiv_Diameter_{x_units}']
        
        # 이상치 제거를 위해 99.5 백분위수까지만 사용
        size_percentile_99_5 = np.percentile(sizes, 99.5)
        filtered_sizes = sizes[sizes <= size_percentile_99_5]
        
        n_bins = min(30, max(10, len(filtered_sizes)//10))  # 더 세밀한 bin 설정
        
        # 히스토그램 생성
        counts, bin_edges, patches = axes[1,0].hist(filtered_sizes, bins=n_bins, 
                                                   alpha=0.7, color='blue', edgecolor='black',
                                                   density=True, label='Histogram')
        
        # 가우시안 피팅
        def gaussian(x, amp, mean, std):
            return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        try:
            # bin 중심값 계산
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 초기 매개변수 추정
            amp_init = np.max(counts)
            mean_init = np.mean(filtered_sizes)
            std_init = np.std(filtered_sizes)
            
            # 가우시안 피팅 수행
            popt, pcov = curve_fit(gaussian, bin_centers, counts, 
                                 p0=[amp_init, mean_init, std_init],
                                 maxfev=2000)
            
            amp_fit, mean_fit, std_fit = popt
            
            # 피팅된 가우시안 곡선 그리기
            x_fit = np.linspace(filtered_sizes.min(), filtered_sizes.max(), 300)
            y_fit = gaussian(x_fit, amp_fit, mean_fit, std_fit)
            
            axes[1,0].plot(x_fit, y_fit, 'r-', linewidth=2, 
                          label=f'Gaussian Fit\nMean: {mean_fit:.3f} {x_units}\nStd: {std_fit:.3f} {x_units}')
            
            # R² 계산
            y_pred = gaussian(bin_centers, amp_fit, mean_fit, std_fit)
            ss_res = np.sum((counts - y_pred) ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"📊 가우시안 피팅 결과:")
            print(f"   평균: {mean_fit:.3f} {x_units}")
            print(f"   표준편차: {std_fit:.3f} {x_units}")
            print(f"   R²: {r_squared:.4f}")
            
        except Exception as e:
            print(f"⚠️ 가우시안 피팅 실패: {e}")
            # 피팅 실패시 평균, 중앙값만 표시
            axes[1,0].axvline(size_stats['mean'], color='red', linestyle='--', 
                            label=f'Mean: {size_stats["mean"]:.2f} {x_units}')
            axes[1,0].axvline(size_stats['50%'], color='green', linestyle='--', 
                            label=f'Median: {size_stats["50%"]:.2f} {x_units}')
        
        axes[1,0].set_xlabel(f'Equivalent Diameter ({x_units})', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        axes[1,0].set_title(f'Particle Size Distribution\n({len(filtered_sizes)} particles)', 
                           fontsize=12, fontweight='bold')
        axes[1,0].legend(fontsize=10, framealpha=0.9)
        axes[1,0].grid(True, alpha=0.3)
        
        # 축 범위 설정 (이상치 제외)
        axes[1,0].set_xlim(0, size_percentile_99_5 * 1.1)
        
        # 4. 산점도: 면적 vs 종횡비 (신뢰성 점수 또는 원형도로 색상 구분)
        if reliability_filter:
            # 신뢰성 필터 사용시 신뢰성 점수로 색상 구분
            color_data = df_particles['Reliability_Score']
            color_label = 'Reliability Score'
            cmap = 'RdYlGn'  # 낮은 점수는 빨강, 높은 점수는 초록
        else:
            # 기본은 원형도로 색상 구분
            color_data = df_particles['Circularity']
            color_label = 'Circularity'
            cmap = 'viridis'
        
        scatter = axes[1,1].scatter(df_particles[f'Area_{x_units}2'], df_particles['Aspect_Ratio'], 
                                   c=color_data, cmap=cmap, alpha=0.7, s=30)
        axes[1,1].set_xlabel(f'Area ({x_units}²)', fontweight='bold')
        axes[1,1].set_ylabel('Aspect Ratio', fontweight='bold')
        
        # 제목에 필터 정보 포함
        title_parts = ['Particle Shape Analysis']
        if reliability_filter:
            title_parts.append(f'(Reliability ≥ {reliability_score_threshold:.2f})')
        axes[1,1].set_title('\n'.join(title_parts), fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label(color_label, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # 필터 임계값 선 추가 (신뢰성 필터 사용시)
        if reliability_filter:
            # 종횡비 임계값 선
            if max_aspect_ratio < axes[1,1].get_ylim()[1]:
                axes[1,1].axhline(y=max_aspect_ratio, color='red', linestyle='--', alpha=0.7,
                                 label=f'Max Aspect Ratio: {max_aspect_ratio}')
                axes[1,1].legend()
        
        plt.tight_layout()
        
        # 파일명 접미사 생성 (ROI 및 필터 정보)
        suffix_parts = []
        if roi_coords is not None:
            x_start, y_start, x_end, y_end = roi_coords
            suffix_parts.append(f"ROI_{x_start}-{y_start}_to_{x_end}-{y_end}")
        if min_size_filter is not None or max_size_filter is not None:
            filter_info = []
            if min_size_filter is not None:
                filter_info.append(f"min{min_size_filter:.1f}")
            if max_size_filter is not None:
                filter_info.append(f"max{max_size_filter:.1f}")
            suffix_parts.append(f"filtered_{'_'.join(filter_info)}{x_units}")
        if reliability_filter:
            suffix_parts.append(f"reliable_{reliability_score_threshold:.2f}")
        
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        
        # 결과 저장
        analysis_filename = f'particle_analysis_results_{len(df_particles)}particles{suffix}.png'
        plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
        print(f"💾 입자 분석 결과 이미지 저장: '{analysis_filename}'")
        
        # 데이터 테이블 저장 (CSV)
        csv_filename = f'particle_data_{len(df_particles)}particles{suffix}.csv'
        df_particles.to_csv(csv_filename, index=False)
        print(f"📊 입자 데이터 테이블 저장: '{csv_filename}'")
        
        # 요약 통계 저장 (텍스트)
        summary_filename = f'particle_summary_{len(df_particles)}particles{suffix}.txt'
        with open(summary_filename, 'w') as f:
            f.write("Particle Analysis Summary\n")
            f.write("========================\n\n")
            f.write(f"Total particles detected (before filtering): {total_detected}\n")
            f.write(f"Particles after filtering: {len(df_particles)}\n")
            if roi_coords is not None:
                f.write(f"Analysis region: ROI ({roi_coords[0]},{roi_coords[1]}) → ({roi_coords[2]},{roi_coords[3]})\n")
            else:
                f.write(f"Analysis region: Full image\n")
            if min_size_filter is not None or max_size_filter is not None:
                f.write(f"Size filters applied:\n")
                if min_size_filter is not None:
                    f.write(f"  - Minimum size: {min_size_filter} {x_units}\n")
                if max_size_filter is not None:
                    f.write(f"  - Maximum size: {max_size_filter} {x_units}\n")
            f.write(f"Image size: {signal.data.shape[1]} x {signal.data.shape[0]} pixels\n")
            f.write(f"Pixel size: {pixel_size_x:.3f} {x_units}/pixel\n")
            f.write(f"Analysis parameters:\n")
            f.write(f"  - Minimum particle size: {min_particle_size} pixels\n")
            f.write(f"  - Threshold: {threshold:.3f}\n\n")
            f.write("Size Distribution Statistics (Equivalent Diameter):\n")
            f.write(f"  - Mean: {size_stats['mean']:.3f} {x_units}\n")
            f.write(f"  - Std Dev: {size_stats['std']:.3f} {x_units}\n")
            f.write(f"  - Min: {size_stats['min']:.3f} {x_units}\n")
            f.write(f"  - Max: {size_stats['max']:.3f} {x_units}\n")
            f.write(f"  - Median: {size_stats['50%']:.3f} {x_units}\n")
            f.write(f"  - Q1 (25%): {size_stats['25%']:.3f} {x_units}\n")
            f.write(f"  - Q3 (75%): {size_stats['75%']:.3f} {x_units}\n")
        
        print(f"📋 입자 분석 요약 저장: '{summary_filename}'")
        
        # 전문적인 가우시안 피팅 히스토그램 별도 생성 (첨부 이미지 스타일)
        plt.figure(figsize=(10, 8))
        
        # 이상치 제거된 데이터 사용
        filtered_sizes_clean = filtered_sizes
        
        # 세밀한 히스토그램 생성
        n_bins_detailed = min(50, max(15, len(filtered_sizes_clean)//8))
        counts_clean, bin_edges_clean, _ = plt.hist(filtered_sizes_clean, bins=n_bins_detailed,
                                                   alpha=0.8, color='blue', edgecolor='black',
                                                   density=False, label='Particle Count')
        
        # 가우시안 피팅 (카운트 기준)
        try:
            bin_centers_clean = (bin_edges_clean[:-1] + bin_edges_clean[1:]) / 2
            
            # 더 나은 초기값 추정을 위해 중앙값 주변 데이터만 사용
            median_size = np.median(filtered_sizes_clean)
            q25, q75 = np.percentile(filtered_sizes_clean, [25, 75])
            iqr = q75 - q25
            
            # 이상치 제거 (IQR 기반)
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            normal_sizes = filtered_sizes_clean[(filtered_sizes_clean >= lower_bound) & 
                                              (filtered_sizes_clean <= upper_bound)]
            
            if len(normal_sizes) > 10:  # 충분한 데이터가 있을 때만 피팅
                # 정규분포 데이터로 새 히스토그램
                counts_normal, bin_edges_normal, _ = plt.hist(normal_sizes, bins=n_bins_detailed//2,
                                                            alpha=0.6, color='lightblue', edgecolor='navy',
                                                            density=False, label='Filtered Data')
                bin_centers_normal = (bin_edges_normal[:-1] + bin_edges_normal[1:]) / 2
                
                # 초기 매개변수 (개선된)
                amp_init = np.max(counts_normal)
                mean_init = np.mean(normal_sizes)
                std_init = np.std(normal_sizes)
                
                # 가우시안 함수
                def gaussian_count(x, amp, mean, std):
                    return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
                
                # 피팅 (더 관대한 경계 조건)
                bounds = ([0, mean_init*0.5, std_init*0.1], 
                         [amp_init*2, mean_init*2, std_init*5])
                
                popt_clean, _ = curve_fit(gaussian_count, bin_centers_normal, counts_normal,
                                        p0=[amp_init, mean_init, std_init],
                                        bounds=bounds,
                                        maxfev=5000)
                
                amp_clean, mean_clean, std_clean = popt_clean
            
                # 매끄러운 피팅 곡선 (정규분포 범위 내)
                x_smooth = np.linspace(normal_sizes.min(), normal_sizes.max(), 500)
                y_smooth = gaussian_count(x_smooth, amp_clean, mean_clean, std_clean)
                
                plt.plot(x_smooth, y_smooth, 'k-', linewidth=3, 
                        label=f'Gaussian Fit (Mean Size: {mean_clean:.3f} {x_units})')
                
                # 텍스트 정보 추가 (첨부 이미지 스타일)
                plt.text(0.65, 0.85, f'Particles in {x_units} (Mean Size {mean_clean:.5f} {x_units})',
                        transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8, edgecolor='black'))
                
                # 한국어 텍스트 (첨부 이미지 참조)
                plt.text(0.65, 0.78, f'나노입자들의 {mean_clean:.1f} nm 크기',
                        transform=plt.gca().transAxes, fontsize=10, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                print(f"📊 개선된 가우시안 피팅 성공!")
                print(f"   필터링된 입자 수: {len(normal_sizes)}")
                print(f"   피팅 평균: {mean_clean:.3f} {x_units}")
                print(f"   피팅 표준편차: {std_clean:.3f} {x_units}")
                
            else:
                print(f"⚠️ 정규분포 피팅을 위한 데이터 부족: {len(normal_sizes)}개")
                raise ValueError("Not enough normal data for fitting")
            
        except Exception as e:
            print(f"⚠️ 세부 가우시안 피팅 실패: {e}")
            # 피팅 실패시 기본 통계만 표시
            plt.text(0.65, 0.85, f'Particles in {x_units}\n(Mean Size {np.mean(filtered_sizes_clean):.3f} {x_units})',
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 축 설정 (첨부 이미지 스타일)
        plt.xlabel('Size (nm)', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=14, fontweight='bold')
        plt.title('Particle Size Distribution with Gaussian Fitting', fontsize=16, fontweight='bold')
        
        # 격자 및 범례
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, framealpha=0.9)
        
        # 축 범위 최적화
        plt.xlim(0, np.percentile(filtered_sizes_clean, 99) * 1.1)
        plt.ylim(0, np.max(counts_clean) * 1.1)
        
        # 축 눈금 설정
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 가우시안 피팅 히스토그램 저장
        gaussian_filename = f'particle_gaussian_histogram_{len(df_particles)}particles{suffix}.png'
        plt.savefig(gaussian_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 가우시안 피팅 히스토그램 저장: '{gaussian_filename}'")
        
        # 상위 10개 가장 큰 입자 표시
        largest_particles = df_particles.nlargest(10, f'Equiv_Diameter_{x_units}')
        print(f"\n🏆 가장 큰 입자 Top 10:")
        print(largest_particles[[f'Particle_ID', f'Equiv_Diameter_{x_units}', f'Area_{x_units}2', 'Circularity']].to_string(index=False))
        
        print(f"\n✅ 입자 분석 완료!")
        print(f"📁 생성된 파일:")
        print(f"   - 분석 이미지: {analysis_filename}")
        print(f"   - 데이터 테이블: {csv_filename}")
        print(f"   - 요약 보고서: {summary_filename}")
        
    else:
        print("❌ 검출된 입자가 없습니다. 매개변수를 조정해보세요.")

print("✅ EMD 파일 분석이 완료되었습니다!")
print(f"📏 최종 설정: 스케일바 {scale_length} {x_units}, 위치: {position_str}")
