import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

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

# 5️⃣ 이미지 저장 (GUI 없이)
plt.figure(figsize=(10, 8))
plt.imshow(signal.data, cmap='gray')
plt.title('EMD 이미지 분석')
plt.colorbar(label='강도')
plt.xlabel('X 픽셀')
plt.ylabel('Y 픽셀')
plt.savefig('emd_analysis_result.png', dpi=300, bbox_inches='tight')
print("💾 이미지가 'emd_analysis_result.png'로 저장되었습니다.")

print("✅ EMD 파일 분석이 완료되었습니다!")
