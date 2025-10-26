import hyperspy.api as hsimport hyperspy.api as hs

import numpy as npimport numpy as np

import matplotlib.pyplot as pltimport matplotlib.pyplot as plt

from skimage import filters, morphologyfrom skimage import filters, morphology

from scipy import ndimagefrom scipy import ndimage



print("🔍 입자 검출 디버깅 시작...")print("🔍 입자 검출 디버깅 시작...")



# 파일 로드# 파일 로드

signal = hs.load("Camera Ceta 1042 27000 x 0001.emd")signal = hs.load("Camera Ceta 1042 27000 x 0001.emd")

if isinstance(signal, list):if isinstance(signal, list):

    signal = signal[0]    signal = signal[0]



data = signal.data.astype(float)data = signal.data.astype(float)

print(f"📊 원본 데이터 범위: {data.min():.2f} ~ {data.max():.2f}")print(f"📊 원본 데이터 범위: {data.min():.2f} ~ {data.max():.2f}")

print(f"📊 원본 데이터 평균: {data.mean():.2f}")print(f"📊 원본 데이터 평균: {data.mean():.2f}")



# ROI 설정 (중심 영역 400x400)# ROI 설정 (중심 영역 400x400)

center_x, center_y = 1024, 1024center_x, center_y = 1024, 1024

roi_size = 400roi_size = 400

roi_x1 = center_x - roi_size // 2roi_x1 = center_x - roi_size // 2

roi_x2 = center_x + roi_size // 2roi_x2 = center_x + roi_size // 2

roi_y1 = center_y - roi_size // 2roi_y1 = center_y - roi_size // 2

roi_y2 = center_y + roi_size // 2roi_y2 = center_y + roi_size // 2



roi_data = data[roi_y1:roi_y2, roi_x1:roi_x2].copy()roi_data = data[roi_y1:roi_y2, roi_x1:roi_x2].copy()

print(f"\n📦 ROI 데이터 범위: {roi_data.min():.2f} ~ {roi_data.max():.2f}")print(f"\n📦 ROI 데이터 범위: {roi_data.min():.2f} ~ {roi_data.max():.2f}")

print(f"📦 ROI 데이터 평균: {roi_data.mean():.2f}")print(f"📦 ROI 데이터 평균: {roi_data.mean():.2f}")



# 전처리 방법들 테스트# 전처리 방법들 테스트

fig, axes = plt.subplots(3, 4, figsize=(20, 15))fig, axes = plt.subplots(3, 4, figsize=(20, 15))

fig.suptitle("입자 검출 과정 디버깅", fontsize=16, fontweight='bold')fig.suptitle("입자 검출 과정 디버깅", fontsize=16, fontweight='bold')



# 1. 원본 이미지# 1. 원본 이미지

ax = axes[0, 0]ax = axes[0, 0]

im = ax.imshow(roi_data, cmap='gray')im = ax.imshow(roi_data, cmap='gray')

ax.set_title(f"1. 원본 ROI\n범위: {roi_data.min():.1f}-{roi_data.max():.1f}")ax.set_title(f"1. 원본 ROI\n범위: {roi_data.min():.1f}-{roi_data.max():.1f}")

plt.colorbar(im, ax=ax)plt.colorbar(im, ax=ax)

ax.axis('off')ax.axis('off')



# 2. 백분위 대비 조정# 2. 백분위 대비 조정

ax = axes[0, 1]ax = axes[0, 1]

vmin, vmax = np.percentile(roi_data, [2, 98])vmin, vmax = np.percentile(roi_data, [2, 98])

contrast_data = np.clip(roi_data, vmin, vmax)contrast_data = np.clip(roi_data, vmin, vmax)

contrast_normalized = (contrast_data - vmin) / (vmax - vmin)im = ax.imshow(contrast_data, cmap='gray')

im = ax.imshow(contrast_normalized, cmap='gray')ax.set_title(f"2. 대비 조정 (2-98%)\n범위: {vmin:.1f}-{vmax:.1f}")

ax.set_title(f"2. 대비 조정 (2-98%)\n정규화: 0-1")plt.colorbar(im, ax=ax)

plt.colorbar(im, ax=ax)ax.axis('off')

ax.axis('off')

# 3. 가우시안 블러

# 3. 가우시안 블러ax = axes[0, 2]

ax = axes[0, 2]blurred = ndimage.gaussian_filter(roi_data, sigma=1.0)

blurred = ndimage.gaussian_filter(roi_data, sigma=1.0)im = ax.imshow(blurred, cmap='gray')

im = ax.imshow(blurred, cmap='gray')ax.set_title(f"3. 가우시안 블러 (σ=1.0)\n평균: {blurred.mean():.1f}")

ax.set_title(f"3. 가우시안 블러 (σ=1.0)\n평균: {blurred.mean():.1f}")plt.colorbar(im, ax=ax)

plt.colorbar(im, ax=ax)ax.axis('off')

ax.axis('off')

# 4. 정규화

# 4. 정규화ax = axes[0, 3]

ax = axes[0, 3]normalized = (roi_data - roi_data.min()) / (roi_data.max() - roi_data.min())

normalized = (roi_data - roi_data.min()) / (roi_data.max() - roi_data.min())im = ax.imshow(normalized, cmap='gray')

im = ax.imshow(normalized, cmap='gray')ax.set_title(f"4. 정규화 [0-1]\n평균: {normalized.mean():.3f}")

ax.set_title(f"4. 정규화 [0-1]\n평균: {normalized.mean():.3f}")plt.colorbar(im, ax=ax)

plt.colorbar(im, ax=ax)ax.axis('off')

ax.axis('off')

# 5-8. 다양한 임계값 방법

# 5-8. 다양한 임계값 방법threshold_methods = [

threshold_methods = [    ('Otsu', filters.threshold_otsu),

    ('Otsu', filters.threshold_otsu),    ('Li', filters.threshold_li),

    ('Li', filters.threshold_li),    ('Yen', filters.threshold_yen),

    ('Yen', filters.threshold_yen),    ('Triangle', filters.threshold_triangle)

    ('Triangle', filters.threshold_triangle)]

]

for idx, (name, method) in enumerate(threshold_methods):

for idx, (name, method) in enumerate(threshold_methods):    ax = axes[1, idx]

    ax = axes[1, idx]    try:

    try:        thresh = method(blurred)

        # 정규화된 데이터에 임계값 적용        binary = blurred > thresh

        thresh = method(contrast_normalized)        im = ax.imshow(binary, cmap='gray')

        binary = contrast_normalized > thresh        ax.set_title(f"5-{idx+1}. {name} 임계값\nthresh={thresh:.2f}\n흰색픽셀: {binary.sum()}")

        im = ax.imshow(binary, cmap='gray')    except Exception as e:

        ax.set_title(f"5-{idx+1}. {name} 임계값\nthresh={thresh:.3f}\n흰색픽셀: {binary.sum()}")        ax.text(0.5, 0.5, f"오류:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)

    except Exception as e:        ax.set_title(f"5-{idx+1}. {name} (실패)")

        ax.text(0.5, 0.5, f"오류:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)    ax.axis('off')

        ax.set_title(f"5-{idx+1}. {name} (실패)")

    ax.axis('off')# 9. 수동 임계값 테스트 (평균 기준)

ax = axes[2, 0]

# 9. 수동 임계값 테스트 (평균 기준)manual_thresh = blurred.mean() + blurred.std()

ax = axes[2, 0]binary_manual = blurred > manual_thresh

manual_thresh = contrast_normalized.mean() + contrast_normalized.std()im = ax.imshow(binary_manual, cmap='gray')

binary_manual = contrast_normalized > manual_threshax.set_title(f"6-1. 수동 임계값\nthresh=평균+1σ={manual_thresh:.2f}\n흰색픽셀: {binary_manual.sum()}")

im = ax.imshow(binary_manual, cmap='gray')ax.axis('off')

ax.set_title(f"6-1. 수동 임계값\nthresh=평균+1σ={manual_thresh:.3f}\n흰색픽셀: {binary_manual.sum()}")

ax.axis('off')# 10. 반전된 임계값 (어두운 입자)

ax = axes[2, 1]

# 10. 반전된 임계값 (어두운 입자)inverted_data = -blurred  # 반전

ax = axes[2, 1]thresh_inv = filters.threshold_otsu(inverted_data)

inverted_data = 1.0 - contrast_normalized  # 반전 (0-1 범위)binary_inv = inverted_data > thresh_inv

thresh_inv = filters.threshold_otsu(inverted_data)im = ax.imshow(binary_inv, cmap='gray')

binary_inv = inverted_data > thresh_invax.set_title(f"6-2. 반전 Otsu (어두운 입자)\nthresh={thresh_inv:.2f}\n흰색픽셀: {binary_inv.sum()}")

im = ax.imshow(binary_inv, cmap='gray')ax.axis('off')

ax.set_title(f"6-2. 반전 Otsu (어두운 입자)\nthresh={thresh_inv:.3f}\n흰색픽셀: {binary_inv.sum()}")

ax.axis('off')# 11. 모폴로지 연산 후

ax = axes[2, 2]

# 11. 모폴로지 연산 후try:

ax = axes[2, 2]    thresh = filters.threshold_otsu(blurred)

try:    binary = blurred > thresh

    thresh = filters.threshold_otsu(contrast_normalized)    cleaned = morphology.remove_small_objects(binary, min_size=5)

    binary = contrast_normalized > thresh    cleaned = morphology.binary_closing(cleaned, morphology.disk(1))

    cleaned = morphology.remove_small_objects(binary, min_size=10)    im = ax.imshow(cleaned, cmap='gray')

    cleaned = morphology.binary_closing(cleaned, morphology.disk(2))    ax.set_title(f"6-3. 모폴로지 정제 후\n객체 수: {cleaned.sum()}")

    im = ax.imshow(cleaned, cmap='gray')except Exception as e:

    ax.set_title(f"6-3. 모폴로지 정제 후\n흰색픽셀: {cleaned.sum()}")    ax.text(0.5, 0.5, f"오류:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)

except Exception as e:    ax.set_title(f"6-3. 모폴로지 (실패)")

    ax.text(0.5, 0.5, f"오류:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)ax.axis('off')

    ax.set_title(f"6-3. 모폴로지 (실패)")

ax.axis('off')# 12. 히스토그램

ax = axes[2, 3]

# 12. 히스토그램ax.hist(roi_data.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')

ax = axes[2, 3]ax.axvline(roi_data.mean(), color='red', linestyle='--', label=f'평균: {roi_data.mean():.1f}')

ax.hist(roi_data.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')ax.axvline(roi_data.mean() + roi_data.std(), color='orange', linestyle='--', label=f'평균+1σ: {roi_data.mean() + roi_data.std():.1f}')

ax.axvline(roi_data.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {roi_data.mean():.1f}')try:

ax.axvline(roi_data.mean() + roi_data.std(), color='orange', linestyle='--', linewidth=2,     thresh_otsu = filters.threshold_otsu(blurred)

          label=f'평균+1σ: {roi_data.mean() + roi_data.std():.1f}')    ax.axvline(thresh_otsu, color='green', linestyle='--', label=f'Otsu: {thresh_otsu:.1f}')

try:except:

    thresh_otsu = filters.threshold_otsu(contrast_normalized)    pass

    # 원본 스케일로 변환ax.set_title("6-4. 강도 분포 히스토그램")

    thresh_otsu_original = thresh_otsu * (vmax - vmin) + vminax.set_xlabel("픽셀 강도")

    ax.axvline(thresh_otsu_original, color='green', linestyle='--', linewidth=2, ax.set_ylabel("빈도")

              label=f'Otsu: {thresh_otsu_original:.1f}')ax.legend(fontsize=8)

except:ax.grid(True, alpha=0.3)

    pass

ax.set_title("6-4. 강도 분포 히스토그램")plt.tight_layout()

ax.set_xlabel("픽셀 강도")plt.savefig("debug_particle_detection.png", dpi=150, bbox_inches='tight')

ax.set_ylabel("빈도")print(f"\n✅ 디버깅 이미지 저장: debug_particle_detection.png")

ax.legend(fontsize=8)plt.close()

ax.grid(True, alpha=0.3)

# 입자가 밝은지 어두운지 판단

plt.tight_layout()print("\n" + "="*60)

plt.savefig("debug_particle_detection.png", dpi=150, bbox_inches='tight')print("🔍 데이터 분석:")

print(f"\n✅ 디버깅 이미지 저장: debug_particle_detection.png")print("="*60)

plt.close()print(f"원본 데이터 평균: {roi_data.mean():.2f}")

print(f"원본 데이터 중앙값: {np.median(roi_data):.2f}")

# 입자가 밝은지 어두운지 판단print(f"원본 데이터 표준편차: {roi_data.std():.2f}")

print("\n" + "="*60)

print("🔍 데이터 분석:")# 중심부와 가장자리 비교

print("="*60)center_region = roi_data[150:250, 150:250]

print(f"원본 데이터 평균: {roi_data.mean():.2f}")edge_region = np.concatenate([

print(f"원본 데이터 중앙값: {np.median(roi_data):.2f}")    roi_data[0:50, :].flatten(),

print(f"원본 데이터 표준편차: {roi_data.std():.2f}")    roi_data[-50:, :].flatten(),

    roi_data[:, 0:50].flatten(),

# 중심부와 가장자리 비교    roi_data[:, -50:].flatten()

center_region = roi_data[150:250, 150:250]])

edge_region = np.concatenate([

    roi_data[0:50, :].flatten(),print(f"\n중심부 평균 강도: {center_region.mean():.2f}")

    roi_data[-50:, :].flatten(),print(f"가장자리 평균 강도: {edge_region.mean():.2f}")

    roi_data[:, 0:50].flatten(),

    roi_data[:, -50:].flatten()if center_region.mean() > edge_region.mean():

])    print("\n⚠️  입자가 밝게 보임 (현재 설정)")

else:

print(f"\n중심부 평균 강도: {center_region.mean():.2f}")    print("\n⚠️  입자가 어둡게 보임 (이미지 반전 필요!)")

print(f"가장자리 평균 강도: {edge_region.mean():.2f}")

print("\n" + "="*60)
if center_region.mean() > edge_region.mean():
    print("\n⚠️  중심부가 더 밝음 → 입자가 밝게 보임 (현재 설정 유지)")
else:
    print("\n⚠️  중심부가 더 어두움 → 입자가 어둡게 보임 (이미지 반전 필요!)")

# 추가 분석: 히스토그램 분석
print(f"\n히스토그램 분석:")
print(f"5번째 백분위수: {np.percentile(roi_data, 5):.2f}")
print(f"95번째 백분위수: {np.percentile(roi_data, 95):.2f}")
print(f"데이터 집중도 (중앙값 기준):")
median_val = np.median(roi_data)
below_median = np.sum(roi_data < median_val)
above_median = np.sum(roi_data > median_val)
print(f"  중앙값 이하: {below_median} 픽셀 ({below_median/(below_median+above_median)*100:.1f}%)")
print(f"  중앙값 이상: {above_median} 픽셀 ({above_median/(below_median+above_median)*100:.1f}%)")

print("\n" + "="*60)
print("💡 권장 사항:")
print("="*60)

# Otsu 임계값 계산
otsu_thresh = filters.threshold_otsu(contrast_normalized)
pixels_above_otsu = np.sum(contrast_normalized > otsu_thresh)
total_pixels = contrast_normalized.size
ratio = pixels_above_otsu / total_pixels * 100

print(f"Otsu 임계값: {otsu_thresh:.3f} (정규화된 데이터)")
print(f"임계값 이상 픽셀: {pixels_above_otsu} ({ratio:.1f}%)")

if ratio < 1:
    print("\n⚠️  입자 비율이 매우 낮음 (1% 미만)")
    print("   → 임계값이 너무 높을 수 있음")
    print("   → Li 임계값 또는 수동 임계값 시도 권장")
elif ratio > 50:
    print("\n⚠️  입자 비율이 매우 높음 (50% 이상)")
    print("   → 임계값이 너무 낮을 수 있음")
    print("   → 배경이 입자로 잘못 인식되고 있을 가능성")
else:
    print(f"\n✅ 입자 비율이 적절함 ({ratio:.1f}%)")

print("\n" + "="*60)
