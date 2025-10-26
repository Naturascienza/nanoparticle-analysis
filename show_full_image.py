#!/usr/bin/env python3
"""
전체 2048x2048 이미지만 보여주기
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("🖼️  전체 이미지 표시")
print("="*70)

# EMD 파일 로드
emd_file = "Camera Ceta 1204 380 kx.emd"
signal = hs.load(emd_file)
data = signal.data

print(f"📊 이미지 크기: {data.shape}")
print(f"✅ 이미지 로드 완료\n")
print("🎯 화면에 전체 이미지가 표시됩니다.")

# 전체 이미지 표시 - 매우 크게
fig, ax = plt.subplots(figsize=(20, 20))
# 더 좁은 범위로 대비 강화 (2%, 98% 대신 5%, 95% 사용)
vmin, vmax = np.percentile(data, [5, 95])
ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.set_title(f'전체 이미지 ({data.shape[0]} × {data.shape[1]} pixels) - 대비 강화', 
             fontsize=18, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ 완료!")
