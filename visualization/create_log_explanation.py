"""
LoG 필터 작동 원리 설명 이미지 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

# 한글 폰트 설정 - Mac용 강화
plt.rcParams['font.family'] = 'AppleGothic'  # Mac 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

print("🎨 LoG 필터 작동 원리 이미지 생성 중...")

# 1. 샘플 이미지 생성 (나노입자 시뮬레이션)
print("  1. 샘플 이미지 생성...")
img_size = 256
image = np.zeros((img_size, img_size))

# 다양한 크기의 입자 추가
particles = [
    (64, 64, 10, 0.8),    # (x, y, radius, intensity)
    (128, 100, 15, 0.9),
    (180, 150, 8, 0.7),
    (80, 180, 12, 0.85),
    (200, 50, 6, 0.6),
]

for x, y, r, intensity in particles:
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= r**2
    image[mask] = intensity

# 노이즈 추가
image += np.random.normal(0, 0.05, image.shape)
image = np.clip(image, 0, 1)

# 백그라운드 추가
background = gaussian_filter(np.random.random((img_size, img_size)) * 0.3, sigma=20)
image += background
image = (image - image.min()) / (image.max() - image.min())

# 2. LoG 필터 적용 단계별
print("  2. LoG 필터 단계별 적용...")

# 2-1. 가우시안 블러 (여러 스케일)
sigma1 = 2
sigma2 = 4
sigma3 = 8

blurred1 = gaussian_filter(image, sigma=sigma1)
blurred2 = gaussian_filter(image, sigma=sigma2)
blurred3 = gaussian_filter(image, sigma=sigma3)

# 2-2. Laplacian 적용
log1 = -gaussian_laplace(image, sigma=sigma1)
log2 = -gaussian_laplace(image, sigma=sigma2)
log3 = -gaussian_laplace(image, sigma=sigma3)

# 2-3. 최종 LoG 결합
log_combined = np.maximum.reduce([log1, log2, log3])

# 3. 시각화
print("  3. 종합 도식 생성...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# ===== Row 1: 원본 → 가우시안 블러 =====
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image, cmap='gray', origin='lower')
ax1.set_title('① 원본 이미지\n(Raw TEM Image)', fontsize=12, fontweight='bold')
ax1.axis('off')

# 화살표
ax_arrow1 = fig.add_subplot(gs[0, 1])
ax_arrow1.text(0.5, 0.5, '→\nGaussian\nBlur', ha='center', va='center',
              fontsize=14, fontweight='bold', color='blue')
ax_arrow1.set_xlim(0, 1)
ax_arrow1.set_ylim(0, 1)
ax_arrow1.axis('off')

ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(blurred2, cmap='gray', origin='lower')
ax2.set_title('② 가우시안 블러\n(Noise Reduction)', fontsize=12, fontweight='bold')
ax2.axis('off')

# 화살표
ax_arrow2 = fig.add_subplot(gs[0, 3])
ax_arrow2.text(0.5, 0.5, '→\nLaplacian\nOperator', ha='center', va='center',
              fontsize=14, fontweight='bold', color='red')
ax_arrow2.set_xlim(0, 1)
ax_arrow2.set_ylim(0, 1)
ax_arrow2.axis('off')

# ===== Row 2: Laplacian → LoG =====
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(log2, cmap='RdBu_r', origin='lower')
ax3.set_title('③ Laplacian\n(Edge Detection)', fontsize=12, fontweight='bold')
ax3.axis('off')

# 다중 스케일
ax4 = fig.add_subplot(gs[1, 1])
ax4.text(0.5, 0.7, '다중 스케일\nMulti-scale LoG', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax4.text(0.5, 0.4, f'σ={sigma1} (작은 입자)\nσ={sigma2} (중간 입자)\nσ={sigma3} (큰 입자)',
        ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 2])
ax5.imshow(log_combined, cmap='hot', origin='lower')
ax5.set_title('④ LoG 결합\n(Combined Response)', fontsize=12, fontweight='bold')
ax5.axis('off')

# 화살표
ax_arrow3 = fig.add_subplot(gs[1, 3])
ax_arrow3.text(0.5, 0.5, '→\nBlob\nDetection', ha='center', va='center',
              fontsize=14, fontweight='bold', color='green')
ax_arrow3.set_xlim(0, 1)
ax_arrow3.set_ylim(0, 1)
ax_arrow3.axis('off')

# ===== Row 3: 최종 검출 결과 =====
ax6 = fig.add_subplot(gs[2, :2])
ax6.imshow(image, cmap='gray', origin='lower')

# 검출된 입자 표시
from scipy import ndimage
binary = log_combined > np.percentile(log_combined, 90)
labels, n_labels = ndimage.label(binary)

for i in range(1, n_labels + 1):
    y_coords, x_coords = np.where(labels == i)
    if len(y_coords) > 5:  # 최소 크기
        y_center = y_coords.mean()
        x_center = x_coords.mean()
        radius = np.sqrt(len(y_coords) / np.pi)
        
        circle = Circle((x_center, y_center), radius, fill=False, 
                       color='lime', linewidth=2, alpha=0.8)
        ax6.add_patch(circle)
        ax6.text(x_center, y_center, str(i), ha='center', va='center',
                color='yellow', fontweight='bold', fontsize=10)

ax6.set_title(f'⑤ 최종 검출 결과 (n={n_labels}개)', 
             fontsize=14, fontweight='bold', color='green')
ax6.axis('off')

# 설명 텍스트
ax7 = fig.add_subplot(gs[2, 2:])
explanation = """
📌 LoG (Laplacian of Gaussian) 작동 원리

1️⃣ 가우시안 블러 (Gaussian Blur)
   • 노이즈 제거 및 스무딩
   • σ (시그마) = 블러 정도 조절

2️⃣ 라플라시안 연산 (Laplacian)
   • 2차 미분으로 가장자리 검출
   • 밝기 변화가 큰 곳 강조

3️⃣ 다중 스케일 검출
   • 작은 σ → 작은 입자 검출
   • 큰 σ → 큰 입자 검출
   • 여러 스케일 결합

4️⃣ Blob 검출
   • 지역 최댓값 = 입자 중심
   • 반지름 = σ × √2

✅ 장점: 
  - 다양한 크기 동시 검출
  - 노이즈에 강건
  - 정확한 중심 위치
"""

ax7.text(0.05, 0.95, explanation, ha='left', va='top',
        fontsize=10, family='AppleGothic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        transform=ax7.transAxes)
ax7.axis('off')

# 전체 제목
fig.suptitle('LoG (Laplacian of Gaussian) 필터 작동 원리', 
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# 저장
filename = 'LoG_principle_diagram.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 저장 완료: {filename}")

print("\n" + "="*70)
print("📊 추가 다이어그램 생성...")
print("="*70)

# 4. 수식 설명 다이어그램
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('LoG 수식 및 원리', fontsize=16, fontweight='bold')

# (a) 가우시안 함수
ax = axes[0, 0]
x = np.linspace(-5, 5, 200)
for sigma in [1, 2, 3]:
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2))
    ax.plot(x, gaussian, label=f'σ={sigma}', linewidth=2)
ax.set_title('(a) 가우시안 함수', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('G(x, σ)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, r'$G(x,\sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{x^2}{2\sigma^2}}$',
       ha='center', va='top', fontsize=12, transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# (b) 라플라시안 함수
ax = axes[0, 1]
for sigma in [1, 2, 3]:
    log = -(1 / (np.pi * sigma**4)) * (1 - x**2 / (2*sigma**2)) * np.exp(-x**2 / (2*sigma**2))
    ax.plot(x, log, label=f'σ={sigma}', linewidth=2)
ax.set_title('(b) LoG 함수 (1D)', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('LoG(x, σ)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.text(0.5, 0.95, r'$\nabla^2 G = -\frac{1}{\pi\sigma^4}(1-\frac{x^2}{2\sigma^2})e^{-\frac{x^2}{2\sigma^2}}$',
       ha='center', va='top', fontsize=11, transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# (c) 2D LoG 커널
ax = axes[1, 0]
size = 31
sigma = 3
x = np.arange(-size//2, size//2+1)
y = np.arange(-size//2, size//2+1)
X, Y = np.meshgrid(x, y)
kernel = -(1 / (np.pi * sigma**4)) * (1 - (X**2 + Y**2) / (2*sigma**2)) * \
         np.exp(-(X**2 + Y**2) / (2*sigma**2))
im = ax.imshow(kernel, cmap='RdBu_r', origin='lower')
ax.set_title(f'(c) 2D LoG 커널 (σ={sigma})', fontweight='bold')
plt.colorbar(im, ax=ax, label='값')
ax.set_xlabel('x')
ax.set_ylabel('y')

# (d) 스케일 공간
ax = axes[1, 1]
sigmas = np.array([1, 2, 3, 4, 5])
particle_sizes = np.linspace(0, 10, 100)
for sigma in sigmas:
    response = np.exp(-((particle_sizes - sigma*np.sqrt(2))**2) / (2*(sigma*0.5)**2))
    ax.plot(particle_sizes, response, label=f'σ={sigma}', linewidth=2)
ax.set_title('(d) 스케일 공간 응답', fontweight='bold')
ax.set_xlabel('입자 크기 (픽셀)')
ax.set_ylabel('검출 응답')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.05, '각 σ는 특정 크기의 입자에 최대 응답',
       ha='center', va='bottom', fontsize=10, transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

# 저장
filename2 = 'LoG_mathematical_explanation.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename2}")

print("\n" + "="*70)
print("✅ 모든 다이어그램 생성 완료!")
print("="*70)
print(f"\n📁 생성된 파일:")
print(f"  1. {filename} - 전체 작동 원리 (PDF 삽입용)")
print(f"  2. {filename2} - 수식 설명 (부록용)")

plt.show()