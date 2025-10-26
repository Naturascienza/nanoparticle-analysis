"""
시스템 워크플로우 플로우차트 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 시스템 워크플로우 플로우차트 생성")
print("="*70)

# 플로우차트 생성 - 개선 버전
fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(0, 12)
ax.set_ylim(0, 22)
ax.axis('off')

# 색상 정의 - 더 선명하게
colors = {
    'input': '#BBDEFB',      # 파란색 (입력)
    'process': '#FFF59D',    # 노란색 (처리)
    'ai': '#A5D6A7',         # 녹색 (AI)
    'validation': '#FFCC80', # 주황색 (검증)
    'output': '#F48FB1',     # 분홍색 (출력)
}

def draw_box(ax, x, y, width, height, text, color, fontsize=12, bold=True):
    """박스 그리기 - 개선"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.15", 
        edgecolor='black', facecolor=color, linewidth=3
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', 
           fontsize=fontsize, fontweight=weight,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.3))

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->', linewidth=3):
    """화살표 그리기 - 더 굵게"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color='darkblue', linewidth=linewidth,
        mutation_scale=35
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.6, mid_y, label, ha='left', va='center',
               fontsize=10, style='italic', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        edgecolor='black', linewidth=1.5, alpha=0.9))

def draw_sub_box(ax, x, y, width, height, text, fontsize=10):
    """서브 박스 - 더 선명하게"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        edgecolor='darkgray', facecolor='white', linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
           fontweight='bold')

# ===== 제목 =====
ax.text(6, 21, 'AI 기반 TEM 나노입자 자동 분석 시스템', 
       ha='center', va='center', fontsize=20, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.6', facecolor='#4FC3F7', 
                edgecolor='black', linewidth=4))
ax.text(6, 20.2, '전체 워크플로우 (Workflow)', ha='center', va='center', 
       fontsize=15, style='italic', fontweight='bold')

# ===== Step 1: EMD 파일 로딩 =====
y_current = 18.5
draw_box(ax, 6, y_current, 7, 1.2, '① EMD 파일 로딩\n(Input)', colors['input'], fontsize=14)

# 서브 항목
y_current -= 1
draw_sub_box(ax, 6, y_current, 6.5, 0.6, 
            '▸ HyperSpy로 전자현미경 데이터 읽기')

# 화살표
draw_arrow(ax, 6, y_current - 0.35, 6, y_current - 0.9, label='데이터 로드')

# ===== Step 2: 전처리 =====
y_current -= 1.5
draw_box(ax, 6, y_current, 7, 1.2, '② 전처리 (Pre-processing)', 
        colors['process'], fontsize=14)

# 서브 항목들
y_sub = y_current - 1
x_positions = [3, 6, 9]
labels = ['가우시안 필터\n(노이즈 제거)', '백분위 정규화\n(밝기 보정)', 'CLAHE\n(대비 향상)']
icons = ['🔧', '📊', '🎨']

for x_pos, label, icon in zip(x_positions, labels, icons):
    draw_sub_box(ax, x_pos, y_sub, 2.2, 0.7, f'{icon}\n{label}', fontsize=9)
    # 위로 연결선
    ax.plot([x_pos, x_pos], [y_current - 0.6, y_sub + 0.35], 
           'darkblue', linewidth=2.5, linestyle='-')

# 아래로 모이는 화살표
for x_pos in x_positions:
    draw_arrow(ax, x_pos, y_sub - 0.4, 6, y_sub - 1.2, linewidth=2)

# ===== Step 3: AI 검출 =====
y_current = y_sub - 1.7
draw_box(ax, 6, y_current, 7, 1.2, '③ AI 검출 (Detection)', 
        colors['ai'], fontsize=14)

# 서브 항목들
y_sub = y_current - 1
x_positions = [3, 6, 9]
labels = ['LoG 필터\n(입자 후보)', '이진화\n(배경 분리)', 'Watershed\n(응집 분리)']
icons = ['🔍', '⚫⚪', '💧']

for x_pos, label, icon in zip(x_positions, labels, icons):
    draw_sub_box(ax, x_pos, y_sub, 2.2, 0.7, f'{icon}\n{label}', fontsize=9)
    ax.plot([x_pos, x_pos], [y_current - 0.6, y_sub + 0.35], 
           'darkgreen', linewidth=2.5, linestyle='-')

# 아래로 모이는 화살표
for x_pos in x_positions:
    draw_arrow(ax, x_pos, y_sub - 0.4, 6, y_sub - 1.2, linewidth=2)

# ===== Step 4: 신뢰성 검증 =====
y_current = y_sub - 1.7
draw_box(ax, 6, y_current, 7, 1.2, '④ 신뢰성 검증 (Validation)', 
        colors['validation'], fontsize=14)

# 서브 항목들
y_sub = y_current - 1
x_positions = [3, 6, 9]
labels = ['원형도 필터\n(>0.3)', '종횡비 필터\n(<3.0)', '크기 필터\n(0.5-10nm)']
icons = ['⭕', '↔️', '📏']

for x_pos, label, icon in zip(x_positions, labels, icons):
    draw_sub_box(ax, x_pos, y_sub, 2.2, 0.7, f'{icon}\n{label}', fontsize=9)
    ax.plot([x_pos, x_pos], [y_current - 0.6, y_sub + 0.35], 
           'darkorange', linewidth=2.5, linestyle='-')

# 아래로 모이는 화살표
for x_pos in x_positions:
    draw_arrow(ax, x_pos, y_sub - 0.4, 6, y_sub - 1.2, linewidth=2)

# ===== Step 5: 통계 분석 =====
y_current = y_sub - 1.7
draw_box(ax, 6, y_current, 7, 1.2, '⑤ 통계 분석 (Analysis)', 
        colors['output'], fontsize=14)

# 서브 항목들
y_sub = y_current - 1
x_positions = [3, 6, 9]
labels = ['크기 분포\n히스토그램', '평균/표준편차\n계산', 'CSV/PDF\n리포트']
icons = ['📊', '📈', '📄']

for x_pos, label, icon in zip(x_positions, labels, icons):
    draw_sub_box(ax, x_pos, y_sub, 2.2, 0.7, f'{icon}\n{label}', fontsize=9)
    ax.plot([x_pos, x_pos], [y_current - 0.6, y_sub + 0.35], 
           'darkred', linewidth=2.5, linestyle='-')

# ===== 최종 출력 =====
y_current = y_sub - 1.5
draw_arrow(ax, 6, y_sub - 0.4, 6, y_current + 0.6, label='최종 결과')

draw_box(ax, 6, y_current, 7, 1.2, '✅ 최종 결과물 (Output)', 
        '#81C784', fontsize=14)

# 출력 항목들
y_final = y_current - 1.2
outputs = [
    ('📊', '입자 크기 분포 그래프'),
    ('📋', '입자 데이터 CSV 파일'),
    ('📄', '분석 리포트 PDF'),
    ('🎨', '검출 결과 시각화 이미지')
]

for i, (icon, output) in enumerate(outputs):
    y_pos = y_final - i * 0.5
    ax.text(6, y_pos, f'{icon}  {output}', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='darkgray', linewidth=2))

# ===== 범례 (오른쪽 하단) =====
legend_y = 1.5
legend_items = [
    ('입력/출력', colors['input']),
    ('전처리', colors['process']),
    ('AI 검출', colors['ai']),
    ('검증', colors['validation']),
    ('분석/리포트', colors['output']),
]

ax.text(8.5, legend_y + 1, '단계별 색상', ha='center', va='center',
       fontsize=11, fontweight='bold')

for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.3
    # 작은 색상 박스
    box = FancyBboxPatch(
        (7.8, y_pos - 0.1), 0.4, 0.2,
        boxstyle="round,pad=0.02", 
        edgecolor='black', facecolor=color, linewidth=1
    )
    ax.add_patch(box)
    ax.text(8.3, y_pos, label, ha='left', va='center', fontsize=9)

# ===== 통계 정보 (왼쪽 하단) =====
stats_y = 1.5
stats_text = """
⚡ 성능 지표
━━━━━━━━━━━━━━
• 처리 시간: 3-5분/이미지
• 정확도: ±0.4% 오차
• 재현성: 100%
• 검출 범위: 0.5-10 nm
"""

ax.text(2, stats_y, stats_text, ha='left', va='top',
       fontsize=9, family='monospace',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                edgecolor='black', linewidth=2))

# ===== 프로세스 타임라인 (상단) =====
timeline_y = 17.5
ax.text(1, timeline_y, '처리 시간:', ha='left', va='center',
       fontsize=10, fontweight='bold')

timeline_data = [
    ('로딩', 0.5, 'input'),
    ('전처리', 1, 'process'),
    ('AI검출', 2, 'ai'),
    ('검증', 0.8, 'validation'),
    ('분석', 0.7, 'output'),
]

x_start = 2.5
for label, duration, color_key in timeline_data:
    width = duration * 0.8
    box = FancyBboxPatch(
        (x_start, timeline_y - 0.15), width, 0.3,
        boxstyle="round,pad=0.02", 
        edgecolor='black', facecolor=colors[color_key], linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x_start + width/2, timeline_y, label, 
           ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x_start + width/2, timeline_y - 0.35, f'{duration}분', 
           ha='center', va='top', fontsize=7, style='italic')
    x_start += width + 0.1

# 총 시간
total_time = sum([d[1] for d in timeline_data])
ax.text(x_start + 0.3, timeline_y, f'= {total_time}분', 
       ha='left', va='center', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen'))

plt.tight_layout()

# 저장
filename = 'system_workflow_flowchart.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 저장 완료: {filename}")

print("\n" + "="*70)
print("📊 추가: 간소화 버전 (요약용)")
print("="*70)

# 간소화 버전 (요약)
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')

# 제목
ax2.text(5, 11, 'AI 나노입자 분석 시스템 - 간략 워크플로우', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# 5단계를 세로로 배치
steps = [
    ('① EMD 로딩', 'TEM 이미지 읽기', colors['input'], 9.5),
    ('② 전처리', '노이즈 제거 + 대비 향상', colors['process'], 7.5),
    ('③ AI 검출', 'LoG + Watershed', colors['ai'], 5.5),
    ('④ 검증', '형태 + 크기 필터링', colors['validation'], 3.5),
    ('⑤ 분석', '통계 + 리포트', colors['output'], 1.5),
]

for title, subtitle, color, y_pos in steps:
    # 메인 박스
    box = FancyBboxPatch(
        (1.5, y_pos - 0.6), 7, 1.2,
        boxstyle="round,pad=0.1", 
        edgecolor='black', facecolor=color, linewidth=2.5
    )
    ax2.add_patch(box)
    
    ax2.text(5, y_pos + 0.15, title, ha='center', va='center',
            fontsize=13, fontweight='bold')
    ax2.text(5, y_pos - 0.25, subtitle, ha='center', va='center',
            fontsize=10, style='italic', color='gray')
    
    # 화살표 (마지막 제외)
    if y_pos > 2:
        draw_arrow(ax2, 5, y_pos - 0.7, 5, y_pos - 1.3, linewidth=3)

# 시간 표시
time_labels = ['30초', '1분', '2분', '50초', '40초']
for i, (_, _, _, y_pos) in enumerate(steps):
    ax2.text(9, y_pos, time_labels[i], ha='left', va='center',
            fontsize=9, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

# 총 시간
ax2.text(5, 0.5, '총 처리시간: 약 5분/이미지', ha='center', va='center',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                 edgecolor='black', linewidth=2))

plt.tight_layout()

filename2 = 'system_workflow_simple.png'
plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename2}")

print("\n" + "="*70)
print("📊 추가: 데이터 흐름 다이어그램")
print("="*70)

# 데이터 흐름 다이어그램
fig3, ax3 = plt.subplots(figsize=(16, 10))
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 10)
ax3.axis('off')

# 제목
ax3.text(8, 9.5, '데이터 흐름 다이어그램 (Data Flow)', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# 왼쪽: 입력
draw_box(ax3, 2, 7, 2.5, 1.5, 
        'TEM 이미지\n(2048×2048)\n.emd 파일', 
        colors['input'], fontsize=10)

# 화살표
draw_arrow(ax3, 3.3, 7, 4.5, 7, label='원본 데이터')

# 전처리
draw_box(ax3, 6, 7, 2.5, 1.5, 
        '전처리\n정규화\n대비 향상', 
        colors['process'], fontsize=10)

draw_arrow(ax3, 7.3, 7, 8.5, 7, label='처리된 이미지')

# AI 검출
draw_box(ax3, 10, 7, 2.5, 1.5, 
        'AI 검출\nLoG + Watershed\n이진화', 
        colors['ai'], fontsize=10)

draw_arrow(ax3, 11.3, 7, 12.5, 7, label='입자 영역')

# 검증
draw_box(ax3, 14, 7, 2.5, 1.5, 
        '신뢰성 검증\n형태 필터\n크기 필터', 
        colors['validation'], fontsize=10)

# 아래로 화살표
draw_arrow(ax3, 14, 6.2, 14, 4.5)

# 분석
draw_box(ax3, 14, 3.5, 2.5, 1.5, 
        '통계 분석\n크기 측정\n분포 계산', 
        colors['output'], fontsize=10)

# 왼쪽으로 화살표
draw_arrow(ax3, 12.7, 3.5, 11.3, 3.5)

# 결과물들
outputs_data = [
    (10, 3.5, 'CSV\n데이터'),
    (8, 3.5, 'PNG\n그래프'),
    (6, 3.5, 'PDF\n리포트'),
]

for x, y, label in outputs_data:
    draw_box(ax3, x, y, 1.5, 1, label, '#B2DFDB', fontsize=9)
    draw_arrow(ax3, x - 0.8, y, x + 0.8, y)

# 최종 출력
draw_arrow(ax3, 5.2, 3.5, 3.8, 3.5)
draw_box(ax3, 2, 3.5, 2.5, 1.5, 
        '최종 결과물\n자동 저장\n사용자 확인', 
        '#81C784', fontsize=10)

# 메타데이터 흐름 (점선)
ax3.plot([2, 2, 14, 14], [6.2, 2, 2, 2.7], 'k--', linewidth=1.5, alpha=0.5)
ax3.text(8, 2.2, '메타데이터 (픽셀 크기, 파일명 등)', 
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 처리 통계
stats_box = """
📊 처리 통계
━━━━━━━━━━━━
입력: 4MB (EMD)
중간: 16MB (메모리)
출력: 1MB (결과물)
━━━━━━━━━━━━
메모리: 최대 200MB
CPU: 4코어 활용
"""

ax3.text(2, 0.5, stats_box, ha='center', va='bottom',
        fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 edgecolor='black', linewidth=1.5))

plt.tight_layout()

filename3 = 'data_flow_diagram.png'
plt.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 저장 완료: {filename3}")

print("\n" + "="*70)
print("✅ 모든 워크플로우 다이어그램 생성 완료!")
print("="*70)
print(f"\n📁 생성된 파일:")
print(f"  1. {filename} - 상세 워크플로우 (PDF 메인)")
print(f"  2. {filename2} - 간략 버전 (요약 슬라이드)")
print(f"  3. {filename3} - 데이터 흐름 (부록)")

plt.show()
