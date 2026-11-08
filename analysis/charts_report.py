"""보고서·포스터용 구조도와 동작 흐름도."""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm
A = os.path.dirname(os.path.abspath(__file__)); OUT = f'{A}/그림'
fm.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc'); plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False})
INK, MUTED, IN_, MODEL, OUTC, EDGE = '#17322b', '#51635d', '#e8efec', '#d5ead9', '#fbe7d3', '#7f938b'
def box(ax, x, y, w, h, title, lines=(), fc=IN_, ts=11.5, ls=9.5, k=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.012', fc=fc, ec=EDGE, lw=1, mutation_aspect=1))
    n = len(lines); cy = y + h / 2
    ax.text(x + w / 2, cy + (.016 * k * n if n else 0), title, ha='center', va='center', fontsize=ts, color=INK, fontweight='bold')
    for i, l in enumerate(lines): ax.text(x + w / 2, cy + .016 * k * n - .037 * k * (i + 1), l, ha='center', va='center', fontsize=ls, color=MUTED)
def arrow(ax, a, b, **k): ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3, shrinkA=0, shrinkB=0, **k))
def head(ax, x, t): ax.text(x, .965, t, ha='center', fontsize=12.5, color=INK, fontweight='bold')

# 그림 3-1 전체 구조
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
head(ax, .125, '입력'); head(ax, .5, '처리'); head(ax, .875, '출력')
for y, t, l in [(.71, '기상 예보', ['Open-Meteo 5개 모델', '1·2·3일 전 실행분']), (.47, '전력 실측', ['전력거래소 수요·태양광·풍력', '출력제어 기록']), (.23, '전력 가격', ['EPSIS 제주 시간별 SMP']), (.02, '달력', ['요일·공휴일·계절'])]:
    box(ax, .01, y, .23, .19 if y > .05 else .15, t, l)
box(ax, .33, .71, .36, .19, '수급 예측 모델', ['시간별 수요, 수요-태양광', 'HistGradientBoosting, 월별 재학습'], fc=MODEL)
box(ax, .33, .47, .36, .19, '충전일 선택', ['3일 중 낮 수요-태양광 예측이', '가장 낮은 날 선택'], fc=MODEL)
box(ax, .33, .23, .36, .19, '저가일 예측 모델', ['낮 SMP 0원 이하 확률', '5개 예보별 분류기의 확률 평균'], fc=MODEL)
box(ax, .33, .02, .36, .15, '기록·채점', ['모든 예측을 먼저 저장, 실측 공개 후 비교'], fc=MODEL)
box(ax, .76, .71, .23, .19, '수급 예보', ['낮 수요, 저녁 최대 수요', '수요-태양광'], fc=OUTC)
box(ax, .76, .47, .23, .19, '충전일 추천', ['웹 화면, 알림(계획)'], fc=OUTC)
box(ax, .76, .23, .23, .19, '저가일 등급', ['높음 / 보통 / 낮음'], fc=OUTC)
box(ax, .76, .02, .23, .15, '실시간 기록', ['처음 보는 날의 성적'], fc=OUTC)
ax.plot([.275, .275], [.095, .805], color=MUTED, lw=1.3)
for y in (.805, .565, .325, .095): ax.plot([.24, .275], [y, y], color=MUTED, lw=1.3)
arrow(ax, (.275, .805), (.33, .805)); arrow(ax, (.275, .325), (.33, .325))
for y in (.805, .565, .325, .095): arrow(ax, (.69, y), (.76, y))
arrow(ax, (.51, .71), (.51, .66)); arrow(ax, (.51, .23), (.51, .17))
fig.savefig(f'{OUT}/fig12_system.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 그림 3-2 동작 흐름
fig, ax = plt.subplots(figsize=(11, 2.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
steps = [('매일 예보 수집', ['1·2·3일 전 예보', '최근 SMP'], IN_), ('고정 모델로 예측', ['내일·모레·글피', '수요-태양광, 등급'], MODEL), ('예측 저장', ['실측 공개 전에', '파일에 기록'], MODEL), ('충전일 안내', ['3일 중 추천일', '등급과 주의 사항'], OUTC), ('이용자 충전', ['11~14시 충전', '(실증 예정)'], OUTC), ('사후 채점', ['실제 가격·수요와', '비교해 공개'], IN_)]
w = .145; g = (1 - 6 * w) / 5
for i, (t, l, c) in enumerate(steps):
    x = i * (w + g); box(ax, x, .1, w, .8, t, l, fc=c, ts=11, ls=9.2, k=2.05)
    if i < 5: arrow(ax, (x + w, .5), (x + w + g, .5))
fig.savefig(f'{OUT}/fig13_flow.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig); print('ok')

# 포스터용 구조도: 실제 크기(폭 276pt)에서 읽히도록 세로로 구성
fig, ax = plt.subplots(figsize=(3.83, 3.25)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); K = 1.75
def bx(x, y, w, h, t, l, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.012', fc=c, ec=EDGE, lw=.8)); n = 1 if l else 0
    ax.text(x + w / 2, y + h / 2 + (.035 if l else 0), t, ha='center', va='center', fontsize=9.5, color=INK, fontweight='bold')
    if l: ax.text(x + w / 2, y + h / 2 - .04, l, ha='center', va='center', fontsize=8, color=MUTED)
bx(0, .8, 1, .19, '입력', '기상 예보 5종 · 전력 실측 · SMP · 달력', IN_)
bx(0, .52, .485, .19, '수급 예측 모델', '수요, 수요-태양광', MODEL); bx(.515, .52, .485, .19, '저가일 예측 모델', 'SMP 0원 이하 확률', MODEL)
bx(0, .24, .485, .19, '충전일 추천', '3일 중 여유 큰 날', OUTC); bx(.515, .24, .485, .19, '저가일 등급', '높음 · 보통 · 낮음', OUTC)
bx(0, 0, 1, .15, '실시간 기록 · 사후 채점', '', IN_)
for x in (.2425, .7575):
    arrow(ax, (x, .8), (x, .71)); arrow(ax, (x, .52), (x, .43)); arrow(ax, (x, .24), (x, .15))
fig.savefig(f'{OUT}/fig15_system_poster.png', dpi=300, bbox_inches='tight', pad_inches=.02, facecolor='white'); plt.close(fig)

# 포스터용 출력제어일 곡선 (실제 크기 폭 276pt)
import pandas as pd
pr = pd.read_csv(f'{A}/out_grid/profile_curtail.csv', index_col=0); hh = pr.index.values - .5
fig, ax = plt.subplots(figsize=(3.83, 1.95)); ax.axvspan(11, 14, color='#e6e5e0', alpha=.8, zorder=0); ax.plot(hh, pr.net_n, color='#8a8984', lw=1.6); ax.plot(hh, pr.net_c, color='#24600a', lw=2)
ax.text(12.5, pr.net_n.loc[13] + 125, f'그 밖의 날 {pr.net_n.loc[13]:.0f}', ha='center', fontsize=9, color=INK); ax.text(12.5, pr.net_c.loc[13] - 92, f'출력제어일 {pr.net_c.loc[13]:.0f}', ha='center', fontsize=9, color=INK, fontweight='bold')
ax.set_xlim(0, 24); ax.set_ylim(150, 950); ax.set_xticks(range(0, 25, 6)); ax.set_yticks([200, 500, 800]); ax.tick_params(labelsize=8.5, length=2); ax.spines[['top', 'right']].set_visible(False); ax.set_xlabel('시각', fontsize=9); ax.set_ylabel('MW', fontsize=9)
fig.savefig(f'{OUT}/fig18_profile_poster.png', dpi=300, bbox_inches='tight', pad_inches=.02, facecolor='white'); plt.close(fig)

# 포스터용 강건성 신뢰구간 그림 (폭 276pt)
rows = [('주말 우선 대비 (기본)', 66.7, 48.1, 87.4, '#24600a'), ('42일 블록 부트스트랩', 66.7, 45.1, 91.0, '#24600a'), ('ERA5 추정 순부하', 65.1, 46.2, 85.5, '#24600a'), ('기존 입력(best_match)', 60.4, 41.7, 81.5, '#24600a'), ('맑은 날 규칙 대비', 7.4, 1.4, 14.6, '#d9601a'), ('맑은 날 대비, ERA5', 8.8, 3.9, 14.4, '#d9601a')]
fig, ax = plt.subplots(figsize=(3.83, 1.85)); yy = list(range(len(rows)))[::-1]
for yi, (nm, m, lo, hi, c) in zip(yy, rows):
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.6); ax.plot(m, yi, 'o', color=c, ms=4.5); ax.text(hi + 2, yi, f'{m:.1f}', va='center', fontsize=8, color=INK)
ax.axvline(0, color='#8a8984', lw=.9, ls='--'); ax.set_yticks(yy, [r[0] for r in rows], fontsize=8.5); ax.set_xlim(-5, 105); ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', length=0)
ax.set_xlabel('제안 모델이 낮춘 수요-태양광 (MW, 95% CI)', fontsize=8.5); ax.spines[['top', 'right', 'left']].set_visible(False); ax.grid(axis='x', color='#e6e5e0', lw=.7)
fig.savefig(f'{OUT}/fig19_forest_poster.png', dpi=300, bbox_inches='tight', pad_inches=.02, facecolor='white'); plt.close(fig)
