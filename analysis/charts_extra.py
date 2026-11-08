"""추가 분석 그래프. smp_model.py를 먼저 실행해야 한다."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager as fm

A = os.path.dirname(os.path.abspath(__file__)); OUT = f'{A}/그림'; os.makedirs(OUT, exist_ok=True)
fm.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.axisbelow': True, 'axes.unicode_minus': False})
BLUE, ORANGE, INK, GRID, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#e6e5e0', '#52514e'
d = pd.read_csv(f'{A}/out_smp/daily.csv', parse_dates=['date']).set_index('date')
y = d.y.values
m2 = (d.p2 >= d.thr2).astype(int).values
rules = [('주말·공휴일 고정', d.offday.values.astype(int)), ('봄·가을만', d.month.isin([3, 4, 5, 10, 11]).astype(int).values), ('모델 (2일 전 예보)', m2)]

# 1. 규칙 비교: 같은 잉여일을 몇 일의 추천으로 잡는가
fig, ax = plt.subplots(figsize=(9, 3.6))
yp = np.arange(len(rules))[::-1]; h = .34
for (nm, m), yi in zip(rules, yp):
    rec = 100 * (y & m).sum() / y.sum(); pre = 100 * (y & m).sum() / m.sum()
    ax.barh(yi + h / 2 + .01, rec, h, color=BLUE); ax.barh(yi - h / 2 - .01, pre, h, color=ORANGE)
    ax.text(rec + 1, yi + h / 2 + .01, f'{rec:.0f}%', va='center', fontsize=11, color=INK); ax.text(pre + 1, yi - h / 2 - .01, f'{pre:.0f}%', va='center', fontsize=11, color=INK)
ax.set_yticks(yp, [f'{nm}\n추천 {m.sum()}일' for nm, m in rules]); ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel('%')
ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.bar(0, 0, color=BLUE, label='재현율: 실제 저가일 중 포착'); ax.bar(0, 0, color=ORANGE, label='정밀도: 추천일 중 실제 저가일')
ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.45, 1.0), fontsize=11)
ax.set_title(f'낮 SMP 0원 이하인 날 포착 성능 (2025.06~2026.10, 실제 저가일 {y.sum()}일)', loc='left', fontsize=13, color=INK, pad=34)
fig.savefig(f'{OUT}/fig6_rule_compare.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 2. 2026년 날짜 띠: 실제, 모델, 고정 규칙
s = d['2026-01-01':'2026-10-31']; rows = [('실제 낮 SMP 0원 이하', s.y.values, INK), ('모델 추천 (2일 전 예보)', (s.p2 >= s.thr2).astype(int).values, BLUE), ('주말·공휴일 고정', s.offday.values.astype(int), ORANGE)]
fig, ax = plt.subplots(figsize=(10, 2.8))
for i, (nm, v, c) in enumerate(rows):
    ax.vlines(s.index[v == 1], 2 - i - .34, 2 - i + .34, color=c, lw=1.8)
ax.set_yticks([2, 1, 0], [r[0] for r in rows]); ax.set_ylim(-.6, 2.6); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-11-01')); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m월')); ax.grid(axis='x', color=GRID, lw=.8)
ax.set_title('2026년 저가일과 추천일 비교', loc='left', fontsize=13, color=INK)
fig.savefig(f'{OUT}/fig7_timeline_2026.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 3. 안내 등급별 실제 비율 (extras_model.py 결과)
c = pd.read_csv(f'{A}/out_smp/daily_calibrated.csv', parse_dates=['date']).set_index('date')
t = pd.cut(c.p_cal, [0, .1, .4, 1.01], right=False, labels=['낮음', '보통', '높음']); g = c.groupby(t, observed=True).y.agg(['mean', 'size', 'sum']).loc[['높음', '보통', '낮음']]
fig, ax = plt.subplots(figsize=(8, 2.8)); yy = np.arange(3)[::-1]
ax.barh(yy, g['mean'] * 100, .55, color=BLUE)
for yi, (k, r) in zip(yy, g.iterrows()): ax.text(r['mean'] * 100 + 1, yi, f"{r['mean'] * 100:.1f}%  ({int(r['sum'])}/{int(r['size'])}일)", va='center', fontsize=11, color=INK, bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=3)
base_rate = 100 * c.y.mean(); ax.axvline(base_rate, color=MUTED, lw=1, ls=':'); ax.text(base_rate + .8, 2.47, f'전체 평균 {base_rate:.1f}%', fontsize=10, color=MUTED)
ax.set_yticks(yy, g.index); ax.set_xlim(0, 100); ax.set_xlabel('실제 낮 SMP 0원 이하 비율 (%)'); ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_title('2일 전 안내 등급별 실제 적중 (2025.06~2026.10)', loc='left', fontsize=13, color=INK, pad=14)
fig.savefig(f'{OUT}/fig8_tier_reliability.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
# 4. 유연 충전: 0원 이하 시간 충전 비중
fx = pd.read_csv(f'{A}/out_price/flex_compare.csv'); order = ['첫날 11~14시', '주말·공휴일 우선 11~14시', '과거 패턴 11~14시', '가격 예측 11~14시', '앙상블 확률 11~14시', '완벽 예측 11~14시']; names = ['첫날 고정', '주말·공휴일 우선', '과거 가격 패턴', '시간별 가격 예측', '앙상블 확률', '완벽한 예측(상한)']
fig, ax = plt.subplots(figsize=(9, 4.4)); yy = np.arange(len(order))[::-1]; h = .36
for yi, k in zip(yy, order):
    dv = fx[(fx.구간 == '개발') & (fx.방식 == k)].저가시간비중.iloc[0]; hd = fx[(fx.구간 == '확인') & (fx.방식 == k)].저가시간비중.iloc[0]
    ax.barh(yi + h / 2 + .01, dv, h, color=BLUE); ax.barh(yi - h / 2 - .01, hd, h, color=ORANGE)
    ax.text(dv + .25, yi + h / 2 + .01, f'{dv:.1f}%', va='center', fontsize=10.5, color=INK); ax.text(hd + .25, yi - h / 2 - .01, f'{hd:.1f}%', va='center', fontsize=10.5, color=INK)
ax.barh(0, 0, color=BLUE, label='개발 구간 (2025.06~2026.05)'); ax.barh(0, 0, color=ORANGE, label='확인 구간 (2026.06~2026.11)')
ax.set_yticks(yy, names); ax.set_xlim(0, 24); ax.set_xlabel('낮 11~14시 충전 중 SMP 0원 이하 시간의 비중 (%)'); ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.42, 1.0), fontsize=11); ax.set_title('3일 중 하루를 골라 충전할 때 잉여 시간에 충전되는 비중', loc='left', fontsize=13, color=INK, pad=34)
fig.savefig(f'{OUT}/fig9_flex_surplus.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('ok', [(nm, int(m.sum())) for nm, m in rules])

# 5. 공식 예측과 비교 (robustness3.py, robustness4.py)
of = pd.read_csv(f'{A}/out_grid/rob_official_fair.csv').set_index('방식'); tq = pd.read_csv(f'{A}/out_grid/rob_demand.csv'); tq = tq[tq.구간 == '확인'].set_index('방식')
items = [('같은 요일유형 패턴', tq.loc['같은 요일유형 패턴', 'MAPE'], '#b8c2b0'), ('공개 하루 전 예측', of.loc['공식 하루 전 (원값)', 'MAPE'], ORANGE), ('공개 하루 전 (편향 보정)', of.loc['공식 하루 전 (편향 보정)', 'MAPE'], ORANGE), ('회귀, 2일 전 (기온+일사)', tq.loc['Tao 회귀(기온+일사)', 'MAPE'], '#b8c2b0'), ('모델, 2일 전 (4개 예보)', of.loc['모델 2일 전 (4개 예보)', 'MAPE'], BLUE), ('모델, 1일 전 (4개 예보)', of.loc['모델 1일 전 (4개 예보)', 'MAPE'], BLUE)]
fig, ax = plt.subplots(figsize=(9, 3.5)); yy = np.arange(len(items))[::-1]
for yi, (nm, v, col) in zip(yy, items): ax.barh(yi, v, .62, color=col); ax.text(v + .08, yi, f'{v:.2f}%', va='center', fontsize=11, color=INK)
ax.set_yticks(yy, [x[0] for x in items]); ax.set_xlim(0, 9.6); ax.set_xlabel('시간별 제주 수요 MAPE (%), 2025.06~2026.09'); ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); ax.set_title('전력거래소 공개 예측과 비교', loc='left', fontsize=13, color=INK, pad=12)
fig.savefig(f'{OUT}/fig10_grid_forecast.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
# 6. 충전일 선택
cn = pd.read_csv(f'{A}/out_grid/final_charge_2025.csv', index_col=0); order6 = ['첫날 고정', '주말 우선', '달력 모델', '주말 우선+맑은 날', '맑은 날 규칙', '회귀', '운영 모델', '완벽한 선택']; names6 = ['첫날 고정', '주말·공휴일 우선', '달력만 쓴 모델', '주말 우선+맑은 날', '맑은 날 규칙', '회귀 (같은 입력)', '운영 모델', '완벽한 선택']
fig, ax = plt.subplots(figsize=(9, 3.9)); yy = np.arange(len(order6))[::-1]
for yi, k in zip(yy, order6):
    v = cn.loc[k, 'net']; ax.barh(yi, v, .62, color=BLUE if k == '운영 모델' else '#b8c2b0'); ax.text(v + 6, yi, f'{v:.0f}', va='center', fontsize=10.5, color=INK)
ax.set_yticks(yy, names6); ax.set_xlim(0, 640); ax.set_xlabel('11~14시 충전 시간의 실제 수요-태양광 (MW), 낮을수록 전력망 여유가 큰 시간'); ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); ax.set_title('3일 중 하루를 고르는 방식별 수요-태양광 (2025년)', loc='left', fontsize=13, color=INK, pad=12)
fig.savefig(f'{OUT}/fig11_charge_netload.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
# 7. 출력제어일의 시간대 곡선
pr = pd.read_csv(f'{A}/out_grid/profile_curtail.csv', index_col=0); hh = pr.index.values - .5
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.plot(hh, pr.net_n, color='#8a8984', lw=2); ax.plot(hh, pr.net_c, color=ORANGE, lw=2.4); ax.axvspan(11, 14, color=GRID, alpha=.7, zorder=0)
ax.text(13.1, pr.net_c.loc[13] - 42, f'출력제어일 {pr.net_c.loc[13]:.0f}MW', fontsize=11, color=INK, ha='center'); ax.text(13.1, pr.net_n.loc[13] + 50, f'그 밖의 날 {pr.net_n.loc[13]:.0f}MW', fontsize=11, color=INK, ha='center'); ax.text(11.1, 862, '충전 시간 11~14시', fontsize=10, color=MUTED)
ax.set_xlim(0, 24); ax.set_ylim(200, 920); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel('시각'); ax.set_ylabel('수요-태양광 (MW)'); ax.grid(axis='y', color=GRID, lw=.8); ax.set_title('공식 출력제어일과 그 밖의 날의 하루 곡선 (2023.01~2024.05)', loc='left', fontsize=13, color=INK, pad=12)
fig.savefig(f'{OUT}/fig16_curtail_profile.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
# 8. 같은 일수 할인 배정 2026
d26 = pd.read_csv(f'{A}/out_grid/final_discount_2026.csv').set_index('방식').loc[['주말·공휴일 고정', '봄·가을 고정', '달력 모델', '맑은 날 규칙', '운영 모델']]; nm8 = [f'{n} ({int(r.일수)}일)' for n, r in zip(['주말·공휴일 고정', '봄·가을 고정', '달력만 쓴 모델', '맑은 날 규칙', '운영 모델'], d26.itertuples())]
fig, ax = plt.subplots(figsize=(9, 3.6)); yy = np.arange(len(d26))[::-1]; h = .36
for yi, (_, r) in zip(yy, d26.iterrows()):
    ax.barh(yi + h / 2 + .01, r.저가일포착, h, color=BLUE); ax.barh(yi - h / 2 - .01, r.정밀도, h, color=ORANGE); ax.text(r.저가일포착 + 1, yi + h / 2 + .01, f'{r.저가일포착:.0f}%', va='center', fontsize=10.5, color=INK); ax.text(r.정밀도 + 1, yi - h / 2 - .01, f'{r.정밀도:.0f}%', va='center', fontsize=10.5, color=INK)
ax.barh(0, 0, color=BLUE, label='실제 저가일 중 포착'); ax.barh(0, 0, color=ORANGE, label='선택한 날 중 실제 저가일'); ax.set_yticks(yy, nm8); ax.set_xlim(0, 100); ax.set_xlabel('%'); ax.grid(axis='x', color=GRID, lw=.8); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.42, 1.0), fontsize=11); ax.set_title('2025년까지 고정한 기준으로 2026년 할인일을 배정할 때', loc='left', fontsize=13, color=INK, pad=32)
fig.savefig(f'{OUT}/fig17_discount_2026.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
