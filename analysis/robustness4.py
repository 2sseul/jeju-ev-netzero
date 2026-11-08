"""최종 검증: 공식 예측과의 공정 비교, 일관된 4개 모델 전체 표, 같은 일수 할인 배정. 2026년은 2025년까지만 학습·기준을 고정한 표본 외 시험."""
import os, glob
A__ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A__}/grid_forecast.py', encoding='utf8').read().split('\nT0, TD = ')[0].replace("'2026-11-01 00:00'", "'2026-11-12 00:00'")
exec(src)
from sklearn.linear_model import Ridge
from scipy import stats as st
pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)
def blk(u, L=21, B=3000, seed=0):
    u = np.asarray(u, float); u = u[~np.isnan(u)]; n = len(u); rng = np.random.default_rng(seed); o = [u[np.concatenate([(x + np.arange(L)) % n for x in rng.integers(0, n, n // L + 1)])[:n]].mean() for _ in range(B)]; return np.round(np.percentile(o, [2.5, 97.5]), 1).tolist()
def nw_dm(d, lag=7):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); u = d - d.mean(); v = (u @ u) / n
    for k in range(1, lag + 1): v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / np.sqrt(v / n); return float(d.mean()), float(t), float(2 * (1 - st.norm.cdf(abs(t))))
def kpx_wide(pat):
    fr = []
    for f in sorted(glob.glob(f'{R_}/kpx_fc/{pat}')):
        x = pd.read_excel(f, header=None, engine='openpyxl'); hdr = x.iloc[1].astype(str).tolist(); hc_ = [i for i, h in enumerate(hdr) if h.endswith('h')]; d = x.iloc[2:, [0] + hc_].copy(); d.columns = ['date'] + [int(hdr[i][:-1]) for i in hc_]; d['date'] = pd.to_datetime(d.date.astype(str).str[:8], format='%Y%m%d', errors='coerce'); fr.append(d.dropna(subset=['date']).set_index('date').apply(pd.to_numeric, errors='coerce'))
    return pd.concat(fr).sort_index(kind='mergesort').pipe(lambda z: z[~z.index.duplicated(keep='last')])
M4 = ['ecmwf_ifs025', 'gfs_global', 'icon_global', 'gem_global']; vv = ['temp', 'rain', 'ws', 'rad', 'cloud']; hrs = [12, 13, 14]
def m4(L):
    d = dates(sum(W[L][m][vv] for m in M4) / 4).dropna(subset=['rad']); d = d.join(G[['net_s', 'dem', 'solar', 'dow', 'offday', 'holiday', 'month', 'doy']], how='left'); return d[d.date >= '2024-05-05']
FR = {L: m4(L) for L in (1, 2, 3)}; XC4 = vv + ['hour', 'month', 'doy', 'dow', 'offday', 'holiday']; XCAL = ['hour', 'month', 'doy', 'dow', 'offday', 'holiday']
def walk(L, X, y, a, b, freeze=None):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        cut = per.start_time if freeze is None or per.start_time < pd.Timestamp(freeze) else pd.Timestamp(freeze); t = FR[2][(FR[2].date < cut) & FR[2][y].notna()]; c_ = FR[L][FR[L].date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, y).predict(c_[X]), index=c_.index))
    return pd.concat(out)
def to_day(p, L): x = pd.DataFrame({'p': p, 'd': FR[L].date.reindex(p.index), 'h': FR[L].hour.reindex(p.index)}); return x[x.h.isin(hrs)].groupby('d').p.mean()
# ---------- A. 공식 예측과 공정 비교 ----------
print('[A] 공식 예측 공정 비교 (2025.06~2026.09)', flush=True)
OFc = kpx_wide('pss_*.xlsx'); H_ = list(range(1, 25)); Dm_ = PV['dem'][H_]; ix = OFc.index.intersection(Dm_.index); y_ = Dm_.loc[ix].stack(); o_ = OFc.loc[ix, H_].stack(); both = y_.index.intersection(o_.index); yl = y_[both]; ol = o_[both]; print(' 시간 정렬 상관: 공식 h와 실제 h-1, h, h+1')
for s in (-1, 0, 1):
    a = Dm_.loc['2025-06-01':'2026-09-30'].values.ravel(); b = OFc.reindex(Dm_.loc['2025-06-01':'2026-09-30'].index)[H_].values.ravel(); a = pd.Series(a).shift(s); print('   ', s, round(float(a.corr(pd.Series(b))), 4))
ee = (Dm_ - OFc.reindex(Dm_.index)[H_]); print(' 연도별 공식 편향(실제-공식)', ee.stack().groupby(level=0).mean().groupby(lambda d: d.year).mean().round(1).to_dict(), ' 시간대별(2025~)', ee['2025-06-01':'2026-09-30'].mean().round(0).iloc[[2, 7, 12, 18, 20]].to_dict())
bias = ee.shift(3).rolling(28, min_periods=14).mean(); OFb = OFc.reindex(Dm_.index)[H_] + bias  # 3일 전까지의 시간대별 편향을 더해 보정
hold = ('2025-06-01', '2026-09-30'); YY = Dm_.loc[hold[0]:hold[1]]
def met(P, nm):
    P = P.reindex(YY.index)[H_]; e = (YY - P); s = e.stack(); ys = YY.stack().reindex(s.index); d = e.abs().mean(axis=1); return dict(방식=nm, MAPE=round(float((s.abs() / ys).mean() * 100), 2), MAE=round(float(s.abs().mean()), 1), RMSE=round(float(np.sqrt((s ** 2).mean())), 1), 편향=round(float(s.mean()), 1)), d
Fx = {L: frame(L) for L in (1, 2)}; XOD = WM + CALG
def wk(Fy, X):
    out = []
    for per in pd.period_range('2025-06', '2026-09', freq='M'):
        t = Fy[(Fy.date < per.start_time) & Fy.dem.notna()]; c_ = Fy[Fy.date.dt.to_period('M') == per]; out.append(pd.DataFrame({'p': hgb(t, X, 'dem').predict(c_[X]), 'date': c_.date.values, 'hour': c_.hour.values}))
    return pd.concat(out).pivot_table(index='date', columns='hour', values='p')
def wk4(L):
    out = []
    for per in pd.period_range('2025-06', '2026-09', freq='M'):
        t = FR[2][(FR[2].date < per.start_time) & FR[2].dem.notna()]; c_ = FR[L][FR[L].date.dt.to_period('M') == per]; out.append(pd.DataFrame({'p': hgb(t, XC4, 'dem').predict(c_[XC4]), 'date': c_.date.values, 'hour': c_.hour.values}))
    return pd.concat(out).pivot_table(index='date', columns='hour', values='p')
PPs = {'공식 하루 전 (원값)': OFc, '공식 하루 전 (편향 보정)': OFb, '모델 1일 전 (best_match)': wk(Fx[1], XOD), '모델 2일 전 (best_match)': wk(Fx[2], XOD), '모델 1일 전 (4개 예보)': wk4(1), '모델 2일 전 (4개 예보)': wk4(2)}; rows = []; D_ = {}
for k, P in PPs.items(): r, d = met(P, k); rows.append(r); D_[k] = d
Ta = pd.DataFrame(rows); print(Ta.to_string(index=False)); Ta.to_csv(f'{OG}/rob_official_fair.csv', index=False, encoding='utf-8-sig')
for k in list(PPs)[2:]:
    for base in ['공식 하루 전 (원값)', '공식 하루 전 (편향 보정)']:
        u = (D_[base] - D_[k]).dropna(); r = nw_dm(u.values); print(f'  {base} - {k}: {r[0]:+.2f}MW p={r[2]:.3g} CI {blk(u.values)}')
print('  월별 MAPE 공식/모델2일(4개)', {str(p): (round(float(((YY - OFc.reindex(YY.index)[H_]).abs() / YY).loc[str(p)].stack().mean() * 100), 1), round(float(((YY - PPs['모델 2일 전 (4개 예보)'].reindex(YY.index)[H_]).abs() / YY).loc[str(p)].stack().mean() * 100), 1)) for p in pd.period_range('2025-06', '2026-09', freq='Q')})
# ---------- B. 4개 예보 충전일 전체 표 ----------
print('\n[B] 4개 예보 모델: 3일 중 하루', flush=True)
Pn = {L: walk(L, XC4, 'net_s', '2024-09', '2026-11', freeze='2026-01-01') for L in FR}; pm = {L: to_day(Pn[L], L) for L in FR}; rd = {L: FR[L][FR[L].hour.isin(hrs)].groupby('date').rad.mean() for L in FR}
def tao4(Fy):
    Hr = pd.get_dummies(Fy.hour, prefix='h').reindex(columns=[f'h_{i}' for i in range(1, 25)], fill_value=0).astype(float).values; Mo = pd.get_dummies(Fy.month, prefix='m').reindex(columns=[f'm_{i}' for i in range(1, 13)], fill_value=0).astype(float).values; dw = np.where(Fy.holiday == 1, 6, Fy.dow.values); HD = np.zeros((len(Fy), 168)); HD[np.arange(len(Fy)), (Fy.hour.values.astype(int) - 1) * 7 + dw.astype(int)] = 1; T = (Fy.temp.values / 30)[:, None]; Tp = np.hstack([T, T ** 2, T ** 3]); R = (Fy.rad.values / 1000)[:, None]; Cl = (Fy.cloud.values / 100)[:, None]; parts = [HD, Mo, Tp, R * Hr, R * Mo, Cl * Hr]
    for k in range(3): parts += [Tp[:, [k]] * Mo, Tp[:, [k]] * Hr]
    return np.hstack(parts)
def walk_l(L):
    out = []; Xt = tao4(FR[2])
    for per in pd.period_range('2024-09', '2025-12', freq='M'):
        m = ((FR[2].date < per.start_time) & FR[2].net_s.notna()).values; c_ = FR[L][FR[L].date.dt.to_period('M') == per]; out.append(pd.Series(Ridge(alpha=1.0).fit(Xt[m], FR[2].net_s.values[m]).predict(tao4(c_)), index=c_.index))
    return pd.concat(out)
pl = {L: to_day(walk_l(L), L) for L in FR}; Pc = {L: walk(L, XCAL, 'net_s', '2024-09', '2026-11', freeze='2026-01-01') for L in FR}; pc = {L: to_day(Pc[L], L) for L in FR}
Jw = J.copy(); Jw.columns = range(1, 25); actS = PV['net_s'][hrs].mean(axis=1); sm = Jw[hrs].mean(axis=1); low = (Jw[hrs] <= 0).mean(axis=1) * 100; Era = pd.read_csv(f'{OG}/label_check_era.csv', parse_dates=['date']).set_index('date').net_e
def bund(a, b, need_act=True):
    days = [d for d in pd.date_range(a, b) if d in sm.index and all(d in pm[L].index for L in pm) and (not need_act or pd.notna(actS.get(d)))]; out = []
    for s in range(len(days) - 2):
        B = days[s:s + 3]
        if (B[2] - B[0]).days != 2: continue
        Z = list(zip(B, (1, 2, 3))); wkd = [d for d in B if OFF[d] == 1]; Rl = lambda z: rd[z[1]][z[0]]
        pk = {'첫날 고정': B[0], '주말 우선': wkd[0] if wkd else B[0], '달력 모델': min(Z, key=lambda z: pc[z[1]][z[0]])[0], '맑은 날 규칙': max(Z, key=Rl)[0], '주말 우선+맑은 날': max([z for z in Z if z[0] in wkd], key=Rl)[0] if wkd else max(Z, key=Rl)[0], '운영 모델': min(Z, key=lambda z: pm[z[1]][z[0]])[0]}
        if need_act: pk['회귀'] = min(Z, key=lambda z: pl[z[1]][z[0]])[0]; pk['완벽한 선택'] = min(B, key=lambda d: actS[d])
        else: pk['완벽한 선택(최저가)'] = min(B, key=lambda d: sm[d])
        for k, d in pk.items(): out.append(dict(start=B[0], 방식=k, net=actS.get(d, np.nan), era=Era.get(d, np.nan), smp=sm[d], low=low[d]))
    R = pd.DataFrame(out); return {c_: R.pivot(index='start', columns='방식', values=c_) for c_ in ['net', 'era', 'smp', 'low']}
R25 = bund('2025-01-01', '2025-12-31'); od = ['첫날 고정', '주말 우선', '달력 모델', '주말 우선+맑은 날', '맑은 날 규칙', '회귀', '운영 모델', '완벽한 선택']; T = pd.DataFrame({k: R25[k][od].mean().round(1) for k in R25}); print(' 2025', len(R25['net'])); print(T.to_string()); T.to_csv(f'{OG}/final_charge_2025.csv', encoding='utf-8-sig')
for base in ['주말 우선', '달력 모델', '맑은 날 규칙', '주말 우선+맑은 날', '회귀']:
    for c_, nm in [('net', '수요-태양광'), ('era', '추정 순부하')]:
        u = (R25[c_][base] - R25[c_]['운영 모델']); print(f'   모델-{base} {nm} {u.mean():.1f}MW CI {blk(u.values)} CI42 {blk(u.values, 42)} 승패 {(u > 0).sum()}/{(u < 0).sum()}')
N = R25['net']; print('  분기별 주말 대비', (N['주말 우선'] - N['운영 모델']).groupby(N.index.quarter).mean().round(1).to_dict(), ' 맑은 날 대비', (N['맑은 날 규칙'] - N['운영 모델']).groupby(N.index.quarter).mean().round(1).to_dict(), ' 달성률', round(100 * (N['첫날 고정'].mean() - N['운영 모델'].mean()) / (N['첫날 고정'].mean() - N['완벽한 선택'].mean()), 1))
R26 = bund('2026-01-01', '2026-11-07', False); od2 = ['첫날 고정', '주말 우선', '달력 모델', '주말 우선+맑은 날', '맑은 날 규칙', '운영 모델', '완벽한 선택(최저가)']; T2 = pd.DataFrame({k: R26[k][od2].mean().round(1) for k in ['smp', 'low']}); print(' 2026 표본 외 (2025년까지 학습)', len(R26['smp'])); print(T2.to_string()); T2.to_csv(f'{OG}/final_charge_2026.csv', encoding='utf-8-sig')
for base in ['주말 우선', '달력 모델', '맑은 날 규칙', '주말 우선+맑은 날']:
    u = R26['low']['운영 모델'] - R26['low'][base]; v = R26['smp'][base] - R26['smp']['운영 모델']; print(f'   모델-{base}: 저가시간 {u.mean():+.1f}%p CI {blk(u.values)}, SMP {v.mean():+.1f}원 CI {blk(v.values)}')
# ---------- C. 같은 일수 할인 배정 ----------
print('\n[C] 같은 일수 배정 (2일 전 결정)', flush=True)
loy = (Jw[list(range(11, 17))].min(axis=1) <= 0)
def scen(a, b, base_a, base_b):
    ev = pd.date_range(a, b); ev = ev[ev.isin(sm.index) & ev.isin(pm[2].index)]; offs = OFF.reindex(ev) == 1; q = offs.mean(); bs = pd.date_range(base_a, base_b); res = {}
    qa = actS.reindex(bs).quantile(q); qr = rd[2].reindex(bs).quantile(1 - q); qc = pc[2].reindex(bs).quantile(q); qm = pm[2].reindex(bs).quantile(q)
    picks = {'주말·공휴일 고정': offs, '봄·가을 고정': pd.Series(ev.month.isin([3, 4, 5, 10, 11]), index=ev), '달력 모델': pc[2].reindex(ev) <= qc, '맑은 날 규칙': rd[2].reindex(ev) >= qr, '운영 모델': pm[2].reindex(ev) <= qm}
    rows = []
    for k, pk in picks.items():
        pk = pk.fillna(False).astype(bool); rows.append(dict(방식=k, 일수=int(pk.sum()), 수요태양광=round(float(actS.reindex(ev)[pk].mean()), 1), 추정순부하=round(float(Era.reindex(ev)[pk].mean()), 1), SMP=round(float(sm.reindex(ev)[pk].mean()), 1), 저가시간비중=round(float(low.reindex(ev)[pk].mean()), 1), 저가일포착=round(float(loy.reindex(ev)[pk].sum() / loy.reindex(ev).sum() * 100), 1), 정밀도=round(float(loy.reindex(ev)[pk].mean() * 100), 1)))
    print(f'  {a}~{b}: 저가일 {int(loy.reindex(ev).sum())} / {len(ev)}일, 기준 비중 {q:.3f}'); t = pd.DataFrame(rows); print(t.to_string(index=False))
    n = len(ev); rng = np.random.default_rng(0); yl = loy.reindex(ev).values.astype(float); lw = low.reindex(ev).values; pm_ = picks['운영 모델'].fillna(False).values.astype(bool); out = {}
    for k in ['주말·공휴일 고정', '봄·가을 고정', '달력 모델', '맑은 날 규칙']:
        pb = picks[k].fillna(False).values.astype(bool); d1, d2 = [], []
        for _ in range(3000):
            i = np.concatenate([(x + np.arange(21)) % n for x in rng.integers(0, n, n // 21 + 1)])[:n]; yy = yl[i]
            if yy.sum() == 0: continue
            d1.append(100 * (yy[pm_[i]].sum() - yy[pb[i]].sum()) / yy.sum()); d2.append(np.nanmean(lw[i][pm_[i]]) - np.nanmean(lw[i][pb[i]]))
        print(f'    모델-{k}: 저가일 포착 차이 CI {np.round(np.percentile(d1, [2.5, 97.5]), 1).tolist()}, 저가시간 비중 차이 CI {np.round(np.nanpercentile(d2, [2.5, 97.5]), 1).tolist()}')
    return t
t25 = scen('2025-01-01', '2025-12-31', '2024-09-01', '2024-12-31'); t26 = scen('2026-01-01', '2026-11-07', '2025-01-01', '2025-12-31'); t26.to_csv(f'{OG}/final_discount_2026.csv', index=False, encoding='utf-8-sig'); t25.to_csv(f'{OG}/final_discount_2025.csv', index=False, encoding='utf-8-sig')
# ---------- C2. 현행 정책 일정과 비교 ----------
print('\n[C2] 현행 정책 일정: 봄(4/18~5/31)·가을(9/5~10/31) 주말·공휴일', flush=True)
def season(a1, a2, b1, b2): return pd.DatetimeIndex(list(pd.date_range(a1, a2)) + list(pd.date_range(b1, b2)))
win26 = season('2026-04-18', '2026-05-31', '2026-09-05', '2026-10-31'); win26 = win26[win26.isin(sm.index) & win26.isin(pm[2].index)]; pol = OFF.reindex(win26) == 1; q = float(pol.mean())
base25 = season('2025-04-18', '2025-05-31', '2025-09-05', '2025-10-31'); thr = pm[2].reindex(base25).quantile(q); thr_r = rd[2].reindex(base25).quantile(1 - q)
pk_m = pm[2].reindex(win26) <= thr; pk_r = rd[2].reindex(win26) >= thr_r; yv = loy.reindex(win26)
print(f'  정책 창 {len(win26)}일, 저가일 {int(yv.sum())}일, 정책 할인일 {int(pol.sum())}일 (비중 {q:.3f}, 2025년 같은 창에서 기준값 결정)')
rows = []
for k, pk in [('현행 일정 (봄·가을 주말·공휴일)', pol), ('같은 창, 맑은 날 규칙', pk_r), ('같은 창, 운영 모델', pk_m)]:
    pk = pk.fillna(False).astype(bool); rows.append(dict(방식=k, 일수=int(pk.sum()), 저가일포착=round(float(yv[pk].sum() / yv.sum() * 100), 1), 정밀도=round(float(yv[pk].mean() * 100), 1), 저가시간비중=round(float(low.reindex(win26)[pk].mean()), 1), SMP=round(float(sm.reindex(win26)[pk].mean()), 1)))
T5 = pd.DataFrame(rows); print(T5.to_string(index=False)); T5.to_csv(f'{OG}/final_policy_window_2026.csv', index=False, encoding='utf-8-sig')
n = len(win26); rng = np.random.default_rng(0); a_ = pk_m.fillna(False).values.astype(bool); b_ = pol.values.astype(bool); yy_ = yv.values.astype(float); o1 = []
for _ in range(3000):
    i = np.concatenate([(x + np.arange(14)) % n for x in rng.integers(0, n, n // 14 + 1)])[:n]
    if yy_[i].sum() == 0: continue
    o1.append(100 * (yy_[i][a_[i]].sum() - yy_[i][b_[i]].sum()) / yy_[i].sum())
print('  모델-현행 저가일 포착 차이 CI(14일 블록)', np.round(np.percentile(o1, [2.5, 97.5]), 1).tolist())
pd.DataFrame({'p2': pm[2], 'rad2': rd[2]}).to_csv(f'{OG}/m4_daily_pred_lead2.csv', encoding='utf-8-sig')
# ---------- D. 예보 일치 ---------
a25 = actS['2025-01-01':'2025-12-31'].dropna(); q20 = a25 <= a25.quantile(.2); print('\n2025 수요-태양광 하위 20%의 평일 비중', round(float((OFF.reindex(a25.index)[q20] == 0).mean() * 100), 1)); print('done')
