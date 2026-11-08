"""심사 대비 강건성 점검. grid_forecast.py의 준비부를 재사용한다.
1 데이터 품질(단절 점검) 2 예보 시점 검증(리드별 오차, 플라시보) 3 수요: 전통 회귀(Tao 기준 모델)와 비교, DM 검정, 최악 구간
4 충전일: 맑은 날 규칙 등과 비교, 시간대·기간·연도·분기·블록 길이 민감도 5 공식 출력제어 기록과 직접 대조.
주 가설(결과 보기 전 고정): H1 수요 모델 < 전통 회귀, H2 충전일 모델 < 주말 우선, H3 충전일 모델 < 일사량 최대 규칙. Holm 보정."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_forecast.py'), encoding='utf8').read().split('\nT0, TD = ')[0]
exec(src)
from sklearn.linear_model import Ridge
from scipy import stats as st
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
F = {L: frame(L) for L in (1, 2, 3)}; F2_ = F[2]; XOD = WM + CALG; XON = WS + CALG; RES = {}

def nw_dm(d, lag=7):
    """Diebold-Mariano: 일별 손실 차이, Newey-West 분산. 양수면 두 번째 방식의 손실이 더 큼."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); u = d - d.mean(); v = (u @ u) / n
    for k in range(1, lag + 1): v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / np.sqrt(v / n); return float(d.mean()), float(t), float(2 * (1 - st.norm.cdf(abs(t)))), n
def daily_abs(y, p, idx):
    e = pd.DataFrame({'e': (y - p).abs(), 'd': G.date.reindex(idx)}, index=idx).dropna(); return e.groupby('d').e.mean()
def blk(u, L=21, B=3000, seed=0):
    u = np.asarray(u, float); n = len(u); rng = np.random.default_rng(seed); o = [u[np.concatenate([(x + np.arange(L)) % n for x in rng.integers(0, n, n // L + 1)])[:n]].mean() for _ in range(B)]; return np.round(np.percentile(o, [2.5, 97.5]), 1).tolist()

# ---------- 1. 데이터 품질 ----------
print('[1] 데이터 품질', flush=True)
Dq = G[['dem']].dropna(); print(' 수요 행', len(Dq), '중복 시각', int(Dq.index.duplicated().sum()), '0 이하', int((Dq.dem <= 0).sum()), '100MW 미만', int((Dq.dem < 100).sum()), '1500MW 초과', int((Dq.dem > 1500).sum()))
dj = Dq.dem.diff().abs(); print(' 수요 시간 변화 상위', dj.nlargest(5).round(0).to_dict())
q = pd.DataFrame({'dem_mean': G.dem, 'dem_max': G.dem, 'sol_max': G.solar, 'wnd_max': G.wind}).groupby(G.index.to_period('Q')).agg({'dem_mean': 'mean', 'dem_max': 'max', 'sol_max': 'max', 'wnd_max': 'max'}).round(0); print(q['2021Q1':'2026Q3'].to_string())
# 태양광: 맑은 낮의 기록 출력 대 예보 일사 (단절이면 비율이 꺾임)
c = F2_[F2_.hour.between(11, 14) & (F2_.rad_mean > 500)]; rs = (c.solar / c.rad_mean).groupby(c.date.dt.to_period('Q')).agg(['median', 'size']); rs['median'] = (rs['median'] * 1000).round(0); print(' 태양광/예보일사 x1000 분기별\n', rs.T.to_string())
c2 = F2_[(F2_.ws_mean > 15)]; rw = (c2.wind_pot / c2.ws3_mean).groupby(c2.date.dt.to_period('Q')).median() * 1e4; print(' 풍력/풍속^3 x1e4 분기별', rw.round(1).to_dict())
Dd_ = G.dem.resample('D').mean(); print(' 수요 단위 변경 전후(2023-08~10 월평균)', Dd_['2023-08':'2023-10'].resample('M').mean().round(0).tolist())

# ---------- 2. 예보 시점 검증 ----------
print('\n[2] 예보 시점 검증', flush=True)
era = []
for sx, x in zip(SITES, json.load(open(f'{A_}/wx/era.json'))):
    h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); era.append(h.set_index('ts')[['shortwave_radiation', 'temperature_2m', 'wind_speed_10m', 'cloud_cover']].astype(float))
E = sum(era) / 4; E.columns = ['rad_mean', 'temp_mean', 'ws_mean', 'cloud_mean']; W_ = {L: load_wx(L) for L in (1, 2, 3)}; ix = E.index.intersection(W_[1].index).intersection(W_[2].index).intersection(W_[3].index); ix = ix[(ix >= '2024-03-01') & (ix <= '2024-06-30')]; print(' 비교 시간', len(ix), ix.min(), ix.max())
for v in ['rad_mean', 'temp_mean', 'ws_mean', 'cloud_mean']: print('  ', v, 'MAE 리드 1/2/3', [round(float((W_[L].loc[ix, v] - E.loc[ix, v]).abs().mean()), 2) for L in (1, 2, 3)])
def walk2(Fx, X, y, a, b, tr_end=None, tr_start=None):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = Fx[(Fx.date < per.start_time) & Fx[y].notna()]
        if tr_start is not None: t = t[t.date >= per.start_time - pd.Timedelta(days=tr_start)]
        c_ = Fx[Fx.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, y).predict(c_[X]), index=c_.index))
    return pd.concat(out)
T0, TD = '2023-07', '2026-09'; HOLD = ('2025-06-01', '2026-09-30'); DEV = ('2023-07-01', '2025-05-31')
P_ops = walk2(F2_, XOD, 'dem', T0, TD)
Fp = F2_.copy()
for c_ in WM: Fp[c_] = Fp[c_].shift(24 * 7)  # 날씨를 7일 밀어 넣은 플라시보
P_placebo = walk2(Fp, XOD, 'dem', T0, TD)

# ---------- 3. 수요: 전통 회귀와 비교 ----------
print('\n[3] 수요: 기준 모델 비교', flush=True)
def tao(Fx, rad=False):
    Hr = pd.get_dummies(Fx.hour, prefix='h').reindex(columns=[f'h_{i}' for i in range(1, 25)], fill_value=0).astype(float).values; Mo = pd.get_dummies(Fx.month, prefix='m').reindex(columns=[f'm_{i}' for i in range(1, 13)], fill_value=0).astype(float).values
    dw = np.where(Fx.holiday == 1, 6, Fx.dow.values); HD = np.zeros((len(Fx), 168)); HD[np.arange(len(Fx)), (Fx.hour.values - 1) * 7 + dw] = 1
    T = (Fx.temp_mean.values / 30)[:, None]; Tp = np.hstack([T, T ** 2, T ** 3]); parts = [HD, Mo, Tp]
    for k in range(3): parts += [Tp[:, [k]] * Mo, Tp[:, [k]] * Hr]
    if rad: R = (Fx.rad_mean.values / 1000)[:, None]; Cl = (Fx.cloud_mean.values / 100)[:, None]; parts += [R * Hr, R * Mo, Cl * Hr]
    return np.hstack(parts)
def walk_lin(Fx, y, a, b, rad, Ftrain=None):
    Ft = F2_ if Ftrain is None else Ftrain; out = []; XT = tao(Ft, rad)
    for per in pd.period_range(a, b, freq='M'):
        m = ((Ft.date < per.start_time) & Ft[y].notna()).values; c_ = Fx[Fx.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(Ridge(alpha=1.0).fit(XT[m], Ft[y].values[m]).predict(tao(c_, rad)), index=c_.index))
    return pd.concat(out)
P_tao = walk_lin(F2_, 'dem', T0, TD, False); P_taor = walk_lin(F2_, 'dem', T0, TD, True); P_lag = walk2(F2_, WM + CALG + ['dem_lag', 'dem_lag7', 'dem_same'], 'dem', T0, TD); P_year = F2_.dem.shift(24 * 364)
PP = {'HGB 날씨+달력(운영)': P_ops, 'HGB 최근 실측 포함': P_lag, 'Tao 회귀(기온)': P_tao, 'Tao 회귀(기온+일사)': P_taor, '플라시보(날씨 7일 밀림)': P_placebo, '같은 요일유형 패턴': F2_.dem_same, '7일 전': F2_.dem_lag7, '364일 전': P_year}
rows = []
for nm, (a, b) in [('개발', DEV), ('확인', HOLD)]:
    m = (F2_.date >= a) & (F2_.date <= b) & F2_.dem.notna(); y = F2_.dem[m]
    for k, p in PP.items():
        e = (y - p.reindex(y.index)).dropna(); yy = y.reindex(e.index); rows.append(dict(구간=nm, 방식=k, MAPE=round(float((e.abs() / yy).mean() * 100), 2), MAE=round(float(e.abs().mean()), 1), RMSE=round(float(np.sqrt((e ** 2).mean())), 1), 편향=round(float(e.mean()), 1), 낮MAPE=round(float((e.abs() / yy)[F2_.hour.reindex(e.index).isin([12, 13, 14])].mean() * 100), 2)))
Tq = pd.DataFrame(rows); print(Tq.to_string(index=False)); Tq.to_csv(f'{OG}/rob_demand.csv', index=False, encoding='utf-8-sig')
m = (F2_.date >= HOLD[0]) & (F2_.date <= HOLD[1]) & F2_.dem.notna(); y = F2_.dem[m]; PV_ = {}
for nm, base, cmp_ in [('운영-Tao기온', P_tao, P_ops), ('운영-Tao기온일사', P_taor, P_ops), ('최근실측-Tao기온일사', P_taor, P_lag), ('운영-패턴', F2_.dem_same, P_ops)]:
    d = daily_abs(y, base.reindex(y.index), y.index) - daily_abs(y, cmp_.reindex(y.index), y.index); r = nw_dm(d.values); print(f'  DM {nm}: 평균 차이 {r[0]:.2f}MW t={r[1]:.2f} p={r[2]:.2g} n={r[3]}'); PV_[nm] = r[2]
ape = ((y - P_ops.reindex(y.index)).abs() / y); da = ape.groupby(G.date.reindex(y.index)).mean() * 100; print('  운영 모델 일별 MAPE 분위 50/90/95/99', da.quantile([.5, .9, .95, .99]).round(2).tolist(), ' 상위 8일', da.nlargest(8).round(1).to_dict())
print('  휴일/평일 MAPE', {('휴일' if k else '평일'): round(float(v * 100), 2) for k, v in ape.groupby(F2_.offday.reindex(y.index)).mean().items()}, ' 계절', {k: round(float(v * 100), 2) for k, v in ape.groupby(F2_.month.reindex(y.index) % 12 // 3).mean().items()})
P_win = walk2(F2_, XOD, 'dem', '2025-06', TD, tr_start=730); print('  학습 최근 2년만 MAPE', round(float(((y - P_win.reindex(y.index)).abs() / y).mean() * 100), 2))

# ---------- 4. 충전일 ----------
print('\n[4] 충전일: 규칙 비교와 민감도', flush=True)
def walk_op(FL, X, y, a, b):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = F2_[(F2_.date < per.start_time) & F2_[y].notna()]; c_ = FL[FL.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, y).predict(c_[X]), index=c_.index))
    return pd.concat(out)
A0, A1 = '2024-03', '2025-12'; Pn = {L: walk_op(F[L], XON, 'net_s', A0, A1) for L in F}; Pl = {L: walk_lin(F[L], 'net_s', A0, A1, True) for L in F}; print('  모델 준비 완료', flush=True)
SM = J.copy(); SM.columns = range(1, 25)
def dmean(s, Fr, hrs): x = pd.DataFrame({'p': s, 'date': Fr.date.reindex(s.index), 'hour': Fr.hour.reindex(s.index)}).dropna(); return x[x.hour.isin(hrs)].groupby('date').p.mean()
def bundles(a, b, hrs=(12, 13, 14), win=3):
    hrs = list(hrs); act = PV['net_s'][hrs].mean(axis=1); sol = PV['solar'][hrs].mean(axis=1); dem = PV['dem'][hrs].mean(axis=1); smp = SM[hrs].mean(axis=1); z = (SM[hrs] <= 0).mean(axis=1)
    pm = {L: dmean(Pn[L], F[L], hrs) for L in F}; pl = {L: dmean(Pl[L], F[L], hrs) for L in F}; rd = {L: F[L][F[L].hour.isin(hrs)].groupby('date').rad_mean.mean() for L in F}; cl = {L: F[L][F[L].hour.isin(hrs)].groupby('date').cloud_mean.mean() for L in F}; pat = same_type(PV['net_s'], 4)[hrs].mean(axis=1)
    days = [x for x in pd.date_range(a, b) if pd.notna(act.get(x)) and all(x in pm[L].index for L in pm)]; out = []; leads = (1, 2, 3)[:win]
    for s in range(len(days) - win + 1):
        B = days[s:s + win]
        if (B[-1] - B[0]).days != win - 1: continue
        Z = list(zip(B, leads)); wk = [d for d in B if OFF[d] == 1]
        pk = {'첫날 고정': B[0], '주말 우선': wk[0] if wk else B[0], '과거 패턴': min(B, key=lambda d: pat.get(d, 1e9)), '구름 최소': min(Z, key=lambda z_: cl[z_[1]][z_[0]])[0], '일사 최대': max(Z, key=lambda z_: rd[z_[1]][z_[0]])[0], '주말 우선+맑은 날': (max(wk, key=lambda d: rd[leads[B.index(d)]][d]) if wk else max(Z, key=lambda z_: rd[z_[1]][z_[0]])[0]), 'Tao 회귀': min(Z, key=lambda z_: pl[z_[1]][z_[0]])[0], '운영 모델': min(Z, key=lambda z_: pm[z_[1]][z_[0]])[0], '완벽한 예측': min(B, key=lambda d: act[d])}
        for k, d in pk.items(): out.append(dict(start=B[0], 방식=k, net=act[d], sol=sol[d], dem=dem[d], smp=smp.get(d, np.nan), low=100 * z.get(d, np.nan), day=d))
    R = pd.DataFrame(out); return {c_: R.pivot(index='start', columns='방식', values=c_) for c_ in ['net', 'sol', 'dem', 'smp', 'low']}
ALL_ = bundles('2024-03-01', '2025-12-31'); HLD = {k: v['2025-01-01':] for k, v in ALL_.items()}; order = ['첫날 고정', '주말 우선', '과거 패턴', '구름 최소', '일사 최대', '주말 우선+맑은 날', 'Tao 회귀', '운영 모델', '완벽한 예측']
for nm, R in [('전체 2024.03~2025.12', ALL_), ('확인 2025', HLD)]:
    t = pd.DataFrame({k: R[k][order].mean().round(1) for k in R}); t['태양광비중'] = (100 * R['sol'][order].mean() / R['dem'][order].mean()).round(1); print(' ', nm, len(R['net'])); print(t.to_string())
    if nm.startswith('전체'): t.to_csv(f'{OG}/rob_charge.csv', encoding='utf-8-sig')
    for base in ['주말 우선', '일사 최대', '주말 우선+맑은 날', 'Tao 회귀']:
        u = (R['net'][base] - R['net']['운영 모델']).values; dm = nw_dm(u, 21); print(f'   운영 모델 대 {base}: 감소 {u.mean():.1f}MW CI21 {blk(u)} CI7 {blk(u, 7)} CI42 {blk(u, 42)} DM p={dm[2]:.2g} 승/패 {(u > 0).sum()}/{(u < 0).sum()}')
        if nm.startswith('확인') and base in ('주말 우선', '일사 최대'): PV_['충전-' + base] = dm[2]
N = ALL_['net']
print('  연도별', (N['첫날 고정'] - N['운영 모델']).groupby(N.index.year).mean().round(1).to_dict(), ' 주말 대비', (N['주말 우선'] - N['운영 모델']).groupby(N.index.year).mean().round(1).to_dict())
print('  분기별 첫날 대비', (N['첫날 고정'] - N['운영 모델']).groupby(N.index.quarter).mean().round(1).to_dict(), ' 주말 대비', (N['주말 우선'] - N['운영 모델']).groupby(N.index.quarter).mean().round(1).to_dict(), ' 일사 대비', (N['일사 최대'] - N['운영 모델']).groupby(N.index.quarter).mean().round(1).to_dict())
sens = []
for lab, hrs, w in [('11~14시, 3일', (12, 13, 14), 3), ('10~13시, 3일', (11, 12, 13), 3), ('12~15시, 3일', (13, 14, 15), 3), ('10~16시, 3일', (11, 12, 13, 14, 15, 16), 3), ('11~14시, 2일', (12, 13, 14), 2)]:
    R = bundles('2025-01-01', '2025-12-31', hrs, w)['net']; sens.append(dict(조건=lab, 묶음=len(R), 첫날=round(R['첫날 고정'].mean(), 1), 주말=round(R['주말 우선'].mean(), 1), 일사=round(R['일사 최대'].mean(), 1), 운영=round(R['운영 모델'].mean(), 1), 완벽=round(R['완벽한 예측'].mean(), 1), 주말대비CI=blk((R['주말 우선'] - R['운영 모델']).values), 달성률=round(100 * (R['첫날 고정'].mean() - R['운영 모델'].mean()) / (R['첫날 고정'].mean() - R['완벽한 예측'].mean()), 1)))
Ts = pd.DataFrame(sens); print(Ts.to_string(index=False)); Ts.to_csv(f'{OG}/rob_sensitivity.csv', index=False, encoding='utf-8-sig')
for cp in (1, .75, .5, .25, .1):
    print(f'   이행률 {cp:.0%}: 충전 시간 수요-태양광 {N["첫날 고정"].mean() * (1 - cp) + N["운영 모델"].mean() * cp:.1f}MW (따르지 않으면 첫날 충전 가정)')

# ---------- 5. 공식 출력제어 기록과 대조 ----------
print('\n[5] 공식 출력제어 기록과 대조 (2023-01~2024-05, 2일 전 입력)', flush=True)
Cu = G[['curt_solar', 'curt_wind']].copy(); Cu['c'] = ((Cu.curt_solar > 0) | (Cu.curt_wind > 0)).astype(int); Cu['date'] = G.date; Cu['hour'] = G.hour; Cu['cs'] = (Cu.curt_solar > 0).astype(int)
cday = Cu[Cu.hour.between(9, 17)].groupby('date').c.max(); cwin = Cu[Cu.hour.isin([12, 13, 14])].groupby('date').c.max(); cwh = Cu[Cu.hour.isin([12, 13, 14])].groupby('date').c.mean() * 100; csd = Cu[Cu.hour.between(9, 17)].groupby('date').cs.max()
Pc = walk2(F2_, XON, 'net_s', '2023-01', '2024-05'); pc = dmean(Pc, F2_, [12, 13, 14]); act = PV['net_s'][[12, 13, 14]].mean(axis=1); actw = PV['net'][[12, 13, 14]].mean(axis=1); dmd = PV['dem'][[12, 13, 14]].mean(axis=1); idx_ = pc.index[(pc.index >= '2023-01-01') & (pc.index <= '2024-05-31')]; idx_ = [d for d in idx_ if pd.notna(act.get(d)) and pd.notna(actw.get(d))]
Y = cday.reindex(idx_); print(' 일수', len(idx_), '9~17시 출력제어일', int(Y.sum()), '11~14시 포함일', int(cwin.reindex(idx_).sum()))
for lab, sc in [('실제 수요-태양광', -act.reindex(idx_)), ('실제 순부하(풍력 포함)', -actw.reindex(idx_)), ('실제 수요', -dmd.reindex(idx_)), ('2일 전 수요-태양광 예측', -pc.reindex(idx_))]:
    print('  AUC', lab, '9~17시', round(auc(Y, sc), 3), ' 11~14시', round(auc(cwin.reindex(idx_), sc), 3), ' 태양광 제어일', round(auc(csd.reindex(idx_), sc), 3))
qq = pc.reindex(idx_).quantile([.1, .25]); low10 = pc.reindex(idx_) <= qq.iloc[0]; print('  예측 하위 10%일의 출력제어 비율', round(float(Y[low10].mean()), 3), ' 나머지', round(float(Y[~low10].mean()), 3), ' 전체', round(float(Y.mean()), 3))
ds = list(idx_); rows = []
for s in range(len(ds) - 2):
    B = ds[s:s + 3]
    if (B[2] - B[0]).days != 2: continue
    wk = [d for d in B if OFF[d] == 1]; rdd = F2_[F2_.hour.isin([12, 13, 14])].groupby('date').rad_mean.mean()
    pk = {'첫날 고정': B[0], '주말 우선': wk[0] if wk else B[0], '일사 최대': max(B, key=lambda d: rdd[d]), '모델': min(B, key=lambda d: pc[d]), '완벽(실제 수요-태양광)': min(B, key=lambda d: act[d])}
    for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, 제어일=cday[d] * 100, 창제어일=cwin[d] * 100, 창제어시간=cwh[d]))
Rc = pd.DataFrame(rows); print(Rc.groupby('방식')[['제어일', '창제어일', '창제어시간']].mean().round(1).to_string()); Wc = Rc.pivot(index='start', columns='방식', values='창제어시간'); Wd = Rc.pivot(index='start', columns='방식', values='창제어일')
for base in ['첫날 고정', '주말 우선', '일사 최대']: print(f'  모델-{base}: 11~14시 제어 시간 비중 +{(Wc["모델"] - Wc[base]).mean():.1f}%p CI {blk((Wc["모델"] - Wc[base]).values)}, 제어일 +{(Wd["모델"] - Wd[base]).mean():.1f}%p CI {blk((Wd["모델"] - Wd[base]).values)}')
Rc.to_csv(f'{OG}/rob_curtail.csv', index=False, encoding='utf-8-sig')
# Holm
ks = list(PV_); ps = np.array([PV_[k] for k in ks]); ord_ = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
for i, j_ in enumerate(ord_): run = max(run, min(1, (len(ps) - i) * ps[j_])); adj[j_] = run
print('\n[Holm 보정 p]', {k: float(f'{a:.2g}') for k, a in zip(ks, adj)}); print('done')
