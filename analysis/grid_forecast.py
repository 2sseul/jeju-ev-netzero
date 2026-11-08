"""제주 전력 수급 예보: 수요·태양광·풍력·순부하 예측과 저가일·충전 응용 평가.

결과를 보기 전에 고정한 설계
- 모델 설정: HistGradientBoosting(250회, 학습률 0.05, 잎 31, 최소 20, L2 1). 튜닝하지 않는다.
- 평가: 월별 재학습 walk-forward. 학습 시작 2022-07.
- 구간(개발 / 확인): 수요 2023-07~2025-05 / 2025-06~2026-09
                    태양광·수요-태양광 2023-07~2024-12 / 2025-01~2025-12
                    풍력·순부하 2023-07~2023-12 / 2024-01~2024-05
- 2일 전 수요 입력의 과거 수요는 3일 전까지(L+1). '공개 지연' 변형은 35일 전까지만 쓴다.
- 순부하 방식(직접 / 합산)은 개발 구간 MAE로 고른다.
주의: 수요 확인 구간은 앞선 빠른 점검에서 비슷한 입력으로 이미 본 적이 있다.
변경 기록: 처음에는 풍력·순부하를 2023-07~2024-06 / 2024-07~2025-02-24로 고정했으나, 1차 실행에서 풍력 모델이 확인 구간에서 무너졌고(R2 -3.76)
원인이 2024-06부터 기록 풍력 최대 출력이 약 250MW에서 100MW 이하로 떨어진 집계 단절로 확인되어 구간을 줄였다. 결과를 본 뒤의 변경이다.
3일 전 예보는 2024-02 무렵부터 있어 충전일 평가는 2024-03부터이며, 모델은 2일 전 입력으로 학습해 각 리드 입력에 적용한다.
"""
import os, json, glob
A_ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A_}/ens_price.py', encoding='utf8').read().split('\nP = {k: walk(v)')[0].replace('__file__', repr(f'{A_}/ens_price.py'))
exec(src)  # W, daily_wx, clf, Lg, PM, CAL, LAG, MODELS, Dk 등
import unicodedata as ud
from sklearn.metrics import r2_score
OG = f'{A_}/out_grid'; os.makedirs(OG, exist_ok=True); R_ = f'{A_}/raw'; PRJ = os.path.dirname(A_)
SITES = ['jeju', 'gosan', 'seongsan', 'seogwipo']
RN = {'temperature_2m': 'temp', 'relative_humidity_2m': 'hum', 'precipitation': 'rain', 'wind_speed_10m': 'ws', 'shortwave_radiation': 'rad', 'cloud_cover': 'cloud'}
ALL = [os.path.join(r, f) for r, _, fs in os.walk(PRJ) for f in fs if f.endswith('.csv')]
def find(*k):
    h = [a for a in ALL if all(x in ud.normalize('NFC', a) for x in k)]; assert len(h) == 1, (k, h); return h[0]

def load_wx(L):
    fs = ([f'{A_}/wx/fc.json'] if L == 1 else []) + sorted(glob.glob(f'{A_}/wx/ch/d{L}_*.json')); parts = []
    for f in fs:
        o = []
        for s, x in zip(SITES, json.load(open(f))):
            h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); h = h.drop(columns='time').set_index('ts'); h.columns = [c.replace(f'_previous_day{L}', '') for c in h.columns]; h = h.apply(pd.to_numeric, errors='coerce')
            r = np.deg2rad(h.wind_direction_10m); h['wdsin'] = np.sin(r); h['wdcos'] = np.cos(r); h = h.drop(columns='wind_direction_10m').rename(columns=RN); h.columns = [f'{c}_{s}' for c in h.columns]; o.append(h)
        parts.append(pd.concat(o, axis=1))
    w = pd.concat(parts).sort_index(kind='mergesort'); w = w[~w.index.duplicated(keep='last')]
    for v in RN.values(): w[f'{v}_mean'] = w[[f'{v}_{s}' for s in SITES]].mean(axis=1)
    w['ws3_mean'] = w.ws_mean ** 3
    return w.dropna(subset=['rad_mean', 'ws_mean'])

# ---------- 전력 데이터 (ts = h시 끝시각) ----------
def kpx(f, name):
    d = pd.read_csv(f, encoding='cp949'); d = d.melt(id_vars=d.columns[0], var_name='h', value_name=name); d['h'] = d.h.str.replace('시', '').astype(int)
    d['ts'] = pd.to_datetime(d[d.columns[0]]) + pd.to_timedelta(d.h, unit='h'); return d.set_index('ts')[name]
sol = kpx(find('한시간단위', '태양광'), 'solar'); wnd = kpx(find('한시간단위', '풍력'), 'wind')
j_ = pd.read_csv(f'{R_}/gen_jeju_2425.csv', parse_dates=['date']); j_['ts'] = j_.date + pd.to_timedelta(j_.h, unit='h')
s2 = j_[j_.fuel == '태양광'].groupby('ts').mwh.sum(); w2 = j_[j_.fuel == '풍력'].groupby('ts').mwh.sum()
cs = pd.read_csv(find('출력제어횟수', '태양광'), encoding='utf-8-sig'); cw = pd.read_csv(find('출력제어횟수', '풍력'), encoding='cp949'); cw.columns = cs.columns; hr_ = [f'{i}시' for i in range(1, 25)]
def cl(d, num):
    x = d[['일자'] + hr_].melt(id_vars='일자', var_name='h', value_name='v'); x['h'] = x.h.str.replace('시', '').astype(int); x['ts'] = pd.to_datetime(x['일자']) + pd.to_timedelta(x.h, unit='h')
    x['v'] = pd.to_numeric(x.v, errors='coerce').fillna(0) if num else (x.v == '출력제어').astype(float); return x.groupby('ts').v.max()
tsx = pd.date_range('2020-01-01 01:00', '2026-11-01 00:00', freq='h'); G = pd.DataFrame(index=tsx)
G['solar'] = pd.concat([sol, s2[s2.index > sol.index.max()]]); G['wind'] = pd.concat([wnd, w2[w2.index > wnd.index.max()]])
WIND_END = pd.Timestamp('2024-06-01'); G.loc[G.index >= WIND_END, 'wind'] = np.nan
G['curt_solar'] = cl(cs, False); G['curt_wind'] = cl(cw, True); G[['curt_solar', 'curt_wind']] = G[['curt_solar', 'curt_wind']].fillna(0); G['wind_pot'] = G.wind + G.curt_wind
Dm = pd.read_csv(f'{R_}/demand_all.csv', parse_dates=['date']); Dm = Dm.melt(id_vars='date', var_name='h', value_name='dem'); Dm['ts'] = Dm.date + pd.to_timedelta(Dm.h.str[1:].astype(int), unit='h'); G['dem'] = Dm.set_index('ts').dem
G['net'] = G.dem - G.solar - G.wind; G['net_s'] = G.dem - G.solar
G['date'] = (G.index - pd.Timedelta(hours=1)).normalize(); G['hour'] = G.index.hour.where(G.index.hour != 0, 24)
kr_ = holidays.KR(years=range(2019, 2028)); G['month'] = G.date.dt.month; G['doy'] = G.date.dt.dayofyear; G['dow'] = G.date.dt.dayofweek; G['holiday'] = G.date.dt.date.map(lambda x: x in kr_).astype(int); G['offday'] = ((G.dow >= 5) | (G.holiday == 1)).astype(int)
OFF = G.groupby('date').offday.first()

def pivot(c): return G.pivot_table(index='date', columns='hour', values=c).asfreq('D')
def same_type(tab, k, n=28):
    o = OFF.reindex(tab.index)
    a = tab.where(o == 1).shift(k).rolling(n, min_periods=3).mean().where(o == 1); b = tab.where(o == 0).shift(k).rolling(n, min_periods=5).mean().where(o == 0); return a.fillna(b)
def put(F, tab, name):
    s = tab.stack(); s.index.names = ['date', 'hour']; F[name] = s.reindex(pd.MultiIndex.from_arrays([F.date, F.hour])).values
PV = {c: pivot(c) for c in ['dem', 'solar', 'wind', 'net', 'net_s']}
capS = G.solar.rolling('365D').quantile(.995); capW = G.wind_pot.rolling('365D').quantile(.995)

def frame(L):
    w = load_wx(L); F = G.join(w, how='inner'); k = L + 1
    for c in ['dem', 'net', 'net_s', 'solar', 'wind']:
        put(F, PV[c].shift(k), f'{c}_lag'); put(F, PV[c].shift(7), f'{c}_lag7'); put(F, same_type(PV[c], k), f'{c}_same')
    put(F, PV['dem'].shift(35), 'dem_lag35'); put(F, same_type(PV['dem'], 35), 'dem_same35')
    put(F, PV['solar'].shift(k).rolling(14, min_periods=5).mean(), 'solar_p14'); put(F, PV['wind'].shift(k).rolling(14, min_periods=5).mean(), 'wind_p14')
    F['cap_s'] = capS.shift(24 * k).reindex(F.index).ffill(); F['cap_w'] = capW.shift(24 * k).reindex(F.index).ffill(); F['solar_n'] = F.solar / F.cap_s; F['wind_n'] = F.wind_pot / F.cap_w
    return F[F.date >= '2022-07-01']
WM = [f'{v}_mean' for v in RN.values()]; WS = [f'{v}_{s}' for v in RN.values() for s in SITES] + [f'{d}_{s}' for d in ['wdsin', 'wdcos'] for s in SITES] + WM + ['ws3_mean']; CALG = ['hour', 'month', 'doy', 'dow', 'offday', 'holiday']
def hgb(t, X, y): return HR(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(t[X], t[y])
def walk_g(F, X, y, a, b, tr=None):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = F[(F.date < per.start_time) & F[y].notna()]
        if tr is not None: t = t[tr(t)]
        c = F[F.date.dt.to_period('M') == per]
        if len(c): out.append(pd.Series(hgb(t, X, y).predict(c[X]), index=c.index))
    return pd.concat(out)
def dci(y, p, q, L=21):
    """일별 평균 절대오차 차이(q - p)의 블록 부트스트랩 CI. 양수면 p가 더 정확."""
    e = pd.DataFrame({'a': (y - p).abs(), 'b': (y - q).abs(), 'd': y.index}); e['d'] = (e.d - pd.Timedelta(hours=1)).dt.normalize(); v = e.dropna().groupby('d')[['a', 'b']].mean(); n = len(v); rng = np.random.default_rng(0); o = []
    for _ in range(2000):
        st = rng.integers(0, n, n // L + 1); i = np.concatenate([(x + np.arange(L)) % n for x in st])[:n]; o.append(v.b.values[i].mean() - v.a.values[i].mean())
    return np.round(np.percentile(o, [2.5, 97.5]), 2).tolist()
def scores(F, y, preds, a, b, nm, tag):
    rows = []; m = (F.date >= a) & (F.date <= b) & F[y].notna(); act = F.loc[m, y]
    for k, p in preds.items():
        p = p.reindex(act.index); ok = p.notna(); e = (act[ok] - p[ok]); rows.append(dict(대상=tag, 구간=nm, 방식=k, 시간수=int(ok.sum()), MAE=round(e.abs().mean(), 2), 정규화MAE=round(100 * e.abs().mean() / act[ok].mean(), 2), MAPE=round(100 * (e.abs() / act[ok].where(act[ok] > 50)).mean(), 2), R2=round(r2_score(act[ok], p[ok]), 3)))
    return rows
def daily_shape(F, y, p, a, b):
    d = pd.DataFrame({'y': F[y], 'p': p.reindex(F.index), 'date': F.date, 'hour': F.hour}); d = d[(d.date >= a) & (d.date <= b)].dropna(); dd = d.groupby('date').size(); d = d[d.date.isin(dd[dd == 24].index)]
    mi = d[d.hour.between(11, 16)].groupby('date'); ev = d[d.hour.between(18, 22)].groupby('date'); o = pd.DataFrame({'min_y': mi.y.min(), 'min_p': mi.p.min(), 'pk_y': ev.y.max(), 'pk_p': ev.p.max()}); o['r_y'] = o.pk_y - o.min_y; o['r_p'] = o.pk_p - o.min_p; return o

T0, TD = '2023-07', '2026-09'
F2 = frame(2); RES = []; print('[준비] 2일 전 행', len(F2), F2.date.min().date(), F2.date.max().date(), flush=True)
# ---------- 1. 수요 ----------
XF = WM + CALG + ['dem_lag', 'dem_lag7', 'dem_same']; XD = WM + CALG + ['dem_lag35', 'dem_same35']; XC = WM + CALG; XN = CALG + ['dem_lag', 'dem_lag7', 'dem_same']
Pd = {'모델 (3일 전 수요까지)': walk_g(F2, XF, 'dem', T0, TD), '모델 (35일 공개 지연)': walk_g(F2, XD, 'dem', T0, TD), '날씨+달력만': walk_g(F2, XC, 'dem', T0, TD), '날씨 없음': walk_g(F2, XN, 'dem', T0, TD), '같은 요일유형 패턴': F2.dem_same, '7일 전': F2.dem_lag7}
for nm, a, b in [('개발', '2023-07-01', '2025-05-31'), ('확인', '2025-06-01', '2026-09-30')]:
    RES += scores(F2, 'dem', Pd, a, b, nm, '수요'); m = (F2.date >= a) & (F2.date <= b); y = F2.dem[m]; p = Pd['모델 (3일 전 수요까지)'].reindex(y.index)
    print(f' 수요 {nm} CI 패턴 대비', dci(y, p, F2.dem_same[m]), ' 날씨 없음 대비', dci(y, p, Pd['날씨 없음'].reindex(y.index)), ' 공개 지연이 패턴 대비', dci(y, Pd['모델 (35일 공개 지연)'].reindex(y.index), F2.dem_same[m]), flush=True)
    for hh, lb in [([12, 13, 14], '낮 11~14시'), ([19, 20, 21], '저녁 18~21시')]:
        q = m & F2.hour.isin(hh); print(f'  {nm} {lb} MAPE', {k: round(100 * float(((F2.dem[q] - v.reindex(F2.dem[q].index)).abs() / F2.dem[q]).mean()), 2) for k, v in Pd.items()})
for nm, a, b in [('개발', '2023-07-01', '2025-05-31'), ('확인', '2025-06-01', '2026-09-30')]:
    d = daily_shape(F2, 'dem', Pd['모델 (3일 전 수요까지)'], a, b); print(f'  {nm} 저녁 피크 MAPE', round(100 * ((d.pk_y - d.pk_p).abs() / d.pk_y).mean(), 2), ' 일수', len(d))
print(pd.DataFrame(RES).to_string(index=False), flush=True)
# ---------- 2. 태양광·풍력 ----------
XS = WS + ['hour', 'month', 'doy']
Ps = walk_g(F2, XS, 'solar_n', T0, '2026-10', tr=lambda t: t.curt_solar == 0) * F2.cap_s; Pw = walk_g(F2, XS, 'wind_n', T0, '2024-05') * F2.cap_w
Pz = {'모델': Ps, '14일 패턴': F2.solar_p14, '같은 요일유형 패턴': F2.solar_same}; r1 = []
for nm, a, b in [('개발', '2023-07-01', '2024-12-31'), ('확인', '2025-01-01', '2025-12-31')]:
    Fd = F2[F2.hour.between(9, 17)]; r1 += scores(Fd, 'solar', Pz, a, b, nm, '태양광(9~17시)'); m = (Fd.date >= a) & (Fd.date <= b) & Fd.solar.notna(); print(f' 태양광 {nm} CI 패턴 대비', dci(Fd.solar[m], Ps.reindex(Fd.index[m]), Fd.solar_p14[m]))
for nm, a, b in [('개발', '2023-07-01', '2023-12-31'), ('확인', '2024-01-01', '2024-05-31')]:
    r1 += scores(F2, 'wind', {'모델': Pw, '14일 패턴': F2.wind_p14}, a, b, nm, '풍력'); m = (F2.date >= a) & (F2.date <= b) & F2.wind.notna(); print(f' 풍력 {nm} CI 패턴 대비', dci(F2.wind[m], Pw.reindex(F2.index[m]), F2.wind_p14[m]))
print(pd.DataFrame(r1).to_string(index=False), flush=True); RES += r1
# ---------- 3. 순부하 ----------
XNET = WS + CALG + ['net_lag', 'net_lag7', 'net_same']; XNS = WS + CALG + ['net_s_lag', 'net_s_lag7', 'net_s_same']
Pn_dir = walk_g(F2, XNET, 'net', T0, '2024-05'); Pn_sum = Pd['모델 (3일 전 수요까지)'] - Ps - Pw; Pns = walk_g(F2, XNS, 'net_s', T0, '2025-12')
Pn = {'직접 예측': Pn_dir, '수요-태양광-풍력 합산': Pn_sum, '같은 요일유형 패턴': F2.net_same, '7일 전': F2.net_lag7}; r2_ = []
for nm, a, b in [('개발', '2023-07-01', '2023-12-31'), ('확인', '2024-01-01', '2024-05-31')]: r2_ += scores(F2, 'net', Pn, a, b, nm, '순부하')
for nm, a, b in [('개발', '2023-07-01', '2024-12-31'), ('확인', '2025-01-01', '2025-12-31')]: r2_ += scores(F2, 'net_s', {'직접 예측': Pns, '같은 요일유형 패턴': F2.net_s_same, '7일 전': F2.net_s_lag7}, a, b, nm, '수요-태양광')
print(pd.DataFrame(r2_).to_string(index=False)); RES += r2_
dv = pd.DataFrame(r2_); dv = dv[(dv.대상 == '순부하') & (dv.구간 == '개발') & dv.방식.isin(['직접 예측', '수요-태양광-풍력 합산'])]; pick = dv.sort_values('MAE').방식.iloc[0]; print(' 개발 MAE 기준 선택:', pick, flush=True); Pbest = Pn[pick]
shape = []
for nm, a, b in [('개발', '2023-07-01', '2023-12-31'), ('확인', '2024-01-01', '2024-05-31')]:
    m = (F2.date >= a) & (F2.date <= b) & F2.net.notna(); print(f' 순부하 {nm} CI 패턴 대비', dci(F2.net[m], Pbest.reindex(F2.index[m]), F2.net_same[m]))
    for k, p in [('모델', Pbest), ('패턴', F2.net_same)]:
        d = daily_shape(F2, 'net', p, a, b); rec = dict(구간=nm, 방식=k, 일수=len(d), 낮저점평균=round(d.min_y.mean(), 1), 낮저점MAE=round((d.min_y - d.min_p).abs().mean(), 1), 저녁피크평균=round(d.pk_y.mean(), 1), 저녁피크MAE=round((d.pk_y - d.pk_p).abs().mean(), 1), 상승폭평균=round(d.r_y.mean(), 1), 상승폭MAE=round((d.r_y - d.r_p).abs().mean(), 1), 상승폭상관=round(d.r_y.corr(d.r_p), 3)); shape.append(rec); print('  ', rec)
pd.DataFrame(shape).to_csv(f'{OG}/net_shape.csv', index=False, encoding='utf-8-sig'); pd.DataFrame(RES).to_csv(f'{OG}/forecast_scores.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'dem': F2.dem, 'p_dem': Pd['모델 (3일 전 수요까지)'], 'p_dem_delay': Pd['모델 (35일 공개 지연)'], 'solar': F2.solar, 'p_solar': Ps, 'wind': F2.wind, 'p_wind': Pw, 'net': F2.net, 'p_net': Pbest, 'net_same': F2.net_same}).dropna(subset=['p_dem']).round(2).to_csv(f'{OG}/hourly_d2.csv')
# ---------- 4. 저가일 판별에 도움이 되는가 ----------
pf = pd.DataFrame({'d': Pd['모델 (3일 전 수요까지)'], 's': Ps, 'date': F2.date, 'hour': F2.hour}); pf = pf[pf.hour.isin([12, 13, 14])].groupby('date')[['d', 's']].mean()
Dg = Dk.join(pf.rename(columns={'d': 'pd_mid', 's': 'ps_mid'})); Dg['pn_mid'] = Dg.pd_mid - Dg.ps_mid
V4 = {'앙상블(현재)': [], '+예측 수요': ['pd_mid'], '+예측 수요·태양광': ['pd_mid', 'ps_mid', 'pn_mid']}; P4 = {}
for k, ex in V4.items():
    rows = []
    for per in pd.period_range('2025-06', '2026-09', freq='M'):
        t = Dg[(Dg.index < per.start_time) & Dg.y.notna()]; b = Dg[(Dg.index.to_period('M') == per) & Dg.y.notna()]
        rows.append(pd.Series(np.mean([clf(t, list(PM[m].columns) + CAL + LAG + ex).predict_proba(b[list(PM[m].columns) + CAL + LAG + ex])[:, 1] for m in MODELS], axis=0), index=b.index))
    P4[k] = pd.concat(rows)
print('\n[저가일 판별, 2일 전]')
for nm, a, b in [('개발', '2025-06-01', '2026-05-31'), ('확인', '2026-06-01', '2026-09-30'), ('전체', '2025-06-01', '2026-09-30')]:
    for k, s in P4.items():
        x = s[a:b]; y = Lg.y.reindex(x.index).astype(int); print(' ', nm, k, 'n', len(x), 'pos', int(y.sum()), 'AUC', round(auc(y, x), 3), 'PR', round(ap(y, x), 3))
x = P4['앙상블(현재)']; y = Lg.y.reindex(x.index).astype(int).values; n = len(y); rng = np.random.default_rng(0)
for k in ['+예측 수요', '+예측 수요·태양광']:
    o = []
    for _ in range(1000):
        st = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(v + np.arange(7)) % n for v in st])[:n]
        if y[i].min() == y[i].max(): continue
        o.append(ap(y[i], P4[k].values[i]) - ap(y[i], x.values[i]))
    print(' ', k, 'PR 차이 CI', np.round(np.percentile(o, [2.5, 97.5]), 3).tolist(), flush=True)
pd.DataFrame(P4).to_csv(f'{OG}/lowprice_probs.csv')
# ---------- 5. 순부하 예측으로 충전일 고르기 (3일 중 하루, 11~14시) ----------
def walk_cross(FL, X, y, a, b):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = F2[(F2.date < per.start_time) & F2[y].notna()]; c = FL[FL.date.dt.to_period('M') == per]
        if len(c): out.append(pd.Series(hgb(t, X, y).predict(c[X]), index=c.index))
    return pd.concat(out)
Pnl = {2: walk_cross(F2, XNS, 'net_s', '2024-03', '2025-12')}
for L in (1, 3):
    FL = frame(L); Pnl[L] = walk_cross(FL, XNS, 'net_s', '2024-03', '2025-12'); print(' 리드', L, '완료', flush=True)
win_ = [12, 13, 14]; ds = {L: pd.DataFrame({'p': v, 'date': G.date.reindex(v.index), 'hour': G.hour.reindex(v.index)}) for L, v in Pnl.items()}
pm = {L: d[d.hour.isin(win_)].groupby('date').p.mean() for L, d in ds.items()}
actS = PV['net_s'][win_].mean(axis=1); actSol = PV['solar'][win_].mean(axis=1); actDem = PV['dem'][win_].mean(axis=1); patt = same_type(PV['net_s'], 4)[win_].mean(axis=1); smp = J.copy(); smp.columns = range(1, 25); smpw = smp[win_].mean(axis=1)
def bundles(a, b):
    days = [x for x in pd.date_range(a, b) if pd.notna(actS.get(x)) and all(x in pm[L].index for L in pm)]; rows = []
    for s in range(len(days) - 2):
        B = days[s:s + 3]
        if (B[2] - B[0]).days != 2: continue
        pk = {'첫날 고정': B[0], '주말·공휴일 우선': next((d for d in B if OFF[d] == 1), B[0]), '과거 패턴': min(B, key=lambda d: patt.get(d, 1e9)), '순부하 예측': min(zip(B, (1, 2, 3)), key=lambda z: pm[z[1]][z[0]])[0], '완벽한 예측': min(B, key=lambda d: actS[d])}
        for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, net=actS[d], sol=actSol[d], dem=actDem[d], smp=smpw.get(d, np.nan)))
    return pd.DataFrame(rows)
def bci(u, v, L=21):
    n = len(u); rng = np.random.default_rng(0); o = []
    for _ in range(2000):
        st = rng.integers(0, n, n // L + 1); i = np.concatenate([(x + np.arange(L)) % n for x in st])[:n]; o.append(u[i].mean() - v[i].mean())
    return np.round(np.percentile(o, [2.5, 97.5]), 1).tolist()
print('\n[충전일 선택: 충전 시간의 실제 (수요-태양광), MW]'); rc = []
for nm, a, b in [('개발', '2024-03-01', '2024-12-31'), ('확인', '2025-01-01', '2025-12-31'), ('전체', '2024-03-01', '2025-12-31')]:
    R = bundles(a, b); W_ = R.pivot(index='start', columns='방식', values='net'); So = R.pivot(index='start', columns='방식', values='sol'); De = R.pivot(index='start', columns='방식', values='dem'); Sm = R.pivot(index='start', columns='방식', values='smp')
    f, t = W_['첫날 고정'].mean(), W_['완벽한 예측'].mean()
    for k in W_.columns: rec = dict(구간=nm, 방식=k, 묶음=len(W_), 순부하=round(W_[k].mean(), 1), 수요=round(De[k].mean(), 1), 태양광=round(So[k].mean(), 1), 태양광비중=round(100 * So[k].mean() / De[k].mean(), 1), SMP=round(Sm[k].mean(), 1), 달성률=round(100 * (f - W_[k].mean()) / (f - t), 1)); rc.append(rec); print(' ', rec)
    for base in ['첫날 고정', '주말·공휴일 우선', '과거 패턴']: print(f'  순부하 예측 - {base} 감소 CI', bci(W_[base].values, W_['순부하 예측'].values), ' 42일', bci(W_[base].values, W_['순부하 예측'].values, 42))
    if nm == '전체':
        g = (W_['첫날 고정'] - W_['순부하 예측']); print('  계절별 감소', g.groupby(g.index.quarter).mean().round(1).to_dict(), ' 분기 주말우선 대비', (W_['주말·공휴일 우선'] - W_['순부하 예측']).groupby(g.index.quarter).mean().round(1).to_dict(), ' 모델이 첫날과 다른 날 고른 비율', round(float((g != 0).mean()), 3))
pd.DataFrame(rc).to_csv(f'{OG}/charge_netload.csv', index=False, encoding='utf-8-sig')
# ---------- 6. 참여율별 계통 효과 (단순 계산) ----------
EV, KWH, COMPLY = 49335, 43.0 / 5, .5; d = daily_shape(F2, 'net', F2.net, '2023-07-01', '2024-05-31'); print('\n[계통 효과] 낮 저점 평균', round(d.min_y.mean(), 1), '저녁 피크 평균', round(d.pk_y.mean(), 1), '상승폭 평균', round(d.r_y.mean(), 1)); rr = []
for part in [.01, .05, .1, .2, .5]:
    mw = EV * part * KWH * COMPLY / 1000 / 3; rr.append(dict(참여율=part, 이동MW=round(mw, 1), 상승폭대비=round(100 * 2 * mw / d.r_y.mean(), 1)))
print(pd.DataFrame(rr).to_string(index=False)); pd.DataFrame(rr).to_csv(f'{OG}/ramp_effect.csv', index=False, encoding='utf-8-sig')
# ---------- 7. 라벨 검증: SMP 0원 이하인 날의 수급 ----------
tw = F2[(F2.date < '2024-06-01') & F2.wind_n.notna()]; capfix = capW[:'2024-05-31'].dropna().iloc[-1]; wm = hgb(tw, XS, 'wind_n'); F2['wind_est'] = wm.predict(F2[XS]) * capfix
print('\n[기록 풍력과 추정 가능량, 낮 11~14시 평균 MW]', {str(k): [round(float(v.wind.mean()), 1), round(float(v.wind_est.mean()), 1)] for k, v in F2[F2.hour.isin(win_) & (F2.date < '2024-06-01') & (F2.date >= '2023-07-01')].groupby(F2.date.dt.year)})
Ldf = F2[F2.hour.isin(win_)].groupby('date')[['dem', 'solar', 'wind_est', 'net_s']].mean(); Ldf['y'] = Lg.y.reindex(Ldf.index); Ldf['off'] = OFF.reindex(Ldf.index); Ldf = Ldf['2024-06-05':'2025-12-31'].dropna(); Ldf['net_est'] = Ldf.net_s - Ldf.wind_est; Ldf['share_s'] = Ldf.solar / Ldf.dem; Ldf['share_est'] = (Ldf.solar + Ldf.wind_est) / Ldf.dem
print(f'[라벨 검증] 2024-06-05~2025-12-31, {len(Ldf)}일, 저가일 {int(Ldf.y.sum())}'); print(Ldf.groupby('y')[['dem', 'solar', 'wind_est', 'net_s', 'net_est', 'share_s', 'share_est']].mean().round(2).to_string())
print(' AUC 수요-태양광', round(auc(Ldf.y, -Ldf.net_s), 3), ' 수요-태양광-추정풍력', round(auc(Ldf.y, -Ldf.net_est), 3), ' 재생비중(추정)', round(auc(Ldf.y, Ldf.share_est), 3), ' 수요', round(auc(Ldf.y, -Ldf.dem), 3), ' 태양광', round(auc(Ldf.y, Ldf.solar), 3))
qq = Ldf.net_est.quantile(.1); print(' 추정 순부하 하위 10%의 저가일 비율', round(Ldf.y[Ldf.net_est <= qq].mean(), 3), ' 나머지', round(Ldf.y[Ldf.net_est > qq].mean(), 3), ' 저가일 중 하위 10%에 든 비율', round((Ldf.net_est[Ldf.y == 1] <= qq).mean(), 3)); Ldf.round(2).to_csv(f'{OG}/label_check.csv'); print('done')
