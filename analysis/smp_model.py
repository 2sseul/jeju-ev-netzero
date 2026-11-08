"""② 제주 낮 SMP<=0 일 예측(1일·2일 전 예보, 월별 walk-forward) + ③ 고정 할인 규칙 비교"""
import os, json, glob, warnings, unicodedata as ud
import numpy as np, pandas as pd, holidays
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap, f1_score
warnings.filterwarnings('ignore')
S = os.path.dirname(os.path.abspath(__file__)); R = f'{S}/raw'; P = os.path.dirname(S)
ALL = [os.path.join(r, f) for r, _, fs in os.walk(P) for f in fs if f.endswith('.csv')]
def find(*k):
    h = [a for a in ALL if all(x in ud.normalize('NFC', a) for x in k)]; assert len(h) == 1, (k, h); return h[0]
SITES = ['jeju', 'gosan', 'seongsan', 'seogwipo']
REN = {'temperature_2m': 'temp', 'relative_humidity_2m': 'hum', 'precipitation': 'rain', 'wind_speed_10m': 'ws', 'shortwave_radiation': 'rad', 'cloud_cover': 'cloud'}
def wx(files, suf):
    fr = []
    for f in files:
        o = []
        for s, x in zip(SITES, json.load(open(f))):
            h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); h = h.drop(columns='time').set_index('ts'); h.columns = [c.replace(suf, '') for c in h.columns]
            r = np.deg2rad(h.wind_direction_10m); h['wdsin'] = np.sin(r); h['wdcos'] = np.cos(r); h = h.drop(columns='wind_direction_10m').rename(columns=REN)
            h.columns = [f'{c}_{s}' for c in h.columns]; o.append(h)
        fr.append(pd.concat(o, axis=1))
    w = pd.concat(fr).sort_index(); w = w[~w.index.duplicated()]
    for v in ['temp', 'hum', 'rain', 'ws', 'rad', 'cloud']: w[f'{v}_mean'] = w[[f'{v}_{s}' for s in SITES]].mean(axis=1)
    w['ws3_mean'] = w.ws_mean ** 3; return w
F0 = wx([f'{S}/wx/fc.json'], '_previous_day1')
F1 = wx(sorted(glob.glob(f'{S}/wx/ch/d1_*.json')), '_previous_day1'); F2 = wx(sorted(glob.glob(f'{S}/wx/ch/d2_*.json')), '_previous_day2')
print('fc d1', F1.index.min(), F1.index.max(), 'null', int(F1.ws_mean.isna().sum()), '| d2 null', int(F2.ws_mean.isna().sum()), F2.index.min(), F2.index.max())
WC = list(F0.columns)
# 발전 (A·B 학습은 2024-05까지만)
def kpx(f):
    d = pd.read_csv(f, encoding='cp949'); d = d.melt(id_vars=d.columns[0], var_name='h', value_name='v'); d['h'] = d.h.str.replace('시', '').astype(int)
    d['ts'] = pd.to_datetime(d[d.columns[0]]) + pd.to_timedelta(d.h, unit='h'); return d.set_index('ts').v
j = pd.read_csv(f'{R}/gen_jeju_2425.csv', parse_dates=['date']); j['ts'] = j.date + pd.to_timedelta(j.h, unit='h')
sol = kpx(find('한시간단위', '태양광')); wnd = kpx(find('한시간단위', '풍력'))
s2 = j[j.fuel == '태양광'].groupby('ts').mwh.sum(); w2 = j[j.fuel == '풍력'].groupby('ts').mwh.sum()
G = pd.DataFrame({'solar': pd.concat([sol, s2[s2.index > sol.index.max()]]), 'wind': pd.concat([wnd, w2[w2.index > wnd.index.max()]])}).sort_index()
cs = pd.read_csv(find('출력제어횟수', '태양광'), encoding='utf-8-sig'); cw = pd.read_csv(find('출력제어횟수', '풍력'), encoding='cp949'); cw.columns = cs.columns
hr = [f'{i}시' for i in range(1, 25)]
def cl(d, num):
    x = d[['일자'] + hr].melt(id_vars='일자', var_name='h', value_name='v'); x['h'] = x.h.str.replace('시', '').astype(int); x['ts'] = pd.to_datetime(x['일자']) + pd.to_timedelta(x.h, unit='h')
    x['v'] = pd.to_numeric(x.v, errors='coerce').fillna(0) if num else (x.v == '출력제어').astype(float); return x.groupby('ts').v.max()
G['cs'] = cl(cs, False); G['cw'] = cl(cw, True); G[['cs', 'cw']] = G[['cs', 'cw']].fillna(0); G['wp'] = G.wind + G.cw
G = G[(G.index >= '2020-01-01') & (G.index < '2024-06-01')]
G['cap_s'] = G.solar.rolling('365D').quantile(.995).shift(24); G['cap_w'] = G.wp.rolling('365D').quantile(.995).shift(24); G['sn'] = G.solar / G.cap_s; G['wn'] = G.wp / G.cap_w
def cal(d):
    d = d.copy(); d['date'] = (d.index - pd.Timedelta(hours=1)).normalize(); d['hour'] = d.index.hour.where(d.index.hour != 0, 24); d['month'] = d.date.dt.month; d['doy'] = d.date.dt.dayofyear; return d
XA = WC + ['hour', 'month', 'doy']
T = cal(G.join(F0, how='inner')); T = T[T.index >= '2022-07-02'].dropna(subset=WC + ['sn', 'wn'])
reg = lambda t, y: HR(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(t[XA], t[y])
mA, mB = reg(T[T.cs == 0], 'sn'), reg(T, 'wn'); print('A·B 학습', T.index.min(), T.index.max(), len(T))
# SMP·수요
smp = pd.read_csv(f'{R}/smp_jeju.csv', parse_dates=['date']).set_index('date'); hc = [f'c{i}' for i in range(1, 25)]
dem = pd.read_csv(f'{R}/demand_all.csv', parse_dates=['date']).set_index('date')
kr = holidays.KR(years=range(2021, 2028))
L = pd.DataFrame(index=smp.index)
L['midmin'] = smp[[f'c{i}' for i in range(11, 17)]].min(axis=1); L['y'] = (L.midmin <= 0).astype(int)
L['w1114'] = smp[['c12', 'c13', 'c14']].mean(axis=1); L['eve'] = smp[[f'c{i}' for i in range(18, 23)]].mean(axis=1); L['h_le0'] = (smp[hc] <= 0).sum(axis=1); L['h_le0_w'] = (smp[['c12', 'c13', 'c14']] <= 0).sum(axis=1)
L['dem_mid'] = dem[[f'h{i}' for i in range(11, 17)]].mean(axis=1).reindex(L.index).ffill()
# D-3까지 정보만 사용 (1일·2일 전 모두 알 수 있는 값)
L['smin_l3'] = L.midmin.shift(3); L['smean14'] = L.midmin.shift(3).rolling(14).mean(); L['cnt14'] = L.y.shift(3).rolling(14).sum(); L['dem7'] = L.dem_mid.shift(3).rolling(7).mean()
def daily(W):
    d = cal(W.dropna(subset=['ws_mean'])); d['pA'] = mA.predict(d[XA]).clip(0); d['pB'] = mB.predict(d[XA]).clip(0)
    m = d[d.hour.between(10, 16)]; g = m.groupby('date'); a = d.groupby('date')
    o = pd.DataFrame({'pA_mid': g.pA.mean(), 'pB_mid': g.pB.mean(), 'pA_max': g.pA.max(), 'pB_max': g.pB.max(), 'rad_mid': g.rad_mean.mean(), 'cloud_mid': g.cloud_mean.mean(), 'ws_mid': g.ws_mean.mean(),
                      'temp_mid': g.temp_mean.mean(), 'hum_mid': g.hum_mean.mean(), 'rain_mid': g.rain_mean.sum(), 'temp_day': a.temp_mean.mean(), 'ws_day': a.ws_mean.mean()})
    o['pR_mid'] = o.pA_mid + o.pB_mid; o['cdd'] = (o.temp_day - 22).clip(lower=0); o['hdd'] = (12 - o.temp_day).clip(lower=0); o['month'] = o.index.month; o['doy'] = o.index.dayofyear; o['dow'] = o.index.dayofweek
    o['holiday'] = [int(x.date() in kr) for x in o.index]; o['offday'] = ((o.dow >= 5) | (o.holiday == 1)).astype(int)
    return o.join(L, how='inner')
D1, D2 = daily(F1), daily(F2)
CALS = ['month', 'doy', 'dow', 'offday', 'holiday']; LAGS = ['smin_l3', 'smean14', 'cnt14', 'dem7']
AB = ['pA_mid', 'pB_mid', 'pA_max', 'pB_max', 'pR_mid']; WX = ['rad_mid', 'cloud_mid', 'ws_mid', 'temp_mid', 'hum_mid', 'rain_mid', 'temp_day', 'ws_day', 'cdd', 'hdd']
SETS = {'전체(A·B+날씨+달력+지연)': AB + WX + CALS + LAGS, 'A·B+날씨+달력': AB + WX + CALS, '날씨+달력': WX + CALS, '달력+지연(날씨 없음)': CALS + LAGS, '달력만': CALS}
clf = lambda t, X: HC(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=10, random_state=0).fit(t[X], t.y)
START, TEST0, TEST1 = '2024-06-05', '2025-06-01', '2026-10-31'
def thr_of(t, X):
    q = t.index.to_period('Q'); ps, ys = [], []
    for k in q.unique():
        if t[q != k].y.nunique() < 2: continue
        ps.append(clf(t[q != k], X).predict_proba(t[q == k][X])[:, 1]); ys.append(t[q == k].y.values)
    p, y = np.concatenate(ps), np.concatenate(ys); ts = np.arange(.05, .7, .025); return float(ts[int(np.argmax([f1_score(y, (p >= x).astype(int)) for x in ts]))])
def walk(D, X):
    D = D[D.index >= START]; ps = []
    for per in pd.period_range(TEST0, TEST1, freq='M'):
        t = D[D.index < per.start_time]; b = D[D.index.to_period('M') == per]; th = thr_of(t, X)
        ps.append(pd.DataFrame({'p': clf(t, X).predict_proba(b[X])[:, 1], 'thr': th}, index=b.index))
    return pd.concat(ps)
def rep(y, yh, t):
    tp = int((y & yh).sum()); return dict(days=int(yh.sum()), prec=round(tp / max(yh.sum(), 1), 3), rec=round(tp / y.sum(), 3), F1=round(f1_score(y, yh), 3), h0=round(t.h_le0[yh == 1].sum() / t.h_le0.sum(), 3))
out = {}
keep = {}
for lead, D in [('1일 전', D1), ('2일 전', D2)]:
    te = D[(D.index >= TEST0) & (D.index <= TEST1)]; y = te.y.values
    print(f'\n[{lead}] 시험 {te.index.min().date()}~{te.index.max().date()} n={len(te)} pos={y.sum()} base={y.mean():.3f}')
    for nm, X in SETS.items():
        w = walk(D, X); keep[(lead, nm)] = w; p = w.p.reindex(te.index).values; yh = (p >= w.thr.reindex(te.index).values).astype(int); k = int(y.sum()); top = np.zeros(len(p), int); top[np.argsort(-p)[:k]] = 1
        r = dict(AUC=round(auc(y, p), 3), PR=round(ap(y, p), 3), **rep(y, yh, te), rec_topk=round((y & top).sum() / k, 3)); out[f'{lead}|{nm}'] = r; print(' ', nm, r)
te = D1[(D1.index >= TEST0) & (D1.index <= TEST1)]; y = te.y.values; w1 = keep[('1일 전', list(SETS)[0])].reindex(te.index); w2 = keep[('2일 전', list(SETS)[0])].reindex(te.index)
rng = np.random.default_rng(0); n = len(y); A1, A2, Dif = [], [], []
cal_p = keep[('1일 전', '달력+지연(날씨 없음)')].reindex(te.index).p.values
for _ in range(2000):
    s = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(x + np.arange(7)) % n for x in s])[:n]
    if y[i].min() == y[i].max(): continue
    A1.append(auc(y[i], w1.p.values[i])); A2.append(auc(y[i], w2.p.values[i])); Dif.append(A1[-1] - auc(y[i], cal_p[i]))
q = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 3).tolist(); print('\nCI AUC 1일전', q(A1), '2일전', q(A2), '전체-날씨없음 차이', q(Dif))
print('\n[③ 규칙 비교: 2025-06~2026-10, 라벨=낮 SMP<=0]')
yh1 = (w1.p >= w1.thr).astype(int).values; yh2 = (w2.p >= w2.thr).astype(int).values
rules = {'주말·공휴일 고정': te.offday.values.astype(int), '일요일·공휴일': ((te.dow == 6) | (te.holiday == 1)).astype(int).values, '봄·가을(3~5,10,11월)': te.month.isin([3, 4, 5, 10, 11]).astype(int).values, '매일': np.ones(n, int), '모델 1일 전': yh1, '모델 2일 전': yh2}
for nm, m in rules.items():
    s = te[m == 1]; print(' ', nm, rep(y, m, te), '11~14시 평균 SMP %.1f, <=0인 11~14시 시간 %d/%d' % (s.w1114.mean(), s.h_le0_w.sum(), te.h_le0_w.sum()))
print('  (참고) 시험기간 저녁 18~22시 평균 SMP %.1f, 11~14시 전체 평균 %.1f' % (te.eve.mean(), te.w1114.mean()))
print('  평일 비중: <=0일 중 평일 %d/%d' % (((y == 1) & (te.offday == 0)).sum(), y.sum()), ' 모델1일전 평일 포착 %.3f' % (((yh1 == 1) & (y == 1) & (te.offday == 0)).sum() / ((y == 1) & (te.offday == 0)).sum()))
print('  시간대(시험기간 <=0 시간):', {f'{i-1}~{i}시': int((smp.loc[te.index, f"c{i}"] <= 0).sum()) for i in range(10, 18)})
sp = te[(te.index >= '2026-09-05') & (te.index <= '2026-10-31')]; ym = sp.y.values
pol = sp.offday.values.astype(int)
print('\n[2026 가을 할인 기간 9/5~10/31] 일수', len(sp), '주말·공휴일', int(pol.sum()), '낮 SMP<=0일', int(ym.sum()), '그중 주말·공휴일', int((ym & pol).sum()), '일자', [str(x.date()) for x in sp.index[ym == 1]])
m1 = (w1.p >= w1.thr).reindex(sp.index).astype(int).values; print('  모델 1일전', rep(ym, m1, sp), ' 주말·공휴일 규칙', rep(ym, pol, sp))
sp2 = te[(te.index >= '2026-03-01') & (te.index <= '2026-05-31')]; ym2 = sp2.y.values; pol2 = sp2.offday.values.astype(int); m2 = (w1.p >= w1.thr).reindex(sp2.index).astype(int).values
print('[2026 3~5월] 일수', len(sp2), '<=0일', int(ym2.sum()), ' 주말·공휴일 규칙', rep(ym2, pol2, sp2), ' 모델', rep(ym2, m2, sp2))
print('\n월별(1일 전):'); print(te.assign(yh=yh1).groupby([te.index.year, te.index.month]).agg(actual=('y', 'sum'), rec=('yh', 'sum')).T.to_string())
# 2024 라벨 대조: 출력제어일과 SMP
print('\n평균 SMP 비교(2025-06~): 낮SMP<=0일의 예보 pR_mid %.3f, 그 외 %.3f' % (te[te.y == 1].pR_mid.mean(), te[te.y == 0].pR_mid.mean()))
o = f'{S}/out_smp'; os.makedirs(o, exist_ok=True); json.dump(out, open(f'{o}/results.json', 'w'), ensure_ascii=False, indent=1); te.assign(p1=w1.p, thr1=w1.thr, p2=w2.p, thr2=w2.thr).to_csv(f'{o}/daily.csv'); print('saved')
