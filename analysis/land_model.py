"""육지 출력제어 발생일 예측 가능성 (전남·광주 4개 지점 2일 전 예보, 월별 재학습)"""
import os, json, glob, warnings
import numpy as np, pandas as pd, holidays
from sklearn.ensemble import HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap, f1_score
warnings.filterwarnings('ignore')
A = os.path.dirname(os.path.abspath(__file__))
fr = []
for f in sorted(glob.glob(f'{A}/wx/land/d2_*.json')):
    o = []
    for x in json.load(open(f)):
        h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); h = h.drop(columns='time').set_index('ts'); h.columns = [c.replace('_previous_day2', '') for c in h.columns]; o.append(h)
    fr.append(sum(o) / len(o))
W = pd.concat(fr).sort_index().rename(columns={'temperature_2m': 'temp', 'relative_humidity_2m': 'hum', 'precipitation': 'rain', 'wind_speed_10m': 'ws', 'shortwave_radiation': 'rad', 'cloud_cover': 'cloud'}).dropna()
W['date'] = (W.index - pd.Timedelta(hours=1)).normalize(); W['hour'] = W.index.hour.where(W.index.hour != 0, 24); m = W[W.hour.between(10, 16)]; g = m.groupby('date'); a = W.groupby('date')
D = pd.DataFrame({'rad_mid': g.rad.mean(), 'cloud_mid': g.cloud.mean(), 'ws_mid': g.ws.mean(), 'temp_mid': g.temp.mean(), 'hum_mid': g.hum.mean(), 'rain_mid': g.rain.sum(), 'temp_day': a.temp.mean(), 'ws_day': a.ws.mean()})
kr = holidays.KR(years=range(2024, 2027)); D['month'] = D.index.month; D['doy'] = D.index.dayofyear; D['dow'] = D.index.dayofweek; D['holiday'] = [int(x.date() in kr) for x in D.index]; D['offday'] = ((D.dow >= 5) | (D.holiday == 1)).astype(int)
D['cdd'] = (D.temp_day - 22).clip(lower=0); D['hdd'] = (12 - D.temp_day).clip(lower=0)
cs, cw = pd.read_csv(f'{A}/raw/land_curt_solar.csv'), pd.read_csv(f'{A}/raw/land_curt_wind.csv')
ds = pd.to_datetime(cs.iloc[:, 0]); dw = pd.to_datetime(cw.iloc[:, 0]); days = set(ds) | set(dw)
D['y'] = [int(x in days) for x in D.index]; D = D[D.index >= '2024-03-01']
D['cnt14'] = D.y.shift(3).rolling(14).sum(); D['y_l3'] = D.y.shift(3)
print('연도별 발생일', D.groupby(D.index.year).y.agg(['sum', 'count']).to_dict('index'), '| 파일 범위', min(days).date(), max(days).date())
print('전남 포함 비중: 태양광 %.2f 풍력 %.2f' % (cs.iloc[:, 1].astype(str).str.contains('전남').mean(), cw.iloc[:, 1].astype(str).str.contains('전남').mean()))
hc = [f'{i}시' for i in range(1, 25)]
for nm, d in [('태양광', cs), ('풍력', cw)]:
    v = d[hc].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0) if s.dtype != object else (s.astype(str) != '0').astype(int)); v = (v > 0).sum(); print(' ', nm, '제어 시간 상위', v.sort_values(ascending=False).head(6).to_dict())
CAL = ['month', 'doy', 'dow', 'offday', 'holiday']; WX = ['rad_mid', 'cloud_mid', 'ws_mid', 'temp_mid', 'hum_mid', 'rain_mid', 'temp_day', 'ws_day', 'cdd', 'hdd']; LAG = ['cnt14', 'y_l3']
SETS = {'날씨+달력+과거 발생': WX + CAL + LAG, '날씨+달력': WX + CAL, '달력+과거 발생': CAL + LAG, '달력만': CAL}
clf = lambda t, X: HC(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=10, random_state=0).fit(t[X], t.y)
T0, T1 = '2025-07-01', '2026-06-30'; te = D[(D.index >= T0) & (D.index <= T1)]; y = te.y.values
print(f'시험 {T0}~{T1}: {len(te)}일 중 {y.sum()}일')
keep = {}
for nm, X in SETS.items():
    rows = []
    for per in pd.period_range(T0, T1, freq='M'):
        t = D[D.index < per.start_time].dropna(subset=LAG); b = D[D.index.to_period('M') == per]; q = t.index.to_period('Q'); ps, ys = [], []
        for k in q.unique():
            if t[q != k].y.nunique() < 2: continue
            ps.append(clf(t[q != k], X).predict_proba(t[q == k][X])[:, 1]); ys.append(t[q == k].y.values)
        p0, y0 = np.concatenate(ps), np.concatenate(ys); ts = np.arange(.05, .7, .025); th = float(ts[int(np.argmax([f1_score(y0, (p0 >= x).astype(int)) for x in ts]))])
        rows.append(pd.DataFrame({'p': clf(t, X).predict_proba(b[X])[:, 1], 'th': th}, index=b.index))
    w = pd.concat(rows).reindex(te.index); yh = (w.p >= w.th).astype(int).values; tp = int((y & yh).sum()); keep[nm] = w
    print(f' {nm}: AUC {auc(y, w.p):.3f} PR {ap(y, w.p):.3f} days {yh.sum()} prec {tp / max(yh.sum(), 1):.3f} rec {tp / y.sum():.3f} F1 {f1_score(y, yh):.3f}')
for nm, m in [('주말·공휴일', te.offday.values.astype(int)), ('봄(2~5월)', te.month.isin([2, 3, 4, 5]).astype(int).values), ('봄·가을', te.month.isin([2, 3, 4, 5, 10, 11]).astype(int).values), ('3일 전 발생', te.y_l3.fillna(0).astype(int).values)]:
    tp = int((y & m).sum()); print(f' 규칙 {nm}: days {m.sum()} prec {tp / max(m.sum(), 1):.3f} rec {tp / y.sum():.3f} F1 {f1_score(y, m):.3f}')
rng = np.random.default_rng(0); n = len(y); pw = keep['날씨+달력+과거 발생'].p.values; pc = keep['달력+과거 발생'].p.values; A1, Df = [], []
for _ in range(2000):
    s = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(x + np.arange(7)) % n for x in s])[:n]
    if y[i].min() == y[i].max(): continue
    A1.append(auc(y[i], pw[i])); Df.append(A1[-1] - auc(y[i], pc[i]))
print(' AUC CI', np.round(np.percentile(A1, [2.5, 97.5]), 3).tolist(), ' 날씨 추가 효과 CI', np.round(np.percentile(Df, [2.5, 97.5]), 3).tolist())
print(' 주말 비중', round(float((y & te.offday.values).sum() / y.sum()), 3), ' 월별 실제', te.groupby(te.index.month).y.sum().to_dict())
