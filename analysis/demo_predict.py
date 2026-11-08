"""시연용 실제 예측: 내일(1일 전 모델)과 모레(2일 전 모델)의 낮 SMP 0원 이하 가능성을 계산해 시연/demo_data.json에 저장한다.
사용: python demo_predict.py [기준일 YYYY-MM-DD]   (기준일 생략 시 오늘)
최근 SMP는 raw/smp_recent.csv(수집/smp_recent.py)에서 읽는다."""
import os, sys, json, subprocess, datetime as dt
A_ = os.path.dirname(os.path.abspath(__file__))
TODAY = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
T0 = dt.date.fromisoformat(TODAY); TARGETS = [T0 + dt.timedelta(days=1), T0 + dt.timedelta(days=2)]; WIN0 = T0 - dt.timedelta(days=9)
src = open(f'{A_}/smp_model.py', encoding='utf8').read().split('\nout = {}')[0].replace('__file__', repr(f'{A_}/smp_model.py'))
TAG = "dem = pd.read_csv(f'{R}/demand_all.csv'"
assert TAG in src
src = src.replace(TAG, "_r = pd.read_csv(f'{R}/smp_recent.csv', parse_dates=['date']).set_index('date'); smp = pd.concat([smp, _r[_r.index > smp.index.max()]]).sort_index(); smp = smp.reindex(pd.date_range(smp.index.min(), TARGETS[1]))\n" + TAG)
src = src.replace("D1, D2 = daily(F1), daily(F2)", "D1, D2 = None, None")   # 아래에서 최근 예보를 붙인 뒤 계산
exec(src)
from sklearn.linear_model import LogisticRegression
LAT = "33.5141,33.2938,33.3868,33.2461"; LON = "126.5297,126.1628,126.8802,126.5653"
V = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,cloud_cover".split(',')
def recent(lag):
    f = f'{S}/wx/recent_d{lag}.json'; pv = ','.join(f'{v}_previous_day{lag}' for v in V)
    subprocess.run(['curl', '-s', '-m', '120', '-o', f, f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pv}&start_date=2026-11-01&end_date={TARGETS[1]}&timezone=Asia%2FSeoul'], check=True)
    return wx([f], f'_previous_day{lag}')
W1, W2 = recent(1), recent(2)
F1a = pd.concat([F1, W1[W1.index > F1.index.max()]]); F2a = pd.concat([F2, W2[W2.index > F2.index.max()]])
print('예보 끝', W1.dropna(subset=['ws_mean']).index.max(), W2.dropna(subset=['ws_mean']).index.max())
D1, D2 = daily(F1a), daily(F2a)
X = AB + WX + CALS + LAGS
known = L.midmin.notna()
def fit(D):
    t = D[(D.index >= START) & known.reindex(D.index)]; q = t.index.to_period('Q'); ps, ys = [], []
    for k in q.unique():
        ps.append(clf(t[q != k], X).predict_proba(t[q == k][X])[:, 1]); ys.append(t[q == k].y.values)
    lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    pl = LogisticRegression(C=1e6).fit(lo(np.concatenate(ps)), np.concatenate(ys)); m = clf(t, X)
    return lambda b: pl.predict_proba(lo(m.predict_proba(b[X])[:, 1]))[:, 1], t.index.max()
TIER = lambda p: '높음' if p >= .4 else ('보통' if p >= .1 else '낮음')
# 11월 사후 대조: 학습은 10월 말까지
cut = pd.Timestamp('2026-10-31'); f2o, _ = fit(D2[D2.index <= cut]); nov = D2[(D2.index > cut) & known.reindex(D2.index).fillna(False)]
nov_rows = [dict(date=str(i.date()), p=round(float(p), 3), tier=TIER(p), actual=int(y_), midmin=float(m_)) for i, p, y_, m_ in zip(nov.index, f2o(nov), nov.y, nov.midmin)]
print('11월 대조', nov_rows)
# 실제 예측: 알려진 모든 라벨로 학습
f1, last1 = fit(D1); f2, last2 = fit(D2); out = []
for tgt, f, D, lead in [(TARGETS[0], f1, D1, 1), (TARGETS[1], f2, D2, 2)]:
    b = D[D.index == pd.Timestamp(tgt)]
    if b.empty: print('예보 없음', tgt); continue
    p = float(f(b)[0]); r = b.iloc[0]
    out.append(dict(date=str(tgt), lead=lead, p=round(p, 3), tier=TIER(p), dow='월화수목금토일'[tgt.weekday()], offday=int(r.offday), pA=round(float(r.pA_mid), 2), pB=round(float(r.pB_mid), 2), rad=round(float(r.rad_mid)), ws=round(float(r.ws_mid) / 3.6, 1), cloud=round(float(r.cloud_mid)), temp=round(float(r.temp_mid), 1)))
d = pd.read_csv(f'{A_}/out_smp/daily_calibrated.csv', parse_dates=['date']).set_index('date')
hist = [dict(date=str(i.date()), tier=TIER(x.p_cal), actual=int(x.y)) for i, x in d['2026-09-01':].iterrows()]
tiers = {}
for k, g in d.assign(t=d.p_cal.map(TIER)).groupby('t'): tiers[k] = dict(days=len(g), hit=int(g.y.sum()), rate=round(float(g.y.mean()), 3))
os.makedirs(f'{A_}/시연', exist_ok=True)
data = dict(asof=TODAY, train_last=str(last2.date()), forecasts=out, november=nov_rows, history=hist, tiers=tiers, smp_last=str(L.midmin.dropna().index.max().date()))
json.dump(data, open(f'{A_}/시연/demo_data.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps({k: data[k] for k in ['asof', 'train_last', 'forecasts', 'tiers', 'smp_last']}, ensure_ascii=False))
