"""다중 예보 앙상블 잉여일 예측, 시간별 SMP 예측, 충전·V2G 시뮬레이션.
개발 구간 2025-06~2026-05에서 입력 구성을 고르고, 2026-06~2026-11-07은 최종 확인용으로 분리한다.
가격 모델은 구성을 하나로 고정해 두 구간에 그대로 적용한다."""
import os, json, glob, warnings
import numpy as np, pandas as pd, holidays
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap, f1_score
warnings.filterwarnings('ignore')
A = os.path.dirname(os.path.abspath(__file__)); O = f'{A}/out_price'; os.makedirs(O, exist_ok=True)
MODELS = ['best_match', 'ecmwf_ifs025', 'gfs_global', 'icon_global', 'gem_global']; VARS = {'temperature_2m': 'temp', 'precipitation': 'rain', 'wind_speed_10m': 'ws', 'shortwave_radiation': 'rad', 'cloud_cover': 'cloud'}
DEV0, DEV1, HOLD0, HOLD1 = '2025-06-01', '2026-05-31', '2026-06-01', '2026-11-07'; START = '2024-06-05'

def load(m, lead):
    fr = []
    for f in sorted(glob.glob(f'{A}/wx/ens/{m}_*.json')):
        s = []
        for x in json.load(open(f)):
            h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); h = h.set_index('ts').drop(columns='time')
            h = h[[f'{v}_previous_day{lead}' for v in VARS]]; h.columns = list(VARS.values()); s.append(h)
        fr.append(sum(s) / len(s))
    w = pd.concat(fr).sort_index(kind='mergesort'); return w[~w.index.duplicated(keep='last')]  # 겹치면 나중 파일(최근 수집)을 쓴다
W = {L: {m: load(m, L) for m in MODELS} for L in (1, 2, 3)}
for L in W: print('lead', L, {m: round(float(W[L][m].ws.notna().mean()), 3) for m in MODELS})
def mean_std(L):
    st = {v: pd.concat([W[L][m][v] for m in MODELS], axis=1) for v in VARS.values()}
    d = pd.DataFrame({f'{v}_ens': st[v].mean(axis=1) for v in st})
    for v in ['rad', 'ws', 'cloud']: d[f'{v}_sd'] = st[v].std(axis=1)
    return d
ENS = {L: mean_std(L) for L in W}
# 가격
hc = [f'c{i}' for i in range(1, 25)]
J = pd.read_csv(f'{A}/raw/smp_jeju.csv', parse_dates=['date']).set_index('date'); Jr = pd.read_csv(f'{A}/raw/smp_recent.csv', parse_dates=['date']).set_index('date'); J = pd.concat([J, Jr[Jr.index > J.index.max()]]).sort_index()[hc]
LD = pd.read_csv(f'{A}/raw/smp_land.csv', parse_dates=['date']).set_index('date')[hc]
kr = holidays.KR(years=range(2021, 2028)); idx = pd.date_range('2024-01-01', '2027-12-31')
Lg = pd.DataFrame(index=idx); jr = J.reindex(idx); lr = LD.reindex(idx); mids = [f'c{i}' for i in range(11, 17)]
Lg['midmin'] = jr[mids].min(axis=1); Lg['y'] = (Lg.midmin <= 0).astype(float).where(Lg.midmin.notna())
Lg['smin4'] = Lg.midmin.shift(4); Lg['smean14'] = Lg.midmin.shift(4).rolling(14).mean(); Lg['cnt14'] = Lg.y.shift(4).rolling(14).sum()
Lg['lmin4'] = lr[mids].min(axis=1).shift(4); Lg['lmean14'] = lr[mids].mean(axis=1).shift(4).rolling(14).mean(); Lg['lcnt14'] = (lr[mids].min(axis=1) <= 0).astype(float).shift(4).rolling(14).sum(); Lg['spread14'] = Lg.smean14 - Lg.lmean14
Lg['dow'] = idx.dayofweek; Lg['month'] = idx.month; Lg['doy'] = idx.dayofyear; Lg['holiday'] = [int(x.date() in kr) for x in idx]; Lg['offday'] = ((Lg.dow >= 5) | (Lg.holiday == 1)).astype(int)
def dates(w): d = w.copy(); d['date'] = (d.index - pd.Timedelta(hours=1)).normalize(); d['hour'] = d.index.hour.where(d.index.hour != 0, 24); return d
def daily_wx(w, suf, cols):
    d = dates(w); m = d[d.hour.between(10, 16)]; g = m.groupby('date')[cols].mean(); g.columns = [f'{c}{suf}' for c in cols]; return g
# ---------- 1. 잉여일 분류 (2일 전) ----------
BM = daily_wx(W[2]['best_match'], '_bm', list(VARS.values()))
EM = daily_wx(ENS[2], '', list(ENS[2].columns))
PM = {m: daily_wx(W[2][m], f'_{m}', list(VARS.values())) for m in MODELS}
Dd = Lg.join(BM).join(EM)
for m in MODELS: Dd = Dd.join(PM[m])
CAL = ['month', 'doy', 'dow', 'offday', 'holiday']; LAG = ['smin4', 'smean14', 'cnt14']; LAND = ['lmin4', 'lmean14', 'lcnt14', 'spread14']
XBM = [c for c in BM.columns]; XEM = [c for c in EM.columns if c.endswith('_ens')]; XSD = [c for c in EM.columns if c.endswith('_sd')]; XALL = [c for m in MODELS for c in PM[m].columns]
SETS = {'기본 예보 1개': XBM + CAL + LAG, '앙상블 평균': XEM + CAL + LAG, '앙상블 평균+분산': XEM + XSD + CAL + LAG, '앙상블 평균+분산+육지 SMP': XEM + XSD + CAL + LAG + LAND, '기본 예보+육지 SMP': XBM + CAL + LAG + LAND, '5개 모델 개별 입력': XALL + CAL + LAG, '달력+과거 가격': CAL + LAG}
clf = lambda t, X: HC(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=10, random_state=0).fit(t[X], t.y.astype(int))
Dk = Dd[Dd.index >= START]
def walk(X, per_model=None):
    rows = []
    for per in pd.period_range(DEV0, HOLD1, freq='M'):
        t = Dk[(Dk.index < per.start_time) & Dk.y.notna()]; b = Dk[(Dk.index.to_period('M') == per) & Dk.y.notna() & (Dk.index <= HOLD1)]
        if b.empty: continue
        if per_model:
            p = np.mean([clf(t, x).predict_proba(b[x])[:, 1] for x in per_model], axis=0)
        else: p = clf(t, X).predict_proba(b[X])[:, 1]
        rows.append(pd.Series(p, index=b.index))
    return pd.concat(rows)
P = {k: walk(v) for k, v in SETS.items()}
P['5개 모델 확률 평균'] = walk(None, [list(PM[m].columns) + CAL + LAG for m in MODELS])
def score(s, a, b):
    t = s[(s.index >= a) & (s.index <= b)]; y = Lg.y.reindex(t.index).astype(int).values; k = int(y.sum()); top = np.zeros(len(t), int); top[np.argsort(-t.values)[:k]] = 1
    return dict(n=len(t), pos=k, AUC=round(auc(y, t), 3), PR=round(ap(y, t), 3), rec_topk=round((y & top).sum() / k, 3))
print('\n[1] 2일 전 잉여일 분류')
tab = []
for k, s in P.items():
    d, h = score(s, DEV0, DEV1), score(s, HOLD0, HOLD1); tab.append(dict(입력=k, 개발_AUC=d['AUC'], 개발_PR=d['PR'], 개발_top=d['rec_topk'], 확인_AUC=h['AUC'], 확인_PR=h['PR'], 확인_top=h['rec_topk'])); print(' ', k, 'DEV', d, 'HOLD', h)
pd.DataFrame(tab).to_csv(f'{O}/classification.csv', index=False, encoding='utf-8-sig')
best = max(P, key=lambda k: score(P[k], DEV0, DEV1)['PR']); print(' 개발 PR 기준 선택:', best)
def cmp_ci(a, b, lo, hi):
    a, b = P[a], P[b]; ix = a[(a.index >= lo) & (a.index <= hi)].index; y = Lg.y.reindex(ix).astype(int).values; n = len(y); rng = np.random.default_rng(0); dA, dP = [], []
    for _ in range(2000):
        s = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(x + np.arange(7)) % n for x in s])[:n]
        if y[i].min() == y[i].max(): continue
        dA.append(auc(y[i], a[ix].values[i]) - auc(y[i], b[ix].values[i])); dP.append(ap(y[i], a[ix].values[i]) - ap(y[i], b[ix].values[i]))
    return np.round(np.percentile(dA, [2.5, 97.5]), 3).tolist(), np.round(np.percentile(dP, [2.5, 97.5]), 3).tolist()
print(' 선택-기본 차이 CI 개발(AUC,PR)', cmp_ci(best, '기본 예보 1개', DEV0, DEV1), '확인', cmp_ci(best, '기본 예보 1개', HOLD0, HOLD1))
pd.DataFrame(P).to_csv(f'{O}/class_probs.csv')

# ---------- 2. 시간별 가격 예측 ----------
pz = jr[hc].copy(); pz.columns = range(1, 25); lz = lr[hc].copy(); lz.columns = range(1, 25)
def hourly_frame(L):
    d = dates(ENS[L]).rename(columns=lambda c: c)
    d['y'] = [pz.at[a, b] if a in pz.index else np.nan for a, b in zip(d.date, d.hour)]
    idxd = d.date; hh = d.hour
    def lagv(tab, k): return [tab.at[a - pd.Timedelta(days=k), b] if (a - pd.Timedelta(days=k)) in tab.index else np.nan for a, b in zip(idxd, hh)]
    prof = pz.shift(4).rolling(14, min_periods=7).mean(); lprof = lz.shift(4).rolling(14, min_periods=7).mean()
    offs = Lg.offday; pw = pz.where(offs.reindex(pz.index) == 1); pn = pz.where(offs.reindex(pz.index) == 0)
    pw4 = pw.shift(4).rolling(28, min_periods=3).mean(); pn4 = pn.shift(4).rolling(28, min_periods=5).mean()
    d['lag4'] = lagv(pz, 4); d['lag7'] = lagv(pz, 7); d['prof'] = [prof.at[a, b] if a in prof.index else np.nan for a, b in zip(idxd, hh)]; d['lprof'] = [lprof.at[a, b] if a in lprof.index else np.nan for a, b in zip(idxd, hh)]
    d['profw'] = [pw4.at[a, b] if a in pw4.index else np.nan for a, b in zip(idxd, hh)]; d['profn'] = [pn4.at[a, b] if a in pn4.index else np.nan for a, b in zip(idxd, hh)]
    for c in CAL + ['smean14', 'cnt14', 'lmean14', 'lcnt14', 'spread14']: d[c] = Lg[c].reindex(idxd).values
    mid = d[d.hour.between(10, 16)].groupby('date')[['rad_ens', 'ws_ens', 'cloud_ens', 'temp_ens']].mean().add_suffix('_mid'); d = d.join(mid, on='date')
    d['same_type'] = np.where(d.offday == 1, d.profw, d.profn); return d
XH = ['temp_ens', 'rain_ens', 'ws_ens', 'rad_ens', 'cloud_ens', 'rad_sd', 'ws_sd', 'cloud_sd', 'hour', 'lag4', 'lag7', 'prof', 'lprof', 'same_type', 'rad_ens_mid', 'ws_ens_mid', 'cloud_ens_mid', 'temp_ens_mid'] + CAL + ['smean14', 'cnt14', 'lmean14', 'lcnt14', 'spread14']
reg = lambda t: HR(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(t[XH], t.y)
PRED = {}
for L in (1, 2, 3):
    fr = hourly_frame(L); fr = fr[fr.date >= START]; rows = []
    for per in pd.period_range(DEV0, HOLD1, freq='M'):
        t = fr[(fr.date < per.start_time) & fr.y.notna()]; b = fr[(fr.date.dt.to_period('M') == per) & fr.y.notna() & (fr.date <= HOLD1)]
        if b.empty: continue
        rows.append(b[['date', 'hour', 'y', 'prof', 'same_type', 'lag7']].assign(p=reg(t).predict(b[XH])))
    PRED[L] = pd.concat(rows); print('lead', L, 'rows', len(PRED[L]), flush=True)
def mets(d):
    o = {}
    for nm, c in [('모델', 'p'), ('14일 평균 패턴', 'prof'), ('같은 요일유형 패턴', 'same_type'), ('7일 전', 'lag7')]:
        x = d.dropna(subset=[c]); o[nm] = round(float((x.y - x[c]).abs().mean()), 2)
    return o
print('\n[2] 시간별 SMP MAE (원/kWh)')
rowm = []
for L in (1, 2, 3):
    for nm, a, b in [('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1)]:
        x = PRED[L][(PRED[L].date >= a) & (PRED[L].date <= b)]; mid = x[x.hour.between(10, 16)]; low = mid[mid.y <= 0]
        rec = dict(lead=L, 구간=nm, **mets(x)); recm = mets(mid); print(' ', L, nm, '전체', mets(x), '낮', recm); rec.update({f'낮_{k}': v for k, v in recm.items()}); rowm.append(rec)
pd.DataFrame(rowm).to_csv(f'{O}/price_mae.csv', index=False, encoding='utf-8-sig')
for L in PRED: PRED[L].to_csv(f'{O}/hourly_pred_d{L}.csv')

# ---------- 3. 충전 시뮬레이션 ----------
def wide(d, c): return d.pivot_table(index='date', columns='hour', values=c).reindex(columns=range(1, 25))
def block3(a):  # a: 일수×24, 끝시각 h 기준. 반환: 시작 인덱스별 3시간 평균 (열 s는 s+1,s+2,s+3시 라벨 = s~s+3시)
    return (a[:, :-2] + a[:, 1:-1] + a[:, 2:]) / 3
def sim(L, a, b, hi=None, label=''):
    d = PRED[L]; d = d[(d.date >= a) & (d.date <= b)]; Y, Pm = wide(d, 'y').dropna(), wide(d, 'p'); Pm = Pm.reindex(Y.index); By, Bp = block3(Y.values), block3(np.nan_to_num(Pm.values, nan=1e6))
    if hi: mask = np.full(By.shape[1], False); mask[hi[0]:hi[1] + 1] = True
    else: mask = np.full(By.shape[1], True)
    r = np.arange(len(By)); s_m = np.where(mask)[0][np.argmin(Bp[:, mask], axis=1)]; s_o = np.where(mask)[0][np.argmin(By[:, mask], axis=1)]
    return Y, By, s_m, s_o
def fixed(Y, s): return (Y.values[:, s] + Y.values[:, s + 1] + Y.values[:, s + 2]) / 3
def boot_ci(a, b):
    n = len(a); rng = np.random.default_rng(0); v = []
    for _ in range(2000):
        s = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(x + np.arange(7)) % n for x in s])[:n]; v.append(1 - b[i].mean() / a[i].mean())
    return np.round(100 * np.percentile(v, [2.5, 97.5]), 1).tolist()
print('\n[3-1] 매일 3시간 충전, 2일 전 결정 (원/kWh)')
res = []
for nm, a, b in [('전체', DEV0, HOLD1), ('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1)]:
    for wn, hi in [('24시간 중 선택', None), ('낮 9~17시만', (9, 14))]:
        Y, By, sm, so = sim(2, a, b, hi); r = np.arange(len(By)); m = By[r, sm]; o = By[r, so]; eve = fixed(Y, 18); mid = fixed(Y, 11); nit = fixed(Y, 2)
        zero = lambda s: float((np.mean([(Y.values[r, s + k] <= 0) for k in range(3)], axis=0)).mean())
        z_m, z_o = zero(sm), zero(so); z_e, z_i = float((Y.values[:, 18:21] <= 0).mean()), float((Y.values[:, 11:14] <= 0).mean()); z_n = float((Y.values[:, 2:5] <= 0).mean())
        rec = dict(구간=nm, 선택범위=wn, 일수=len(By), 저녁18_21=round(eve.mean(), 1), 낮11_14=round(mid.mean(), 1), 새벽2_5=round(nit.mean(), 1), 모델=round(m.mean(), 1), 완벽예측=round(o.mean(), 1), 저녁대비절감률=round(100 * (1 - m.mean() / eve.mean()), 1), CI=boot_ci(eve, m), 낮고정대비절감률=round(100 * (1 - m.mean() / mid.mean()), 1), CI2=boot_ci(mid, m), 달성률_저녁기준=round(100 * (eve.mean() - m.mean()) / (eve.mean() - o.mean()), 1), 저가시간충전비중_모델=round(100 * z_m, 1), 저가_저녁=round(100 * z_e, 1), 저가_낮고정=round(100 * z_i, 1), 저가_완벽=round(100 * z_o, 1))
        res.append(rec); print(' ', rec)
pd.DataFrame(res).to_csv(f'{O}/charge_daily.csv', index=False, encoding='utf-8-sig')
print('\n[3-2] 유연 충전: 3일 묶음마다 1회, 1·2·3일 전 예측으로 날짜와 시간 선택 (낮 9~17시)')
hi = (9, 14); fl = []
for nm, a, b in [('전체', DEV0, HOLD1), ('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1)]:
    days = sorted(set(PRED[1].date) & set(PRED[2].date) & set(PRED[3].date)); days = [x for x in days if pd.Timestamp(a) <= x <= pd.Timestamp(b)]; realized, oracle, f_mid, f_eve = [], [], [], []; share = []; zero_hours = []
    for s in range(0, len(days) - 2, 3):
        best = None; orc = None
        for k, L in zip(range(3), (1, 2, 3)):
            dx = PRED[L][PRED[L].date == days[s + k]].sort_values('hour')
            if len(dx) < 24: continue
            yy, pp = dx.y.values, dx.p.values
            for st in range(hi[0], hi[1] + 1):
                pv, yv = pp[st:st + 3].mean(), yy[st:st + 3].mean()
                if best is None or pv < best[0]: best = (pv, yv, (yy[st:st + 3] <= 0).mean())
                if orc is None or yv < orc: orc = yv
        d0 = PRED[1][PRED[1].date == days[s]].sort_values('hour')
        if best is None or len(d0) < 24: continue
        realized.append(best[1]); oracle.append(orc); zero_hours.append(best[2]); f_mid.append(d0.y.values[11:14].mean()); f_eve.append(d0.y.values[18:21].mean())
    r_, o_, m_, e_ = map(np.array, (realized, oracle, f_mid, f_eve))
    rec = dict(구간=nm, 묶음수=len(r_), 저녁18_21=round(e_.mean(), 1), 낮11_14=round(m_.mean(), 1), 모델=round(r_.mean(), 1), 완벽예측=round(o_.mean(), 1), 저녁대비절감률=round(100 * (1 - r_.mean() / e_.mean()), 1), CI=boot_ci(e_, r_), 낮고정대비절감률=round(100 * (1 - r_.mean() / m_.mean()), 1), CI2=boot_ci(m_, r_), 달성률_저녁기준=round(100 * (e_.mean() - r_.mean()) / (e_.mean() - o_.mean()), 1), 저가시간충전비중=round(100 * np.mean(zero_hours), 1)); fl.append(rec); print(' ', rec)
pd.DataFrame(fl).to_csv(f'{O}/charge_flex.csv', index=False, encoding='utf-8-sig')

# ---------- 4. V2G 차익 시뮬레이션 ----------
print('\n[4] V2G: 하루 1회, 2일 전 결정. 왕복효율 85%, 배터리 비용 c원/kWh(방전량 기준), 원/kWh(방전 1kWh) 일평균')
v = []
for nm, a, b in [('전체', DEV0, HOLD1), ('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1)]:
    d = PRED[2]; d = d[(d.date >= a) & (d.date <= b)]; Y, Pm = wide(d, 'y').dropna(), wide(d, 'p'); Pm = Pm.reindex(Y.index); By, Bp = block3(Y.values), block3(np.nan_to_num(Pm.values, nan=1e6)); n = len(By); eta = .85
    def plan(B):
        best_v, cs, ds = np.full(n, -1e9), np.zeros(n, int), np.zeros(n, int)
        for i in range(n):
            for c_ in range(0, By.shape[1] - 3):
                for d_ in range(c_ + 3, By.shape[1]):
                    x = B[i, d_] - B[i, c_] / eta
                    if x > best_v[i]: best_v[i], cs[i], ds[i] = x, c_, d_
        return best_v, cs, ds
    pv, cs, ds = plan(Bp); ov, oc, od = plan(By); rr = np.arange(n); real = By[rr, ds] - By[rr, cs] / eta
    fx = By[rr, 18] - By[rr, 11] / eta
    for cost in [0, 30, 60]:
        act = pv > cost; vm = (real - cost) * act; vo = np.maximum(ov - cost, 0); vf = (fx - cost); fxact = np.maximum(vf, 0) * 0 + vf
        rec = dict(구간=nm, 일수=n, 비용=cost, 거래일_모델=int(act.sum()), 모델=round(vm.mean(), 1), 고정매일=round(fxact.mean(), 1), 완벽예측=round(vo.mean(), 1), 달성률=round(100 * vm.mean() / vo.mean(), 1) if vo.mean() > 0 else None, 거래당=round(vm[act].mean(), 1) if act.sum() else None); v.append(rec); print(' ', rec)
pd.DataFrame(v).to_csv(f'{O}/v2g.csv', index=False, encoding='utf-8-sig')
print('done')
