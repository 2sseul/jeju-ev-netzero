"""수요 반영(오경보), 비용 기반 기준선, 확률 신뢰도. smp_model.py의 데이터 준비부를 재사용한다."""
import os, pickle, json
import numpy as np, pandas as pd
A_ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A_}/smp_model.py', encoding='utf8').read().split('\nout = {}')[0].replace('__file__', repr(f'{A_}/smp_model.py'))
exec(src)
from sklearn.linear_model import LogisticRegression

# ---------- 수요 특징 ----------
offd = pd.Series([int(x.weekday() >= 5 or x.date() in kr) for x in L.index], index=L.index)
def dem_base():
    out = {}
    dm, of = L.dem_mid, offd
    for t in L.index:
        w = dm[(dm.index >= t - pd.Timedelta(days=16)) & (dm.index <= t - pd.Timedelta(days=3))]
        s = w[of.reindex(w.index) == of[t]]
        out[t] = s.mean() if len(s) else np.nan
    return pd.Series(out)
L['dem_base'] = dem_base()
hist = G.sort_index(); s_all = pd.concat([sol, s2[s2.index > sol.index.max()]]).sort_index()
capS = s_all.rolling('365D').quantile(.995).shift(24).resample('D').last().reindex(L.index).ffill()
capW = float(G.cap_w.dropna().iloc[-1]); print('capS 2026', round(capS.iloc[-1], 1), 'capW', round(capW, 1))
def with_net(D):
    d = D.copy(); d['dem_base'] = L.dem_base.reindex(d.index); d['re_abs'] = capS.reindex(d.index) * d.pA_mid + capW * d.pB_mid
    d['net'] = d.dem_base - d.re_abs; d['re_share'] = d.re_abs / d.dem_base; return d
D1n, D2n = with_net(D1), with_net(D2)
BASE = AB + WX + CALS + LAGS; NET = BASE + ['dem_base', 're_abs', 'net', 're_share']
SETS2 = {'기존': BASE, '수요 추가': NET}
def walk2(D, X):
    """월별 재학습: 시험 예측값과 학습기간 OOF 예측값을 함께 저장"""
    D = D[D.index >= START]; rows = []
    for per in pd.period_range(TEST0, TEST1, freq='M'):
        t = D[D.index < per.start_time]; b = D[D.index.to_period('M') == per]
        q = t.index.to_period('Q'); ps, ys = [], []
        for k in q.unique():
            if t[q != k].y.nunique() < 2: continue
            ps.append(clf(t[q != k], X).predict_proba(t[q == k][X])[:, 1]); ys.append(t[q == k].y.values)
        rows.append(dict(per=per, oof_p=np.concatenate(ps), oof_y=np.concatenate(ys), idx=b.index, p=clf(t, X).predict_proba(b[X])[:, 1]))
    return rows
C = {}
for lead, D in [('d1', D1n), ('d2', D2n)]:
    for nm, X in SETS2.items(): C[(lead, nm)] = walk2(D, X)
te = D2n[(D2n.index >= TEST0) & (D2n.index <= TEST1)]; y = te.y.values; n = len(y)
def pser(r): return pd.concat([pd.Series(x['p'], index=x['idx']) for x in r]).reindex(te.index).values
def f1thr(px, yx):
    ts = np.arange(.05, .7, .025); return float(ts[int(np.argmax([f1_score(yx, (px >= t).astype(int)) for t in ts]))])
def apply_thr(r, fn):
    return pd.concat([pd.Series((x['p'] >= fn(x['oof_p'], x['oof_y'])).astype(int), index=x['idx']) for x in r]).reindex(te.index).values
print('\n[1] 수요 반영 효과')
for lead in ['d1', 'd2']:
    for nm in SETS2:
        r = C[(lead, nm)]; p = pser(r); yh = apply_thr(r, f1thr); tp = int((y & yh).sum()); fp = te[(yh == 1) & (y == 0)]
        print(f' {lead} {nm}: AUC {auc(y, p):.3f} PR {ap(y, p):.3f} days {yh.sum()} prec {tp / yh.sum():.3f} rec {tp / y.sum():.3f} F1 {f1_score(y, yh):.3f} 오경보 {len(fp)} 월별 오경보 {fp.groupby(fp.index.month).size().to_dict()}')
rng = np.random.default_rng(0); pa, pb = pser(C[('d2', '기존')]), pser(C[('d2', '수요 추가')]); dif = []; difp = []
for _ in range(2000):
    s = rng.integers(0, n, n // 7 + 1); i = np.concatenate([(x + np.arange(7)) % n for x in s])[:n]
    if y[i].min() == y[i].max(): continue
    dif.append(auc(y[i], pb[i]) - auc(y[i], pa[i])); difp.append(ap(y[i], pb[i]) - ap(y[i], pa[i]))
print(' d2 차이 AUC CI', np.round(np.percentile(dif, [2.5, 97.5]), 3).tolist(), ' PR CI', np.round(np.percentile(difp, [2.5, 97.5]), 3).tolist())
pickle.dump(dict(C=C, te=te), open(f'{A_}/out_smp/walk_cache.pkl', 'wb'))

# ---------- 비용 기반 기준선 (기존 모델, 2일 전) ----------
print('\n[2] 비용 기반 기준선 (오경보 1 : 놓침 r)')
r0 = C[('d2', '기존')]
def cost_thr(ratio):
    def fn(px, yx):
        ts = np.arange(.02, .9, .02); c = [((px >= t) & (yx == 0)).sum() + ratio * ((px < t) & (yx == 1)).sum() for t in ts]; return float(ts[int(np.argmin(c))])
    return fn
for ratio in [0.5, 1, 2, 3, 5]:
    yh = apply_thr(r0, cost_thr(ratio)); tp = int((y & yh).sum()); fp = int(((yh == 1) & (y == 0)).sum()); fn_ = int(((yh == 0) & (y == 1)).sum())
    # 비교: 같은 비용의 규칙
    cw, ca, cn = (te.offday.values == 1), np.ones(n, bool), np.zeros(n, bool); rc = lambda m: int((m & (y == 0)).sum()) + ratio * int((~m & (y == 1)).sum())
    print(f' 1:{ratio}: days {yh.sum()} prec {tp / max(yh.sum(), 1):.3f} rec {tp / y.sum():.3f} 오경보 {fp} 놓침 {fn_} 총비용 {fp + ratio * fn_:.0f} | 주말고정 {rc(cw):.0f} 매일 {rc(ca):.0f} 추천없음 {rc(cn):.0f}')
# 금전 기준: 저녁 대신 11~14시 충전 시 도매가격 차이
sp = (te.eve - te.w1114).values; print(' 11~14시 대비 저녁 SMP 차이(원/kWh): 전체 %.1f, 저가일 %.1f, 비저가일 %.1f' % (sp.mean(), sp[y == 1].mean(), sp[y == 0].mean()))
print(' 하루 불편비용 k원/kWh 기준 최적 판단 (학습 OOF로 p 기준선 선택 대신 단순 비교):')
yhF = apply_thr(r0, f1thr)
for k in [0, 10, 20, 40, 60]:
    v = lambda m: float(((sp - k) * m).sum() / n)
    print(f'  k={k}: 모델 {v(yhF.astype(bool)):.1f} 주말고정 {v(te.offday.values == 1):.1f} 매일 {v(np.ones(n, bool)):.1f} (원/kWh·일 평균 순이득)')

# ---------- 확률 신뢰도 ----------
print('\n[3] 확률 신뢰도 (2일 전, 기존 모델)')
p_raw = pser(r0); cal_p = []
for x in r0:
    lg = LogisticRegression(C=1e6).fit(np.log(np.clip(x['oof_p'], 1e-4, 1 - 1e-4) / (1 - np.clip(x['oof_p'], 1e-4, 1 - 1e-4))).reshape(-1, 1), x['oof_y'])
    pp = np.clip(x['p'], 1e-4, 1 - 1e-4); cal_p.append(pd.Series(lg.predict_proba(np.log(pp / (1 - pp)).reshape(-1, 1))[:, 1], index=x['idx']))
p_cal = pd.concat(cal_p).reindex(te.index).values
clim = np.array([]); base_rate = []
for x in r0: base_rate.append(pd.Series(x['oof_y'].mean(), index=x['idx']))
base_rate = pd.concat(base_rate).reindex(te.index).values
br = lambda p: float(np.mean((p - y) ** 2))
def ece(p, bins=(0, .05, .1, .2, .35, .5, .7, 1.01)):
    b = np.digitize(p, bins) - 1; return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean()) for i in np.unique(b)))
print(f' Brier 원본 {br(p_raw):.4f} 보정 {br(p_cal):.4f} 기준(과거 평균) {br(base_rate):.4f} | Skill 원본 {1 - br(p_raw) / br(base_rate):.3f} 보정 {1 - br(p_cal) / br(base_rate):.3f} | ECE 원본 {ece(p_raw):.3f} 보정 {ece(p_cal):.3f} | AUC 보정 {auc(y, p_cal):.3f}')
tab = []
for nm, p in [('원본', p_raw), ('보정', p_cal)]:
    bins = [0, .05, .15, .3, .5, .7, 1.01]; g = pd.DataFrame({'p': p, 'y': y}).groupby(pd.cut(p, bins, right=False)).agg(n=('y', 'size'), pred=('p', 'mean'), obs=('y', 'mean')).round(3); print(nm); print(g.to_string()); tab.append(g)
print('안내 등급(보정 확률): ', end='')
tier = pd.cut(p_cal, [0, .1, .4, 1.01], right=False, labels=['낮음', '보통', '높음']); t = pd.DataFrame({'t': tier, 'y': y}).groupby('t').agg(days=('y', 'size'), hit=('y', 'sum'), rate=('y', 'mean')).round(3); print(t.to_dict('index'))
te_out = te[['y', 'offday', 'month', 'w1114', 'eve', 'h_le0']].assign(p_raw=p_raw, p_cal=p_cal, yh=yhF); te_out.to_csv(f'{A_}/out_smp/daily_calibrated.csv')
print('saved')
