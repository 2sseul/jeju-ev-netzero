"""유연 충전 비교: 3일 묶음마다 하루를 골라 충전. 5개 예보 앙상블 확률로 날짜를 고르는 방식을 규칙과 비교한다.
시작 위치 3가지(0,1,2일)를 평균해 묶음 경계의 우연을 줄인다. 개발 구간에서 방식을 고르고 확인 구간으로 검증한다."""
import os
A_ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A_}/ens_price.py', encoding='utf8').read().split('\nP = {k: walk(v)')[0].replace('__file__', repr(f'{A_}/ens_price.py'))
exec(src)
PL = {}
for L in (1, 2, 3):
    fr = {m: daily_wx(W[L][m], f'_{m}', list(VARS.values())) for m in MODELS}; Dx = Lg.copy()
    for m in MODELS: Dx = Dx.join(fr[m])
    Dx = Dx[Dx.index >= START]; lists = [list(fr[m].columns) + CAL + LAG for m in MODELS]; rows = []
    for per in pd.period_range(DEV0, HOLD1, freq='M'):
        t = Dx[(Dx.index < per.start_time) & Dx.y.notna()]; b = Dx[(Dx.index.to_period('M') == per) & Dx.y.notna() & (Dx.index <= HOLD1)]
        if len(b): rows.append(pd.Series(np.mean([clf(t, x).predict_proba(b[x])[:, 1] for x in lists], axis=0), index=b.index))
    PL[L] = pd.concat(rows); print('lead', L, 'AUC', round(auc(Lg.y.reindex(PL[L].index).astype(int), PL[L]), 3), 'PR', round(ap(Lg.y.reindex(PL[L].index).astype(int), PL[L]), 3), flush=True)
pd.DataFrame(PL).to_csv(f'{O}/ens_probs_by_lead.csv')
# 2일 전 앙상블과 기존 모델을 같은 84일 추천으로 비교
old = pd.read_csv(f'{A_}/out_smp/daily.csv', parse_dates=['date']).set_index('date'); ix = old.index.intersection(PL[2].index); y = Lg.y.reindex(ix).astype(int).values
def at_k(p, k): m = np.zeros(len(p), int); m[np.argsort(-p)[:k]] = 1; return dict(days=k, prec=round((y & m).sum() / k, 3), rec=round((y & m).sum() / y.sum(), 3))
print('\n[앙상블 대 기존, 2025-06~2026-10] AUC', round(auc(y, old.p2.reindex(ix)), 3), '->', round(auc(y, PL[2].reindex(ix)), 3), ' PR', round(ap(y, old.p2.reindex(ix)), 3), '->', round(ap(y, PL[2].reindex(ix)), 3))
for k in (60, 84, 120): print(' k', k, '기존', at_k(old.p2.reindex(ix).values, k), '앙상블', at_k(PL[2].reindex(ix).values, k))
J2 = J.copy(); J2.columns = range(1, 25); win = [12, 13, 14]; eve = [19, 20, 21]; nit = [3, 4, 5]
pat = hourly = None
# 과거 패턴: 같은 요일유형의 D-4 이전 28일 낮 평균 (날씨 미사용)
offs = Lg.offday; md = J2[win].mean(axis=1).reindex(idx)
patw = md.where(offs == 1).shift(4).rolling(28, min_periods=3).mean(); patn = md.where(offs == 0).shift(4).rolling(28, min_periods=5).mean(); PAT = pd.Series(np.where(offs == 1, patw, patn), index=idx)
PR1, PR2, PR3 = [pd.read_csv(f'{O}/hourly_pred_d{L}.csv', parse_dates=['date']) for L in (1, 2, 3)]
def regday(d): m = d[d.hour.isin(win)].groupby('date').p.mean(); return m
RG = {1: regday(PR1), 2: regday(PR2), 3: regday(PR3)}
def run(a, b):
    days = [x for x in pd.date_range(a, b) if x in J2.index and all(x in PL[L].index for L in PL)]; out = {k: [] for k in ['첫날 저녁 18~21시', '첫날 새벽 2~5시', '첫날 11~14시', '주말·공휴일 우선 11~14시', '과거 패턴 11~14시', '가격 예측 11~14시', '앙상블 확률 11~14시', '완벽 예측 11~14시']}; zr = {k: [] for k in out}; st = []
    for off in (0, 1, 2):
        for s in range(off, len(days) - 2, 3):
            B = days[s:s + 3]
            if (B[2] - B[0]).days != 2: continue
            st.append(B[0]); pick = {'첫날 저녁 18~21시': (B[0], eve), '첫날 새벽 2~5시': (B[0], nit), '첫날 11~14시': (B[0], win)}
            we = [d for d in B if offs[d] == 1]; pick['주말·공휴일 우선 11~14시'] = (we[0] if we else B[0], win)
            pick['과거 패턴 11~14시'] = (min(B, key=lambda d: PAT.get(d, 1e9)), win)
            pick['가격 예측 11~14시'] = (min(zip(B, (1, 2, 3)), key=lambda z: RG[z[1]].get(z[0], 1e9))[0], win)
            pick['앙상블 확률 11~14시'] = (max(zip(B, (1, 2, 3)), key=lambda z: PL[z[1]][z[0]])[0], win)
            pick['완벽 예측 11~14시'] = (min(B, key=lambda d: J2.loc[d, win].mean()), win)
            for k, (d, h) in pick.items(): out[k].append(J2.loc[d, h].mean()); zr[k].append(float((J2.loc[d, h] <= 0).mean()))
    o_ = np.argsort(np.array(st, dtype='datetime64[ns]'))  # 시작일 순으로 정렬해야 부트스트랩 블록이 연속된 21일이 된다
    return {k: np.array(v)[o_] for k, v in out.items()}, {k: np.array(v)[o_] for k, v in zr.items()}
def bci(a, b):
    n = len(a); rng = np.random.default_rng(0); v = []
    for _ in range(2000):
        s = rng.integers(0, n, n // 21 + 1); i = np.concatenate([(x + np.arange(21)) % n for x in s])[:n]; v.append(1 - b[i].mean() / a[i].mean())
    return np.round(100 * np.percentile(v, [2.5, 97.5]), 1).tolist()
rows = []
for nm, a, b in [('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1), ('전체', DEV0, HOLD1)]:
    o, z = run(a, b); print(f'\n[{nm}] 묶음 수 {len(o["첫날 11~14시"])}')
    base = o['첫날 11~14시']; top = o['완벽 예측 11~14시']
    for k in o:
        rec = dict(구간=nm, 방식=k, 평균SMP=round(o[k].mean(), 1), 저가시간비중=round(100 * z[k].mean(), 1), 낮고정대비절감=round(100 * (1 - o[k].mean() / base.mean()), 1), 달성률=round(100 * (base.mean() - o[k].mean()) / (base.mean() - top.mean()), 1)); rows.append(rec); print(' ', rec)
    def zci(u, v):
        n = len(u); rng = np.random.default_rng(1); w = []
        for _ in range(2000):
            st = rng.integers(0, n, n // 21 + 1); i = np.concatenate([(x + np.arange(21)) % n for x in st])[:n]; w.append(100 * (v[i].mean() - u[i].mean()))
        return np.round(np.percentile(w, [2.5, 97.5]), 1).tolist()
    print('  저가시간 비중 차이 CI(%p): 낮 고정 대비', zci(z['첫날 11~14시'], z['앙상블 확률 11~14시']), ' 주말 우선 대비', zci(z['주말·공휴일 우선 11~14시'], z['앙상블 확률 11~14시']), ' 과거 패턴 대비', zci(z['과거 패턴 11~14시'], z['앙상블 확률 11~14시']))
    print('  앙상블 절감 CI: 낮 고정 대비', bci(base, o['앙상블 확률 11~14시']), ' 주말 우선 대비', bci(o['주말·공휴일 우선 11~14시'], o['앙상블 확률 11~14시']), ' 과거 패턴 대비', bci(o['과거 패턴 11~14시'], o['앙상블 확률 11~14시']))
pd.DataFrame(rows).to_csv(f'{O}/flex_compare.csv', index=False, encoding='utf-8-sig'); print('done')
