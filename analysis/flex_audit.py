"""유연 충전 점검: 진짜 상한, 날씨 없는 분류기, 월별 쏠림, 개선 후보(최신 가격, 11~14시 라벨). flex_charge.py를 먼저 실행해야 한다. 파일을 쓰지 않는다."""
import os
A_ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A_}/ens_price.py', encoding='utf8').read().split('\nP = {k: walk(v)')[0].replace('__file__', repr(f'{A_}/ens_price.py'))
exec(src)
J2 = J.copy(); J2.columns = range(1, 25); win = [12, 13, 14]; offs = Lg.offday
y_win = (J2[win].min(axis=1) <= 0).astype(float).reindex(idx).where(J2[win].min(axis=1).reindex(idx).notna())


def probs(mode='base', target='y', fresh=False):
    """mode base: 5개 예보 평균 / nowx: 달력+과거 가격. fresh: 리드별로 알 수 있는 최신 가격(D-(L+1)) 사용. target win: 11~14시 라벨로 학습."""
    PL = {}
    for L in (1, 2, 3):
        fr = {m: daily_wx(W[L][m], f'_{m}', list(VARS.values())) for m in MODELS}; Dx = Lg.copy()
        if fresh:
            k = L + 1; Dx['smin4'] = Dx.midmin.shift(k); Dx['smean14'] = Dx.midmin.shift(k).rolling(14).mean(); Dx['cnt14'] = Dx.y.shift(k).rolling(14).sum()
        if target == 'win': Dx['y'] = y_win
        for m in MODELS: Dx = Dx.join(fr[m])
        Dx = Dx[Dx.index >= START]; lists = [CAL + LAG] if mode == 'nowx' else [list(fr[m].columns) + CAL + LAG for m in MODELS]; rows = []
        for per in pd.period_range(DEV0, HOLD1, freq='M'):
            t = Dx[(Dx.index < per.start_time) & Dx.y.notna()]; b = Dx[(Dx.index.to_period('M') == per) & Dx.y.notna() & (Dx.index <= HOLD1)]
            if len(b): rows.append(pd.Series(np.mean([clf(t, x).predict_proba(b[x])[:, 1] for x in lists], axis=0), index=b.index))
        PL[L] = pd.concat(rows)
    return PL


saved = pd.read_csv(f'{O}/ens_probs_by_lead.csv', index_col=0, parse_dates=True); saved.columns = [1, 2, 3]
V = {'앙상블(현재)': {L: saved[L].dropna() for L in (1, 2, 3)}, '날씨 없는 분류기': probs('nowx'), '최신 가격 반영': probs(fresh=True), '11~14시 라벨 학습': probs(target='win'), '둘 다': probs(target='win', fresh=True)}
days = [x for x in pd.date_range(DEV0, HOLD1) if x in J2.index and x in saved.index and saved.loc[x].notna().all()]
share = (J2[win] <= 0).mean(axis=1); price = J2[win].mean(axis=1)
rows = []
for s in range(len(days) - 2):
    B = days[s:s + 3]
    if (B[2] - B[0]).days != 2: continue
    pk = {'첫날 고정': B[0], '주말 우선': next((d for d in B if offs[d] == 1), B[0]), '가격 최저 날(기존 상한)': min(B, key=lambda d: price[d]), '비중 최대 날(진짜 상한)': max(B, key=lambda d: (share[d], -price[d]))}
    for k, PL in V.items(): pk[k] = max(zip(B, (1, 2, 3)), key=lambda z: PL[z[1]][z[0]])[0]
    for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, share=100 * share[d], price=price[d]))
R = pd.DataFrame(rows)
S = R.pivot(index='start', columns='방식', values='share'); Pz = R.pivot(index='start', columns='방식', values='price')


def ci(u, v, L=21):
    u, v = u.values, v.values; n = len(u); rng = np.random.default_rng(0); out = []
    for _ in range(3000):
        st = rng.integers(0, n, n // L + 1); i = np.concatenate([(x + np.arange(L)) % n for x in st])[:n]; out.append(v[i].mean() - u[i].mean())
    return np.round(np.percentile(out, [2.5, 97.5]), 1).tolist()


for nm, a, b in [('개발', DEV0, DEV1), ('확인', HOLD0, HOLD1), ('전체', DEV0, HOLD1)]:
    s, p = S[a:b], Pz[a:b]; print(f'\n[{nm}] 묶음 {len(s)}'); print(pd.DataFrame({'비중%': s.mean().round(1), '평균SMP': p.mean().round(1)}).sort_values('비중%').to_string())
    f, t, e = s['첫날 고정'].mean(), s['비중 최대 날(진짜 상한)'].mean(), s['앙상블(현재)'].mean(); print(f' 진짜 상한 대비 달성 {(e - f) / (t - f) * 100:.1f}%')
    for base in ['첫날 고정', '주말 우선', '날씨 없는 분류기']:
        print(f' 앙상블-{base} 비중차 CI 21일', ci(s[base], s['앙상블(현재)']), ' 42일', ci(s[base], s['앙상블(현재)'], 42))
    for k in ['최신 가격 반영', '11~14시 라벨 학습', '둘 다']:
        print(f' {k}-앙상블 비중차 {s[k].mean() - e:+.2f} CI', ci(s['앙상블(현재)'], s[k]), f' 가격차 {p[k].mean() - p["앙상블(현재)"].mean():+.1f}')
m = S[['첫날 고정', '주말 우선', '앙상블(현재)', '비중 최대 날(진짜 상한)']].groupby(S.index.to_period('M')).mean().round(1); m['묶음'] = S.groupby(S.index.to_period('M')).size(); print('\n[월별 비중]'); print(m.to_string())
gain = (S['앙상블(현재)'] - S['첫날 고정']); g = gain.groupby(gain.index.to_period('M')).sum(); print('\n상위 4개월 이득 점유율', round(100 * g.sort_values(ascending=False).head(4).sum() / g.sum(), 1), g.sort_values(ascending=False).head(4).index.astype(str).tolist())
print('11~14시에 0원 이하가 있었던 날', int((share.reindex(days) > 0).sum()), '/', len(days), ' 전 시간 0원 이하', int((share.reindex(days) == 1).sum()))
for k, PL in V.items():
    x = PL[2].reindex(days); yy = Lg.y.reindex(days).astype(int); d_ = x.index <= DEV1
    print(k, '2일전 AUC 개발', round(auc(yy[d_], x[d_]), 3), '확인', round(auc(yy[~d_], x[~d_]), 3), ' PR 개발', round(ap(yy[d_], x[d_]), 3), '확인', round(ap(yy[~d_], x[~d_]), 3))
