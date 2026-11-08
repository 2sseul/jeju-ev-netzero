"""강건성 2차: (a) 3일 전 구름량 이상 점검 (b) 저가일 앙상블 대 맑은 날 규칙 (c) ERA5로 추정한 태양광·풍력 가능량으로 집계 단절 영향 점검."""
import os, subprocess
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_forecast.py'), encoding='utf8').read().split('\nT0, TD = ')[0]
exec(src)
from scipy import stats as st
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
def blk(u, L=21, B=3000, seed=0):
    u = np.asarray(u, float); n = len(u); rng = np.random.default_rng(seed); o = [u[np.concatenate([(x + np.arange(L)) % n for x in rng.integers(0, n, n // L + 1)])[:n]].mean() for _ in range(B)]; return np.round(np.percentile(o, [2.5, 97.5]), 2).tolist()
F = {L: frame(L) for L in (1, 2, 3)}; F2_ = F[2]
# ---------- a. 구름량 ----------
print('[a] 리드별 구름량', flush=True)
Wx = {L: load_wx(L) for L in (1, 2, 3)}
def load_e(f):
    o = []
    for s_, x in zip(SITES, json.load(open(f))):
        h = pd.DataFrame(x['hourly']); h['ts'] = pd.to_datetime(h.time); h = h.drop(columns='time').set_index('ts').apply(pd.to_numeric, errors='coerce'); r = np.deg2rad(h.wind_direction_10m); h['wdsin'] = np.sin(r); h['wdcos'] = np.cos(r); h = h.drop(columns='wind_direction_10m').rename(columns=RN); h.columns = [f'{c}_{s_}' for c in h.columns]; o.append(h)
    w = pd.concat(o, axis=1)
    for v in RN.values(): w[f'{v}_mean'] = w[[f'{v}_{s_}' for s_ in SITES]].mean(axis=1)
    w['ws3_mean'] = w.ws_mean ** 3; return w
E0 = load_e(f'{A_}/wx/era.json')
ix = E0.index.intersection(Wx[1].index).intersection(Wx[2].index).intersection(Wx[3].index); ix = ix[(ix >= '2024-03-01') & (ix <= '2024-06-30')]
for v in ['cloud_mean', 'rad_mean', 'temp_mean', 'ws_mean']:
    print(' ', v, {L: dict(MAE=round(float((Wx[L].loc[ix, v] - E0.loc[ix, v]).abs().mean()), 2), corr=round(float(Wx[L].loc[ix, v].corr(E0.loc[ix, v])), 3), 평균=round(float(Wx[L].loc[ix, v].mean()), 1), sd=round(float(Wx[L].loc[ix, v].std()), 1)) for L in Wx}, 'ERA 평균', round(float(E0.loc[ix, v].mean()), 1))
print('  d3와 d1 상관', round(float(Wx[3].loc[ix, 'cloud_mean'].corr(Wx[1].loc[ix, 'cloud_mean'])), 3), ' 고유값 수 d1,d3', Wx[1].loc[ix, 'cloud_jeju'].nunique(), Wx[3].loc[ix, 'cloud_jeju'].nunique())
ens3 = W[3]['best_match']; i2 = ix.intersection(ens3.index); print('  앙상블 파일 best_match 3일 전과 d3 상관', round(float(ens3.loc[i2, 'cloud'].corr(Wx[3].loc[i2, 'cloud_mean'])), 3), ' 구간별 d3 구름 평균', Wx[3].cloud_mean.groupby(Wx[3].index.to_period('Q')).mean().round(0).to_dict(), ' d1', Wx[1].cloud_mean.groupby(Wx[1].index.to_period('Q')).mean().round(0).to_dict())
# ---------- b. 저가일 앙상블 대 맑은 날 규칙 ----------
print('\n[b] SMP 0원 이하 시간 충전 비중: 앙상블 대 맑은 날 규칙', flush=True)
saved = pd.read_csv(f'{A_}/out_price/ens_probs_by_lead.csv', index_col=0, parse_dates=True); saved.columns = [1, 2, 3]; J2 = J.copy(); J2.columns = range(1, 25); hrs = [12, 13, 14]; off = Lg.offday
rad = {L: daily_wx(W[L]['best_match'], '', ['rad', 'cloud']) for L in (1, 2, 3)}; rad = {L: (lambda d: d[d.index.notna()])(v) for L, v in rad.items()}
def r3(L): d = dates(W[L]['best_match']); m = d[d.hour.isin(hrs)]; return m.groupby('date').rad.mean()
rd3 = {L: r3(L) for L in (1, 2, 3)}
days = [x for x in pd.date_range('2025-06-01', '2026-11-07') if x in J2.index and x in saved.index and saved.loc[x].notna().all()]; rows = []
for s in range(len(days) - 2):
    B = days[s:s + 3]
    if (B[2] - B[0]).days != 2: continue
    Z = list(zip(B, (1, 2, 3))); wk = [d for d in B if off[d] == 1]; rl = lambda z: rd3[z[1]].get(z[0], -1)
    pk = {'첫날 고정': B[0], '주말 우선': wk[0] if wk else B[0], '일사 최대': max(Z, key=rl)[0], '주말 우선+맑은 날': (max([z for z in Z if z[0] in wk], key=rl)[0] if wk else max(Z, key=rl)[0]), '앙상블': max(Z, key=lambda z: saved.loc[z[0], z[1]])[0], '완벽(최저가)': min(B, key=lambda d: J2.loc[d, hrs].mean())}
    for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, low=100 * float((J2.loc[d, hrs] <= 0).mean()), smp=float(J2.loc[d, hrs].mean())))
R = pd.DataFrame(rows); Lw = R.pivot(index='start', columns='방식', values='low'); Sp = R.pivot(index='start', columns='방식', values='smp')
for nm, a, b in [('개발', '2025-06-01', '2026-05-31'), ('확인', '2026-06-01', '2026-11-07'), ('전체', '2025-06-01', '2026-11-07')]:
    l, p = Lw[a:b], Sp[a:b]; print(' ', nm, len(l), pd.DataFrame({'비중': l.mean().round(1), 'SMP': p.mean().round(1)}).T.to_dict())
    for base in ['주말 우선', '일사 최대', '주말 우선+맑은 날']: print(f'    앙상블-{base}: 비중 {(l["앙상블"] - l[base]).mean():+.1f}%p CI {blk((l["앙상블"] - l[base]).values)}, SMP {(p[base] - p["앙상블"]).mean():+.1f}원 CI {blk((p[base] - p["앙상블"]).values)}')
pd.concat({'low': Lw.mean(), 'smp': Sp.mean()}, axis=1).round(2).to_csv(f'{OG}/rob_lowprice_rules.csv', encoding='utf-8-sig')
# ---------- c. ERA5 추정 가능량 ----------
print('\n[c] ERA5 기반 태양광·풍력 가능량', flush=True)
f2 = f'{A_}/wx/era_2024h2_2025.json'; LAT = "33.5141,33.2938,33.3868,33.2461"; LON = "126.5297,126.1628,126.8802,126.5653"; V_ = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,cloud_cover"
if not os.path.exists(f2):
    subprocess.run(['curl', '-s', '-m', '280', '-o', f2, f'https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&hourly={V_}&start_date=2024-07-01&end_date=2025-12-31&timezone=Asia%2FSeoul'], check=True)
E1 = load_e(f2); E = pd.concat([E0, E1[E1.index > E0.index.max()]]).sort_index(); E = E[~E.index.duplicated()]
Gx = G.join(E, how='inner'); Gx = Gx[Gx.date >= '2021-01-01']; capS_ = G.solar.rolling('365D').quantile(.995).shift(24); capW_ = G.wind_pot.rolling('365D').quantile(.995).shift(24); Gx['cap_s'] = capS_.reindex(Gx.index); Gx['cap_w'] = capW_.reindex(Gx.index); Gx['solar_n'] = Gx.solar / Gx.cap_s; Gx['wind_n'] = Gx.wind_pot / Gx.cap_w
XE = [c for c in WS if c in Gx.columns] + ['hour', 'month', 'doy']; ok = (Gx.date < '2024-06-01') & (Gx.date >= '2021-01-01')
tsol = Gx[ok & (Gx.curt_solar == 0)].dropna(subset=['solar_n']); twnd = Gx[ok].dropna(subset=['wind_n'])
# 품질 확인: 2023 말까지 학습, 2024-01~05 예측
a_ = hgb(tsol[tsol.date < '2024-01-01'], XE, 'solar_n'); b_ = hgb(twnd[twnd.date < '2024-01-01'], XE, 'wind_n'); te = Gx[(Gx.date >= '2024-01-01') & (Gx.date < '2024-06-01')].dropna(subset=['solar', 'wind_pot', 'cap_s', 'cap_w'])
print('  표본 외 품질 2024-01~05 R2: 태양광', round(r2_score(te.solar, a_.predict(te[XE]) * te.cap_s), 3), ' 풍력', round(r2_score(te.wind_pot, b_.predict(te[XE]) * te.cap_w), 3))
ms, mw = hgb(tsol, XE, 'solar_n'), hgb(twnd, XE, 'wind_n'); capS0, capW0 = capS_[:'2024-05-31'].dropna().iloc[-1], capW_[:'2024-05-31'].dropna().iloc[-1]; print('  고정 설비 규모', round(capS0), round(capW0))
Px = Gx[Gx.date >= '2024-06-01'].copy(); Px['sol_e'] = np.clip(ms.predict(Px[XE]), 0, None) * capS0; Px['wnd_e'] = np.clip(mw.predict(Px[XE]), 0, None) * capW0
Q_ = Px.groupby(Px.date.dt.to_period('Q')).agg(기록태양광=('solar', 'mean'), 추정태양광=('sol_e', 'mean'), 추정풍력=('wnd_e', 'mean')).round(1); Q_['비율'] = (Q_.기록태양광 / Q_.추정태양광).round(2); print(Q_.to_string())
pre = te; print('  단절 전 기록/추정 태양광(2024-01~05, 표본 외)', round(float(pre.solar.mean() / (a_.predict(pre[XE]) * pre.cap_s).clip(0).mean()), 2))
Px['net_e'] = Px.dem - Px.sol_e - Px.wnd_e; Px['net_se'] = Px.dem - Px.sol_e; g3 = Px[Px.hour.isin(hrs)].groupby('date')[['dem', 'solar', 'sol_e', 'wnd_e', 'net_e', 'net_se', 'net_s']].mean(); g3['y'] = Lg.y.reindex(g3.index); g3 = g3.dropna(subset=['y', 'dem']); L_ = g3['2024-06-05':'2025-12-31'].dropna()
print(f'  라벨 검증 {len(L_)}일, 저가일 {int(L_.y.sum())}'); print(L_.groupby('y')[['dem', 'solar', 'sol_e', 'wnd_e', 'net_e']].mean().round(1).to_string()); print('  재생 비중(추정)', (L_.groupby('y').apply(lambda d: ((d.sol_e + d.wnd_e) / d.dem).mean())).round(3).to_dict(), ' AUC 추정 순부하', round(auc(L_.y, -L_.net_e), 3), ' 기록 수요-태양광', round(auc(L_.y, -L_.net_s), 3), ' 수요', round(auc(L_.y, -L_.dem), 3))
XON = WS + CALG
def walk_op(FL, X, y, a, b):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = F2_[(F2_.date < per.start_time) & F2_[y].notna()]; c_ = FL[FL.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, y).predict(c_[X]), index=c_.index))
    return pd.concat(out)
Pn = {L: walk_op(F[L], XON, 'net_s', '2024-06', '2025-12') for L in F}
def dm(s, Fr): x = pd.DataFrame({'p': s, 'date': Fr.date.reindex(s.index), 'hour': Fr.hour.reindex(s.index)}); return x[x.hour.isin(hrs)].groupby('date').p.mean()
pm = {L: dm(Pn[L], F[L]) for L in F}; rdl = {L: F[L][F[L].hour.isin(hrs)].groupby('date').rad_mean.mean() for L in F}
ds = [d for d in g3.index if d >= pd.Timestamp('2024-06-05') and all(d in pm[L].index for L in pm)]; rows = []
for s in range(len(ds) - 2):
    B = ds[s:s + 3]
    if (B[2] - B[0]).days != 2: continue
    Z = list(zip(B, (1, 2, 3))); wk = [d for d in B if OFF[d] == 1]
    pk = {'첫날 고정': B[0], '주말 우선': wk[0] if wk else B[0], '일사 최대': max(Z, key=lambda z: rdl[z[1]][z[0]])[0], '운영 모델': min(Z, key=lambda z: pm[z[1]][z[0]])[0], '완벽(추정 순부하)': min(B, key=lambda d: g3.net_e[d])}
    for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, net_e=g3.net_e[d], net_se=g3.net_se[d], net_s=g3.net_s[d]))
Rx = pd.DataFrame(rows); Ne = Rx.pivot(index='start', columns='방식', values='net_e'); print(Rx.groupby('방식')[['net_e', 'net_se', 'net_s']].mean().round(1).to_string(), len(Ne))
for nm, a, b in [('2024.06~2025.12', '2024-06-01', '2025-12-31'), ('2025', '2025-01-01', '2025-12-31')]:
    n_ = Ne[a:b]
    for base in ['첫날 고정', '주말 우선', '일사 최대']: print(f'  [{nm}] 추정 순부하 기준 운영 모델 - {base}: 감소 {(n_[base] - n_["운영 모델"]).mean():.1f}MW CI {blk((n_[base] - n_["운영 모델"]).values)}')
Rx.to_csv(f'{OG}/rob_era_charge.csv', index=False, encoding='utf-8-sig'); L_.round(2).to_csv(f'{OG}/label_check_era.csv', encoding='utf-8-sig')
# ---------- d. 리드별 입력 차이가 추천일을 왜곡하는가 ----------
print('\n[d] 리드별 입력 차이 점검', flush=True)
A0 = '2024-09'; A1 = '2025-12'; act = PV['net_s'][hrs].mean(axis=1)
def same_lead(FL, X, a, b):
    """같은 리드 입력으로 학습. 3일 전 예보가 있는 2024-02 이후만 학습."""
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = FL[(FL.date < per.start_time) & (FL.date >= '2024-02-15') & FL.net_s.notna()]; c_ = FL[FL.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, 'net_s').predict(c_[X]), index=c_.index))
    return pd.concat(out)
XNC = [c for c in XON if not c.startswith('cloud')]
VAR = {'현행(2일 전으로 학습)': {L: walk_op(F[L], XON, 'net_s', A0, A1) for L in F}, '현행, 구름량 제외': {L: walk_op(F[L], XNC, 'net_s', A0, A1) for L in F}, '리드별 학습(2024.02~)': {L: same_lead(F[L], XON, A0, A1) for L in F}, '2일 전으로 학습, 짧은 기간(2024.02~)': {L: (lambda L_: pd.concat([pd.Series(hgb(F2_[(F2_.date < per.start_time) & (F2_.date >= '2024-02-15') & F2_.net_s.notna()], XON, 'net_s').predict(F[L_][F[L_].date.dt.to_period('M') == per][XON]), index=F[L_][F[L_].date.dt.to_period('M') == per].index) for per in pd.period_range(A0, A1, freq='M')]))(L) for L in F}}
out = []
for nm, P_ in VAR.items():
    pm_ = {L: dm(P_[L], F[L]) for L in F}; ds_ = [d for d in pd.date_range(A0 + '-01', A1 + '-31') if pd.notna(act.get(d)) and all(d in pm_[L].index for L in F)]; picks = []
    for s in range(len(ds_) - 2):
        B = ds_[s:s + 3]
        if (B[2] - B[0]).days != 2: continue
        Z = list(zip(B, (1, 2, 3))); d_ = min(Z, key=lambda z: pm_[z[1]][z[0]]); wk = [x for x in B if OFF[x] == 1]; r_ = max(Z, key=lambda z: rdl[z[1]][z[0]])[0]; orc = min(B, key=lambda x: act[x]); picks.append(dict(start=B[0], pos=d_[1], m=act[d_[0]], first=act[B[0]], wk=act[wk[0] if wk else B[0]], rad=act[r_], top=act[orc], radpos=[z[1] for z in Z if z[0] == r_][0], topos=B.index(orc) + 1))
    R_ = pd.DataFrame(picks); pos = (R_.pos.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(); e_ = {L: round(float((act.reindex(pm_[L].index) - pm_[L]).abs().mean()), 1) for L in F}; b_ = {L: round(float((pm_[L] - act.reindex(pm_[L].index)).mean()), 1) for L in F}
    rec = dict(방식=nm, 묶음=len(R_), 모델=round(R_.m.mean(), 1), 첫날=round(R_['first'].mean(), 1), 주말=round(R_.wk.mean(), 1), 일사=round(R_.rad.mean(), 1), 완벽=round(R_.top.mean(), 1), 일사대비CI=blk((R_.rad - R_.m).values), 주말대비CI=blk((R_.wk - R_.m).values), 선택위치=pos, 리드오차=e_, 리드편향=b_); out.append(rec); print(' ', rec, flush=True)
print('  완벽 예측의 위치', (R_.topos.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(), ' 일사 규칙 위치', (R_.radpos.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
pd.DataFrame(out).to_csv(f'{OG}/rob_lead_variants.csv', index=False, encoding='utf-8-sig'); print('done')
