"""강건성 3차: (A) 전력거래소 공식 수요 예측과 비교 (B) SMP 시장 구분 대조 (C) best_match 없는 일관된 4개 모델 (D) 같은 할인 일수를 예보로 배정 (E) 낮은 날의 요일 구성과 시간대 곡선."""
import os, glob
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_forecast.py'), encoding='utf8').read().split('\nT0, TD = ')[0]
exec(src)
from scipy import stats as st
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
def blk(u, L=21, B=3000, seed=0):
    u = np.asarray(u, float); n = len(u); rng = np.random.default_rng(seed); o = [u[np.concatenate([(x + np.arange(L)) % n for x in rng.integers(0, n, n // L + 1)])[:n]].mean() for _ in range(B)]; return np.round(np.percentile(o, [2.5, 97.5]), 2).tolist()
def nw_dm(d, lag=7):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); u = d - d.mean(); v = (u @ u) / n
    for k in range(1, lag + 1): v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / np.sqrt(v / n); return float(d.mean()), float(t), float(2 * (1 - st.norm.cdf(abs(t)))), n
def kpx_wide(pat):
    fr = []
    for f in sorted(glob.glob(f'{R_}/kpx_fc/{pat}')):
        x = pd.read_excel(f, header=None, engine='openpyxl'); hdr = x.iloc[1].astype(str).tolist(); hc_ = [i for i, h in enumerate(hdr) if h.endswith('h')]; d = x.iloc[2:, [0] + hc_].copy(); d.columns = ['date'] + [int(hdr[i][:-1]) for i in hc_]; d['date'] = pd.to_datetime(d.date.astype(str).str[:8], format='%Y%m%d', errors='coerce'); fr.append(d.dropna(subset=['date']).set_index('date').apply(pd.to_numeric, errors='coerce'))
    return pd.concat(fr).sort_index(kind='mergesort').pipe(lambda z: z[~z.index.duplicated(keep='last')])
F = {L: frame(L) for L in (1, 2, 3)}; F1, F2_ = F[1], F[2]; XOD = WM + CALG; XLAG = XOD + ['dem_lag', 'dem_lag7', 'dem_same']; XON = WS + CALG
def walk(Fx, X, y, a, b):
    out = []
    for per in pd.period_range(a, b, freq='M'):
        t = Fx[(Fx.date < per.start_time) & Fx[y].notna()]; c_ = Fx[Fx.date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, X, y).predict(c_[X]), index=c_.index))
    return pd.concat(out)
# ---------- A. 공식 수요 예측 ----------
print('[A] 공식 하루 전 수요 예측과 비교', flush=True)
OFc = kpx_wide('pss_*.xlsx'); print(' 공식 예측', OFc.index.min().date(), OFc.index.max().date(), len(OFc), ' 값 범위', float(OFc.min().min()), float(OFc.max().max()))
BID = kpx_wide('bid_lfd_*.xlsx'); com = OFc.index.intersection(BID.index); print(' 시범사업 파일과 차이(평균 절대)', round(float((OFc.loc[com, list(range(1, 25))] - BID.loc[com, list(range(1, 25))]).abs().mean().mean()), 3))
sOf = OFc[list(range(1, 25))].stack(); sOf.index.names = ['date', 'hour']
def put_of(Fx): return pd.Series(sOf.reindex(pd.MultiIndex.from_arrays([Fx.date, Fx.hour])).values, index=Fx.index)
PA = {'모델 1일 전 (날씨+달력)': walk(F1, XOD, 'dem', '2023-07', '2026-09'), '모델 1일 전 (최근 실측 포함)': walk(F1, XLAG, 'dem', '2023-07', '2026-09'), '모델 2일 전 (날씨+달력)': walk(F2_, XOD, 'dem', '2023-07', '2026-09'), '모델 2일 전 (최근 실측 포함)': walk(F2_, XLAG, 'dem', '2023-07', '2026-09')}; of_ = put_of(F1); of2 = put_of(F2_)
def tab(Fx, ofc, P, nm, a, b):
    m = (Fx.date >= a) & (Fx.date <= b) & Fx.dem.notna() & ofc.notna(); y = Fx.dem[m]; rows = []
    for k, p in {'공식 하루 전 예측': ofc, **P}.items():
        e = (y - p.reindex(y.index)).dropna(); yy = y.reindex(e.index); h = Fx.hour.reindex(e.index); rows.append(dict(구간=nm, 방식=k, 시간=len(e), MAPE=round(float((e.abs() / yy).mean() * 100), 2), MAE=round(float(e.abs().mean()), 1), RMSE=round(float(np.sqrt((e ** 2).mean())), 1), 편향=round(float(e.mean()), 1), 낮11_14=round(float((e.abs() / yy)[h.isin([12, 13, 14])].mean() * 100), 2), 저녁18_21=round(float((e.abs() / yy)[h.isin([19, 20, 21])].mean() * 100), 2)))
    return rows
Pl1 = {k: v for k, v in PA.items() if '1일' in k}; Pl2 = {k: v for k, v in PA.items() if '2일' in k}; Rw = []
for nm, a, b in [('개발 2023.07~2025.05', '2023-07-01', '2025-05-31'), ('확인 2025.06~2026.09', '2025-06-01', '2026-09-30')]:
    Rw += tab(F1, of_, Pl1, nm, a, b) + [r for r in tab(F2_, of2, Pl2, nm, a, b) if '공식' not in r['방식']]
TA = pd.DataFrame(Rw); print(TA.to_string(index=False)); TA.to_csv(f'{OG}/rob_official.csv', index=False, encoding='utf-8-sig')
m = (F1.date >= '2025-06-01') & (F1.date <= '2026-09-30') & F1.dem.notna() & of_.notna(); y = F1.dem[m]; dd = F1.date[m]
def dlos(p): return ((y - p.reindex(y.index)).abs()).groupby(dd).mean()
for k, p in Pl1.items():
    d = dlos(of_) - dlos(p); r = nw_dm(d.values); print(f'  공식-{k}: 일평균 {r[0]:+.2f}MW t={r[1]:.2f} p={r[2]:.3g} CI {blk(d.values)}')
for k, p in Pl2.items():
    i2 = p.index.intersection(y.index); yy = F2_.dem.reindex(i2); ddd = F2_.date.reindex(i2); d = ((yy - of2.reindex(i2)).abs().groupby(ddd).mean() - (yy - p.reindex(i2)).abs().groupby(ddd).mean()); r = nw_dm(d.values); print(f'  공식(1일 전)-{k}: 일평균 {r[0]:+.2f}MW t={r[1]:.2f} p={r[2]:.3g} CI {blk(d.values)}')
blend = (of_ + PA['모델 1일 전 (날씨+달력)'].reindex(of_.index)) / 2; print('  공식과 모델(1일 전, 날씨+달력) 평균 MAPE', round(float(((y - blend.reindex(y.index)).abs() / y).mean() * 100), 2), ' 오차 상관', round(float((y - of_.reindex(y.index)).corr(y - PA['모델 1일 전 (날씨+달력)'].reindex(y.index))), 3))
# ---------- B. SMP 시장 구분 ----------
print('\n[B] SMP 시장 구분', flush=True)
DA = kpx_wide('smp_da_*.xlsx'); hrs = [12, 13, 14]; Jw = J.copy(); Jw.columns = range(1, 25); print(' 공식 하루전 SMP', DA.index.min().date(), DA.index.max().date(), len(DA))
ci = DA.index.intersection(Jw.index); ci = ci[ci >= '2024-06-01']; dif = (DA.loc[ci, list(range(1, 25))] - Jw.loc[ci, list(range(1, 25))]).abs(); print(' EPSIS와 하루전 평균 절대 차이', round(float(dif.mean().mean()), 3), ' 일치 시간 비율', round(float((dif < .01).mean().mean()), 4))
ml = [c for c in range(11, 17)]; yE = Jw.loc[ci, ml].min(axis=1) <= 0; yD = DA.loc[ci, ml].min(axis=1) <= 0; print(' 낮 0원 이하 일수 EPSIS', int(yE.sum()), ' 하루전', int(yD.sum()), ' 불일치', int((yE != yD).sum()))
RT = glob.glob(f'{R_}/kpx_fc/smp_rt_*.xlsx')
if RT:
    Rr = pd.read_excel(RT[0], header=None, engine='openpyxl'); print(' 실시간 파일 머리', Rr.iloc[:3, :6].values.tolist(), Rr.shape)
# ---------- C. 일관된 4개 모델 ----------
print('\n[C] best_match 제외, 일관된 4개 예보 평균', flush=True)
M4 = ['ecmwf_ifs025', 'gfs_global', 'icon_global', 'gem_global']; vv = ['temp', 'rain', 'ws', 'rad', 'cloud']
def m4(L):
    w = sum(W[L][m][vv] for m in M4) / 4; d = dates(w); return d
frs = {}
for L in (1, 2, 3):
    d = m4(L).dropna(subset=['rad']); d = d.join(G[['net_s', 'dem', 'solar', 'dow', 'offday', 'holiday', 'month', 'doy']], how='left'); frs[L] = d[d.date >= '2024-05-05']
XC4 = vv + ['hour', 'month', 'doy', 'dow', 'offday', 'holiday']; Pc = {}
for L in frs:
    out = []
    for per in pd.period_range('2024-09', '2025-12', freq='M'):
        t = frs[2][(frs[2].date < per.start_time) & frs[2].net_s.notna()]; c_ = frs[L][frs[L].date.dt.to_period('M') == per]
        if len(c_): out.append(pd.Series(hgb(t, XC4, 'net_s').predict(c_[XC4]), index=c_.index))
    Pc[L] = pd.concat(out)
act = PV['net_s'][hrs].mean(axis=1); E_ = pd.read_csv(f'{OG}/label_check_era.csv', parse_dates=['date']).set_index('date'); pmc = {L: pd.DataFrame({'p': Pc[L], 'd': frs[L].date.reindex(Pc[L].index), 'h': frs[L].hour.reindex(Pc[L].index)}).query('h in [12,13,14]').groupby('d').p.mean() for L in Pc}; rdc = {L: frs[L][frs[L].hour.isin(hrs)].groupby('date').rad.mean() for L in frs}
ds = [d for d in pd.date_range('2024-09-01', '2025-12-31') if pd.notna(act.get(d)) and all(d in pmc[L].index for L in pmc)]; rows = []
for s in range(len(ds) - 2):
    B = ds[s:s + 3]
    if (B[2] - B[0]).days != 2: continue
    Z = list(zip(B, (1, 2, 3))); wk = [d for d in B if OFF[d] == 1]; pk = {'첫날': B[0], '주말': wk[0] if wk else B[0], '일사': max(Z, key=lambda z: rdc[z[1]][z[0]])[0], '모델': min(Z, key=lambda z: pmc[z[1]][z[0]])[0], '완벽': min(B, key=lambda d: act[d])}; pos = [z[1] for z in Z if z[0] == pk['모델']][0]
    for k, d in pk.items(): rows.append(dict(start=B[0], 방식=k, net=act[d], era=E_.net_e.get(d, np.nan), pos=pos))
Rc = pd.DataFrame(rows); Ne = Rc.pivot(index='start', columns='방식', values='net'); Ee = Rc.pivot(index='start', columns='방식', values='era'); print(' 선택 위치', (Rc[Rc.방식 == '모델'].pos.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
for nm, a, b in [('전체 2024.09~2025.12', '2024-09-01', '2025-12-31'), ('2025', '2025-01-01', '2025-12-31')]:
    n_, e_ = Ne[a:b], Ee[a:b]; print(' ', nm, len(n_), n_.mean().round(1).to_dict(), ' 추정 순부하', e_.mean().round(1).to_dict()); print(f'   모델-주말 {(n_["주말"] - n_["모델"]).mean():.1f} CI {blk((n_["주말"] - n_["모델"]).values)}, 모델-일사 {(n_["일사"] - n_["모델"]).mean():.1f} CI {blk((n_["일사"] - n_["모델"]).values)}, 추정 순부하 모델-일사 {(e_["일사"] - e_["모델"]).mean():.1f} CI {blk((e_["일사"] - e_["모델"]).dropna().values)}')
# ---------- D. 같은 할인 일수 ----------
print('\n[D] 같은 일수를 예보로 배정 (2일 전, 낮 수요-태양광 예측이 직전 365일 하위 q인 날)', flush=True)
Pd2 = walk(F2_, XON, 'net_s', '2023-07', '2025-12'); pdm = pd.DataFrame({'p': Pd2, 'd': F2_.date.reindex(Pd2.index), 'h': F2_.hour.reindex(Pd2.index)}).query('h in [12,13,14]').groupby('d').p.mean(); rd2 = F2_[F2_.hour.isin(hrs)].groupby('date').rad_mean.mean()
ev = pd.date_range('2025-01-01', '2025-12-31'); offr = OFF.reindex(ev).mean(); print(' 주말·공휴일 비중', round(float(offr), 3), int(OFF.reindex(ev).sum()), '일')
qp = pdm.rolling(365, min_periods=180).quantile(offr).shift(1); pick_m = (pdm <= qp).reindex(ev).fillna(False); qr = rd2.rolling(365, min_periods=180).quantile(1 - offr).shift(1); pick_r = (rd2 >= qr).reindex(ev).fillna(False); pick_o = (OFF.reindex(ev) == 1); ora = act.reindex(ev); pick_t = ora <= ora.quantile(offr); low = (Jw[hrs] <= 0).mean(axis=1).reindex(ev) * 100; lowd = (Jw[list(range(11, 17))].min(axis=1) <= 0).reindex(ev); sm = Jw[hrs].mean(axis=1).reindex(ev); E2 = E_.net_e.reindex(ev)
rr = []
for k, pk in [('주말·공휴일 고정', pick_o), ('맑은 날 규칙', pick_r), ('운영 모델', pick_m), ('완벽한 선택', pick_t)]:
    rr.append(dict(방식=k, 일수=int(pk.sum()), 수요태양광=round(float(ora[pk].mean()), 1), 추정순부하=round(float(E2[pk].mean()), 1), SMP=round(float(sm[pk].mean()), 1), 저가시간비중=round(float(low[pk].mean()), 1), 저가일포착=round(float(lowd[pk].sum() / lowd.sum() * 100), 1), 저가일정밀도=round(float(lowd[pk].mean() * 100), 1)))
TD_ = pd.DataFrame(rr); print(TD_.to_string(index=False)); print(' 2025 저가일', int(lowd.sum())); TD_.to_csv(f'{OG}/rob_discount_days.csv', index=False, encoding='utf-8-sig')
ww = 21; idx_ = ev; a_ = pd.Series(np.where(pick_m, ora, np.nan), index=idx_); b_ = pd.Series(np.where(pick_o, ora, np.nan), index=idx_)
def boot_diff(pa, pb, s):
    n = len(idx_); rng = np.random.default_rng(0); o = []
    for _ in range(3000):
        i = np.concatenate([(x + np.arange(ww)) % n for x in rng.integers(0, n, n // ww + 1)])[:n]; pa_, pb_ = pa.values[i], pb.values[i]; s_ = s.values[i]; o.append(np.nanmean(s_[pb_]) - np.nanmean(s_[pa_]))
    return np.round(np.nanpercentile(o, [2.5, 97.5]), 1).tolist()
print(' 수요-태양광 감소 운영-주말 CI', boot_diff(pick_m, pick_o, ora), ' 운영-맑은 날 CI', boot_diff(pick_m, pick_r, ora), ' 저가시간 비중 차이(주말-운영, 음수면 운영이 높음) CI', boot_diff(pick_m, pick_o, low))
# ---------- E. 그림 자료 ----------
print('\n[E] 낮은 날의 요일 구성', flush=True)
a25 = act['2025-01-01':'2025-12-31'].dropna(); q20 = a25 <= a25.quantile(.2); print(' 2025 수요-태양광 하위 20%', int(q20.sum()), '일 중 주말·공휴일', int(OFF.reindex(a25.index)[q20].sum()), '일, 비중', round(float(OFF.reindex(a25.index)[q20].mean() * 100), 1), '%', ' 주말·공휴일 중 하위 20%인 날', round(float(q20[OFF.reindex(a25.index) == 1].mean() * 100), 1), '%')
Cu = ((G.curt_solar > 0) | (G.curt_wind > 0)); cd = Cu[G.hour.between(9, 17)].groupby(G.date[G.hour.between(9, 17)]).max()['2023-01-01':'2024-05-30']; pr = pd.DataFrame({'dem': PV['dem'].reindex(cd.index).mean(), 'dem_c': PV['dem'].reindex(cd.index[cd]).mean(), 'net_c': PV['net_s'].reindex(cd.index[cd]).mean(), 'net_n': PV['net_s'].reindex(cd.index[~cd]).mean(), 'dem_n': PV['dem'].reindex(cd.index[~cd]).mean()}); pr.round(1).to_csv(f'{OG}/profile_curtail.csv', encoding='utf-8-sig'); print(pr.loc[[3, 8, 13, 19, 21]].round(0).to_string()); print('done')
