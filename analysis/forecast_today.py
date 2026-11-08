"""실시간 예측 기록. 사용: python forecast_today.py [기준일 YYYY-MM-DD] [--dry]
운영 모델은 best_match 없이 일관된 4개 예보(ECMWF, GFS, ICON, GEM)의 평균과 달력만 입력한다. 평가는 robustness4.py.
학습 자료를 고정해(수요 2026-09, 수요-태양광 2025-12, 저가일 FREEZE) 이후 예측은 처음 보는 날에 대한 기록이 된다.
내일·모레·글피 중 충전하기 좋은 날을 고르고 실시간기록/predictions.csv에 덧붙이며 시연 자료를 만든다."""
import os, sys, json, subprocess, datetime as dt
A_ = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]; DRY = '--dry' in sys.argv  # --dry: 예측만 출력하고 기록하지 않음
TODAY = args[0] if args else dt.date.today().isoformat(); T0_ = dt.date.fromisoformat(TODAY); TARGETS = [T0_ + dt.timedelta(days=k) for k in (1, 2, 3)]; END = TARGETS[-1].isoformat(); REC = f'{A_}/실시간기록'; os.makedirs(REC, exist_ok=True)
FREEZE = '2026-11-07'; LAT = "33.5141,33.2938,33.3868,33.2461"; LON = "126.5297,126.1628,126.8802,126.5653"; PY = sys.executable
def get(url, f):
    for _ in range(3):
        r = subprocess.run(['curl', '-s', '-m', '120', '-o', f, '-w', '%{http_code}', url], capture_output=True, text=True).stdout
        try:
            if r == '200' and len(json.load(open(f))) == 4: return
        except Exception: pass
    raise SystemExit(f'수집 실패 {f}')
subprocess.run([PY, f'{A_}/수집/smp_recent.py', f'{A_}/raw'], capture_output=True)
Vc = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,cloud_cover".split(','); Ve = "temperature_2m,precipitation,wind_speed_10m,shortwave_radiation,cloud_cover".split(',')
for L in (1, 2, 3): get(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={','.join(f'{v}_previous_day{L}' for v in Vc)}&start_date=2026-11-01&end_date={END}&timezone=Asia%2FSeoul", f'{A_}/wx/ch/d{L}_2026-11-01.json')
pve = ','.join(f'{v}_previous_day{L}' for L in (1, 2, 3) for v in Ve)
for m in ['best_match', 'ecmwf_ifs025', 'gfs_global', 'icon_global', 'gem_global']: get(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pve}&start_date=2026-11-01&end_date={END}&timezone=Asia%2FSeoul&models={m}", f'{A_}/wx/ens/{m}_2026-11-01.json')
src = open(f'{A_}/grid_forecast.py', encoding='utf8').read().split('\nT0, TD = ')[0].replace("'2026-11-01 00:00'", f"'{TARGETS[-1] + dt.timedelta(days=1)} 00:00'")
exec(src)
from sklearn.linear_model import LogisticRegression
M4 = ['ecmwf_ifs025', 'gfs_global', 'icon_global', 'gem_global']; vv = ['temp', 'rain', 'ws', 'rad', 'cloud']; win_ = [12, 13, 14]; XC4 = vv + ['hour', 'month', 'doy', 'dow', 'offday', 'holiday']
def m4(L):
    d = dates(sum(W[L][m][vv] for m in M4) / 4).dropna(subset=['rad']); d = d.join(G[['net_s', 'dem', 'dow', 'offday', 'holiday', 'month', 'doy']], how='left'); return d[d.date >= '2024-05-05']
F = {L: m4(L) for L in (1, 2, 3)}
# ---------- 고정 모델로 예측 ----------
md_ = hgb(F[2][(F[2].date <= '2026-09-30') & F[2].dem.notna()], XC4, 'dem'); mn_ = hgb(F[2][(F[2].date <= '2025-12-31') & F[2].net_s.notna()], XC4, 'net_s')
saved = pd.read_csv(f'{A_}/out_price/ens_probs_by_lead.csv', index_col=0, parse_dates=True); saved.columns = [1, 2, 3]
lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
cal2 = saved[2].dropna(); ycal = Lg.y.reindex(cal2.index).astype(int); platt = LogisticRegression(C=1e6).fit(lo(cal2.values), ycal.values); TIER = lambda p: '높음' if p >= .4 else ('보통' if p >= .1 else '낮음')
rows = []
for tgt, L in zip(TARGETS, (1, 2, 3)):
    ts = pd.Timestamp(tgt); fr = {m: daily_wx(W[L][m], f'_{m}', list(VARS.values())) for m in MODELS}; Dx = Lg.copy()
    for m in MODELS: Dx = Dx.join(fr[m])
    Dx = Dx[Dx.index >= START]; lists = [list(fr[m].columns) + CAL + LAG for m in MODELS]; t = Dx[(Dx.index <= FREEZE) & Dx.y.notna()]
    p = float(platt.predict_proba(lo(np.mean([clf(t, x).predict_proba(Dx.loc[[ts], x])[:, 1] for x in lists], axis=0)))[0, 1])
    c = F[L][F[L].date == ts]; cm = c[c.hour.isin(win_)]; ev_ = c[c.hour.between(18, 22)]
    rows.append(dict(issue_date=TODAY, target=str(tgt), lead=L, dow='월화수목금토일'[tgt.weekday()], offday=int(OFF[ts]), dem_mid=round(float(md_.predict(cm[XC4]).mean()), 1), dem_peak=round(float(md_.predict(ev_[XC4]).max()), 1), net_mid=round(float(mn_.predict(cm[XC4]).mean()), 1), p_low=round(p, 3), tier=TIER(p), cloud=round(float(cm.cloud.mean())), ws=round(float(cm.ws.mean()) / 3.6, 1), temp=round(float(cm.temp.mean()), 1), rad=round(float(cm.rad.mean()))))
P_ = pd.DataFrame(rows); P_['recommended'] = (P_.net_mid == P_.net_mid.min()).astype(int); print(P_.to_string(index=False))
if DRY: raise SystemExit('시험 실행: 기록하지 않음')
# ---------- 기록 ----------
fp = f'{REC}/predictions.csv'; old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame(); new = P_.assign(freeze=FREEZE, written_at=dt.datetime.now().isoformat(timespec='seconds'), model='4개 예보')
if len(old) and 'model' not in old: old['model'] = 'best_match'
if len(old) and (old.issue_date == TODAY).any() and (old[old.issue_date == TODAY].model == '4개 예보').any(): print('이미 기록된 기준일, 기록하지 않음')
else: pd.concat([old, new]).to_csv(fp, index=False, encoding='utf-8-sig'); print('기록 추가', fp)
# ---------- 시연 자료 ----------
sm = J.copy(); sm.columns = range(1, 25); smp_last = str(J.dropna(how='all').index.max().date()); log = pd.read_csv(fp); log['actual_mid'] = [float(sm[win_].mean(axis=1).get(pd.Timestamp(x), np.nan)) for x in log.target]; log['actual_low'] = [float(Lg.y.get(pd.Timestamp(x), np.nan)) for x in log.target]; scored = log.dropna(subset=['actual_low'])
# 등급 신뢰도: 각 월을 그 이전 자료로만 보정한 표본 외 값 (보정에 최소 6개월 필요해 2025-12부터)
oos = []
for per in pd.period_range('2025-12', '2026-11', freq='M'):
    tr_ = cal2[cal2.index < per.start_time]; te_ = cal2[cal2.index.to_period('M') == per]
    if len(te_): oos.append(pd.Series(LogisticRegression(C=1e6).fit(lo(tr_.values), ycal.reindex(tr_.index).values).predict_proba(lo(te_.values))[:, 1], index=te_.index))
pc = pd.concat(oos); tiers = {}
for k, g in pd.DataFrame({'t': pc.map(TIER), 'y': ycal.reindex(pc.index)}).groupby('t'): tiers[k] = dict(days=len(g), hit=int(g.y.sum()), rate=round(float(g.y.mean()), 3))
print('표본 외 등급', tiers)
hist = [dict(date=str(i.date()), tier=TIER(v), actual=int(ycal[i])) for i, v in pc['2026-09-01':].items()]
fc25 = pd.read_csv(f'{OG}/final_charge_2025.csv', index_col=0).net; oe = [dict(구간='확인', 묶음=363, 첫날=fc25['첫날 고정'], 주말=fc25['주말 우선'], 일사=fc25['맑은 날 규칙'], 운영모델=fc25['운영 모델'], 완벽=fc25['완벽한 선택'], 달성률=79.0)]; disc = pd.read_csv(f'{OG}/final_discount_2026.csv').to_dict('records')
json.dump(dict(asof=TODAY, freeze=FREEZE, smp_last=smp_last, forecasts=json.loads(P_.to_json(orient='records', force_ascii=False)), tiers=tiers, history=hist, op_eval=oe, discount=disc, log_scored=json.loads(scored[['issue_date', 'target', 'lead', 'tier', 'recommended', 'actual_low', 'actual_mid']].to_json(orient='records', force_ascii=False)), log_days=int(log.issue_date.nunique())), open(f'{A_}/시연/demo_data.json', 'w'), ensure_ascii=False, indent=1)
subprocess.run([PY, f'{A_}/시연/build_demo.py']); print('채점된 기록', len(scored))
