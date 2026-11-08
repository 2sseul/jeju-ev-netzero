"""출력제한 라벨에 수요 특징을 더한 효과 (예보→예보, 2024-01~05 / 2023-07~12). day_ahead.py의 준비부를 재사용한다."""
import os
A_ = os.path.dirname(os.path.abspath(__file__))
src = open(f'{A_}/day_ahead.py', encoding='utf8').read().split('\nR_={}')[0].replace('__file__', repr(f'{A_}/day_ahead.py'))
exec(src)
dem = pd.read_csv(f'{A_}/raw/demand_all.csv', parse_dates=['date']).set_index('date'); dm = dem[[f'h{i}' for i in range(11, 17)]].mean(axis=1)
kr = holidays.KR(years=range(2020, 2027)); of = pd.Series([int(x.weekday() >= 5 or x.date() in kr) for x in dm.index], index=dm.index)
base = {}
for t in dm.index:
    w = dm[(dm.index >= t - pd.Timedelta(days=16)) & (dm.index <= t - pd.Timedelta(days=3))]; s = w[of.reindex(w.index) == of[t]]; base[t] = s.mean() if len(s) else np.nan
base = pd.Series(base); _df = daily_feat
def daily_feat(d):
    o = _df(d); o['dem_base'] = base.reindex(o.index); o['net'] = o.dem_base - o.PR_abs; o['re_share'] = o.PR_abs / o.dem_base; return o
print('수요 없음')
a1 = run(FC, FC, '2022-07-02', '2023-12-31', '2024-01-01', '2024-05-31', '2024-01~05')[0]; a2 = run(FC, FC, '2022-07-02', '2023-06-30', '2023-07-01', '2023-12-31', '2023-07~12')[0]
XC = XC + ['dem_base', 'net', 're_share']
print('수요 추가')
b1, t1, _ = run(FC, FC, '2022-07-02', '2023-12-31', '2024-01-01', '2024-05-31', '2024-01~05'); b2, t2, _ = run(FC, FC, '2022-07-02', '2023-06-30', '2023-07-01', '2023-12-31', '2023-07~12')
yh = (t2.p >= b2['full']['thr']).astype(int); print('2023 하반기 월별 실제/추천', t2.assign(yh=yh).groupby('month').agg(actual=('curtailed', 'sum'), rec=('yh', 'sum')).T.to_dict())
