import pandas as pd,numpy as np,json,collections
d=pd.read_pickle('raw/smp/parsed.pkl'); d['date']=pd.to_datetime(d.date)
for c in d.columns[1:]: d[c]=pd.to_numeric(d[c].astype(str).str.replace(',',''),errors='coerce')
d=d.sort_values('date').drop_duplicates('date').set_index('date'); hs=[f'c{i}' for i in range(1,25)]
print(d.shape,d.index.min().date(),d.index.max().date(),'NaN',int(d[hs].isna().sum().sum()), 'c25=max?',bool((d.c25==d[hs].max(axis=1)).mean()>.98),'c26=min?',bool((d.c26==d[hs].min(axis=1)).mean()>.98))
d[hs].to_csv('raw/smp_jeju.csv')
mid=d[[f'c{i}' for i in range(11,17)]]  # 10~16시
s=pd.DataFrame({'m':mid.min(axis=1),'avg':mid.mean(axis=1),'eve':d[[f'c{i}' for i in range(18,23)]].mean(axis=1),'zero':(d[hs]<=0).sum(axis=1)})
print(s.groupby(s.index.to_period('Q')).agg(days=('m','size'),midmin=('m','mean'),midavg=('avg','mean'),eve=('eve','mean'),d_le0=('m',lambda x:int((x<=0).sum())),neg=('m',lambda x:int((x<0).sum())),h_le0=('zero','sum')).round(1).to_string())
p=s['2024-06-01':]; print('월별 낮 SMP<=0 일수'); print(p.groupby([p.index.year,p.index.month]).m.apply(lambda x:int((x<=0).sum())).unstack().to_string())
print('시간대별 <=0 비중(2025~):'); x=d['2025-01-01':][hs]; print((x<=0).sum().to_dict())
print('요일별 낮SMP<=0 비율:',(p.m<=0).groupby(p.index.dayofweek).mean().round(3).to_dict())
for L in [1,2,3]:
    j=json.load(open(f'wx/fc_d{L}.json')); h=j[0]['hourly']; k=[x for x in h if 'wind_speed' in x][0]
    print('d',L,dict(sorted(collections.Counter(t[:4] for t,v in zip(h['time'],h[k]) if v is not None).items())), len(h['time']))
