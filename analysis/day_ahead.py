"""① 실제 하루 전 예보 검증. 설비 규모는 직전 365일 p99.5(미래 정보 없음)."""
import os, sys, json, glob, warnings, unicodedata as ud
import numpy as np, pandas as pd, holidays
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap, r2_score as r2, f1_score
warnings.filterwarnings('ignore')
S=os.path.dirname(os.path.abspath(__file__)); R=f'{S}/raw'; P=os.path.dirname(S)
ALL=[os.path.join(r,f) for r,_,fs in os.walk(P) for f in fs if f.endswith('.csv')]
def find(*k):
    h=[a for a in ALL if all(x in ud.normalize('NFC',a) for x in k)]; assert len(h)==1,(k,h); return h[0]
SITES=['jeju','gosan','seongsan','seogwipo']
def wx(path,suf=''):
    d=json.load(open(path)); o=[]
    for s,x in zip(SITES,d):
        h=pd.DataFrame(x['hourly']); h['ts']=pd.to_datetime(h.time); h=h.drop(columns='time').set_index('ts'); h.columns=[c.replace(suf,'') for c in h.columns]
        r=np.deg2rad(h.wind_direction_10m); h['wdsin']=np.sin(r); h['wdcos']=np.cos(r); h=h.drop(columns='wind_direction_10m')
        h=h.rename(columns={'temperature_2m':'temp','relative_humidity_2m':'hum','precipitation':'rain','wind_speed_10m':'ws','shortwave_radiation':'rad','cloud_cover':'cloud'})
        h.columns=[f'{c}_{s}' for c in h.columns]; o.append(h)
    w=pd.concat(o,axis=1)
    for v in ['temp','hum','rain','ws','rad','cloud']: w[f'{v}_mean']=w[[f'{v}_{s}' for s in SITES]].mean(axis=1)
    w['ws3_mean']=w.ws_mean**3
    return w
FC=wx(f'{S}/wx/fc.json','_previous_day1'); ER=wx(f'{S}/wx/era.json'); WC=list(FC.columns)
# 전력
def kpx(f,name):
    d=pd.read_csv(f,encoding='cp949'); d=d.melt(id_vars=d.columns[0],var_name='h',value_name=name); d['h']=d.h.str.replace('시','').astype(int)
    d['ts']=pd.to_datetime(d[d.columns[0]])+pd.to_timedelta(d.h,unit='h'); return d.set_index('ts')[name]
sol=kpx(find('한시간단위','태양광'),'solar'); wnd=kpx(find('한시간단위','풍력'),'wind')
j=pd.read_csv(f'{R}/gen_jeju_2425.csv',parse_dates=['date']); j['ts']=j.date+pd.to_timedelta(j.h,unit='h')
s2=j[(j.fuel=='태양광')].groupby('ts').mwh.sum(); w2=j[(j.fuel=='풍력')].groupby('ts').mwh.sum()
G=pd.DataFrame({'solar':pd.concat([sol,s2[s2.index>sol.index.max()]]),'wind':pd.concat([wnd,w2[w2.index>wnd.index.max()]])}).sort_index()
cs=pd.read_csv(find('출력제어횟수','태양광'),encoding='utf-8-sig'); cw=pd.read_csv(find('출력제어횟수','풍력'),encoding='cp949'); cw.columns=cs.columns
hr=[f'{i}시' for i in range(1,25)]
def cl(d,num):
    x=d[['일자']+hr].melt(id_vars='일자',var_name='h',value_name='v'); x['h']=x.h.str.replace('시','').astype(int); x['ts']=pd.to_datetime(x['일자'])+pd.to_timedelta(x.h,unit='h')
    x['v']=pd.to_numeric(x.v,errors='coerce').fillna(0) if num else (x.v=='출력제어').astype(float); return x.groupby('ts').v.max()
G['curt_solar']=cl(cs,False); G['curt_wind']=cl(cw,True); G[['curt_solar','curt_wind']]=G[['curt_solar','curt_wind']].fillna(0)
G=G[G.index>='2020-01-01']; G['wind_pot']=G.wind+G.curt_wind
# 미래 정보 없는 규모: 직전 365일 p99.5, 하루 지연
daily=lambda c:G[c].rolling('365D').quantile(.995).shift(24)
G['cap_s']=daily('solar'); G['cap_w']=daily('wind_pot'); G['solar_n']=G.solar/G.cap_s; G['wind_n']=G.wind_pot/G.cap_w
def frame(W):
    d=G.join(W,how='inner'); d['date']=(d.index-pd.Timedelta(hours=1)).normalize(); d['hour']=d.index.hour.where(d.index.hour!=0,24)
    d['month']=d.date.dt.month; d['doy']=d.date.dt.dayofyear; d['dow']=d.date.dt.dayofweek
    kr=holidays.KR(years=range(2020,2027)); d['holiday']=d.date.dt.date.map(lambda x:x in kr).astype(int); d['offday']=((d.dow>=5)|(d.holiday==1)).astype(int)
    d['curtailed']=((d.curt_solar>0)|(d.curt_wind>0)).astype(int); return d
XA=WC+['hour','month','doy']
def reg(t,y): return HR(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,random_state=0).fit(t[XA],t[y])
def fitAB(t): return reg(t[t.curt_solar==0],'solar_n'),reg(t,'wind_n')
def add(d,a,b):
    d=d.copy(); d['pA']=a.predict(d[XA]).clip(0); d['pB']=b.predict(d[XA]).clip(0); d['PA']=d.pA*d.cap_s; d['PB']=d.pB*d.cap_w; d['PR']=d.PA+d.PB; return d
def daily_feat(d):
    m=d[d.hour.between(10,16)]; g=m.groupby('date'); a=d.groupby('date')
    o=pd.DataFrame({'pA_mid':g.pA.mean(),'pB_mid':g.pB.mean(),'pA_max':g.pA.max(),'pB_max':g.pB.max(),'PA_abs':g.PA.mean(),'PB_abs':g.PB.mean(),'PR_max':g.PR.max(),
      'rad_mid':g.rad_mean.mean(),'cloud_mid':g.cloud_mean.mean(),'ws_mid':g.ws_mean.mean(),'temp_mid':g.temp_mean.mean(),'hum_mid':g.hum_mean.mean(),'rain_mid':g.rain_mean.sum(),'temp_day':a.temp_mean.mean(),'ws_day':a.ws_mean.mean()})
    o['PR_abs']=o.PA_abs+o.PB_abs; o['pR_mid']=o.pA_mid+o.pB_mid; o['cdd']=(o.temp_day-22).clip(lower=0); o['hdd']=(12-o.temp_day).clip(lower=0)
    for c in ['month','doy','dow','offday','holiday','curtailed']: o[c]=a[c].max() if c=='curtailed' else a[c].first()
    o['loss']=a.curt_wind.sum(); return o
XC=['pA_mid','pB_mid','pA_max','pB_max','PA_abs','PB_abs','PR_max','PR_abs','pR_mid','rad_mid','cloud_mid','ws_mid','temp_mid','hum_mid','rain_mid','temp_day','ws_day','cdd','hdd','month','doy','dow','offday','holiday']
XW=[c for c in XC if not c.startswith(('p','P'))]; XCAL=['month','doy','dow','offday','holiday']
def clf(t,X): return HC(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=0).fit(t[X],t.curtailed)
def run(Wtr,Wte,tr0,tr1,te0,te1,name):
    Dtr=frame(Wtr); Dtr=Dtr[(Dtr.date>=tr0)&(Dtr.date<=tr1)].dropna(subset=WC+['solar_n','wind_n']); Dte=frame(Wte); Dte=Dte[(Dte.date>=te0)&(Dte.date<=te1)].dropna(subset=WC)
    # 분기 블록 OOF
    q=Dtr.date.dt.to_period('Q'); oof=[]
    for k in q.unique():
        a,b=fitAB(Dtr[q!=k]); oof.append(add(Dtr[q==k],a,b))
    tr=daily_feat(pd.concat(oof)); a,b=fitAB(Dtr); h=add(Dte,a,b); te=daily_feat(h); y=te.curtailed.values
    qd=tr.index.to_period('Q'); out={}
    for nm,X in [('full',XC),('weather',XW),('calendar',XCAL)]:
        ps,ys=[],[]
        for k in qd.unique():
            if tr[qd!=k].curtailed.nunique()<2: continue
            m=clf(tr[qd!=k],X); ps.append(m.predict_proba(tr[qd==k][X])[:,1]); ys.append(tr[qd==k].curtailed.values)
        p0,y0=np.concatenate(ps),np.concatenate(ys); ts=np.arange(.1,.7,.02); thr=float(ts[int(np.argmax([f1_score(y0,(p0>=t).astype(int)) for t in ts]))])
        p=clf(tr,X).predict_proba(te[X])[:,1]; yh=(p>=thr).astype(int); k=int(y.sum()); top=np.zeros(len(p),int); top[np.argsort(-p)[:k]]=1
        out[nm]=dict(AUC=round(auc(y,p),3),PR=round(ap(y,p),3),thr=round(thr,2),days=int(yh.sum()),prec=round((y&yh).sum()/max(yh.sum(),1),3),rec=round((y&yh).sum()/y.sum(),3),F1=round(f1_score(y,yh),3),loss=round(te.loss[yh==1].sum()/max(te.loss.sum(),1),3),rec_topk=round((y&top).sum()/k,3))
        if nm=='full': P_=p
    tA=h[h.curt_solar==0]; dm=h.hour.between(7,19)
    res=dict(name=name,n=len(te),pos=int(y.sum()),base=round(y.mean(),3),trainpos=round(tr.curtailed.mean(),3),R2A=round(r2(tA.solar,tA.PA),3),R2B=round(r2(h.wind_pot,h.PB),3),**out)
    print(json.dumps(res,ensure_ascii=False)); return res,te.assign(p=P_),h
print('[기간 데이터]',G.index.max(),' 2024 태양광 max',G['2024'].solar.max() if False else G.loc['2024-01-01':'2024-06-30'].solar.max(),'cap_s',G.cap_s.loc['2024-03-01 12:00'].round(1),'cap_w',G.cap_w.loc['2024-03-01 12:00'].round(1))
# 예보 품질: ASOS 2023 대비
wa=[]
for yr in ['23']:
    w=pd.read_csv(find('OBS_ASOS',f'_{yr}.csv'),encoding='cp949'); w['ts']=pd.to_datetime(w['일시']); wa.append(w)
w=wa[0]; w=w.rename(columns={'풍속(m/s)':'ws','기온(°C)':'temp','일사(MJ/m2)':'rad','전운량(10분위)':'cloud','지점명':'st'})
mp={'제주':'jeju','고산':'gosan','성산':'seongsan','서귀포':'seogwipo'}; w['st']=w.st.map(mp)
print('[예보 오차: 2023 ASOS 대비, 낮 10~16시 일평균]')
for st in ['jeju','gosan']:
    a=w[w.st==st].set_index('ts'); a=a[(a.index.hour>=10)&(a.index.hour<=16)]; f=FC.reindex(a.index); e=ER.reindex(a.index)
    def st_(o,m,lbl,scale=1):
        x=pd.DataFrame({'o':pd.to_numeric(o,errors='coerce'),'m':m*scale}).dropna(); dd=x.groupby(x.index.normalize()).mean(); c=np.corrcoef(dd.o,dd.m)[0,1]; return f"{lbl} corr={c:.2f} MAE%={100*(dd.o-dd.m).abs().mean()/dd.o.mean():.0f}"
    print(' ',st,st_(a.ws,f[f'ws_{st}'],'풍속(fc)',1/3.6),'|',st_(a.ws,e[f'ws_{st}'],'풍속(era)',1/3.6),'|',st_(a.rad.fillna(0),f[f'rad_{st}'],'일사(fc)',0.0036),'|',st_(a.rad.fillna(0),e[f'rad_{st}'],'일사(era)',0.0036),'|',st_(a.temp,f[f'temp_{st}'],'기온(fc)'))
R_={}
print('\n[S1] 시험 2024-01~05 (입찰제 이전)')
R_['S1_fc'],te1,h1=run(FC,FC,'2022-07-02','2023-12-31','2024-01-01','2024-05-31','예보→예보')
R_['S1_er'],te1e,_=run(ER,ER,'2022-07-02','2023-12-31','2024-01-01','2024-05-31','ERA5→ERA5 동일기간')
R_['S1_er_long'],_,_=run(ER,ER,'2021-01-01','2023-12-31','2024-01-01','2024-05-31','ERA5 3년 학습→ERA5')
R_['S1_er_fc'],_,_=run(ER,FC,'2021-01-01','2023-12-31','2024-01-01','2024-05-31','ERA5 3년 학습→예보')
print('\n[S2] 시험 2023-07~12')
R_['S2_fc'],te2,_=run(FC,FC,'2022-07-02','2023-06-30','2023-07-01','2023-12-31','예보→예보')
R_['S2_er'],_,_=run(ER,ER,'2022-07-02','2023-06-30','2023-07-01','2023-12-31','ERA5→ERA5 동일기간')
print('\n[S3] 시험 2023 전체 (README와 대응)')
R_['S3_er'],te3,_=run(ER,ER,'2021-01-01','2022-12-31','2023-01-01','2023-12-31','ERA5→ERA5')
R_['S3_er_fc'],_,_=run(ER,FC,'2021-01-01','2022-12-31','2023-01-01','2023-12-31','ERA5→예보')
# 규칙 기준선과 CI (S1)
t=te1; y=t.curtailed.values; rng=np.random.default_rng(0)
def rule(nm,m):
    m=m.astype(int); print(f' {nm}: days={m.sum()} prec={(y&m).sum()/max(m.sum(),1):.3f} rec={(y&m).sum()/y.sum():.3f} F1={f1_score(y,m):.3f} loss={t.loss[m==1].sum()/t.loss.sum():.3f}')
print('\n[S1 기준선]'); rule('매일',np.ones(len(y),bool)); rule('봄(3,4,5월)',t.month.isin([3,4,5]).values); rule('주말공휴일',t.offday.values==1)
A=[]; n=len(y)
for _ in range(2000):
    s=rng.integers(0,n,n//7+1); i=np.concatenate([(x+np.arange(7))%n for x in s])[:n]
    if y[i].min()==y[i].max(): continue
    A.append(auc(y[i],t.p.values[i]))
print(' 예보 AUC 95% CI',np.round(np.percentile(A,[2.5,97.5]),3).tolist())
print(' 월별', t.assign(yh=(t.p>=R_['S1_fc']['full']['thr']).astype(int)).groupby('month').agg(actual=('curtailed','sum'),rec=('yh','sum')).T.to_dict())
out=f'{S}/out_da'; os.makedirs(out,exist_ok=True); json.dump(R_,open(f'{out}/results.json','w'),ensure_ascii=False,indent=1); te1.to_csv(f'{out}/daily_2024.csv'); te1e.to_csv(f'{out}/daily_2024_era.csv'); te2.to_csv(f'{out}/daily_2023h2.csv')
