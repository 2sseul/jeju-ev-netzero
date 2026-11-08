import sys, glob, os, zipfile, io
import pandas as pd
R=sys.argv[1]
def rd(f,**k):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,**k)
        except Exception: pass
fr=[]
for f in sorted(glob.glob(f'{R}/dem/*.bin')):
    d=rd(f).iloc[:,:25]; d.columns=['date']+[f'h{i}' for i in range(1,25)]
    d['date']=pd.to_datetime(d['date'],errors='coerce'); d['ver']=os.path.basename(f); fr.append(d)
hs=[f'h{i}' for i in range(1,25)]
dem=pd.concat(fr).dropna(subset=['date']).sort_values(['date','ver']).drop_duplicates('date',keep='last')
dem[hs]=dem[hs].apply(pd.to_numeric,errors='coerce')
big=dem[hs].median(axis=1)>1e4; dem.loc[big,hs]=dem.loc[big,hs]/1000   # 2023-09 이전 kWh -> MWh
dem=dem[dem.date>='2021-01-01'].drop(columns='ver')
print('demand',dem.shape,dem.date.min().date(),dem.date.max().date(),'gaps',365*6-len(dem)); print(dem.groupby(dem.date.dt.year).h12.median().round(0).to_dict())
dem.to_csv(f'{R}/demand_all.csv',index=False)
g=[]
for y in ['2024','2025']:
    d=rd(f'{R}/gen/{y}.bin',dtype=str).iloc[:,:5]; d.columns=['date','h','region','fuel','mwh']; g.append(d)
g=pd.concat(g); print(g[g.region.str.contains('제주')].groupby(['region','fuel']).size().to_dict())
j=g[g.region.str.contains('제주')].copy(); j['mwh']=pd.to_numeric(j.mwh,errors='coerce'); j['date']=pd.to_datetime(j.date); j['h']=j.h.astype(int)
print(j.groupby([j.date.dt.year,'fuel','region']).mwh.agg(['count','sum','max']).round(0).to_string())
j.to_csv(f'{R}/gen_jeju_2425.csv',index=False)
z=zipfile.ZipFile(f'{R}/f_15132422.bin')
for i in z.infolist():
    name=i.filename.encode('cp437').decode('cp949','ignore'); d=rd(io.BytesIO(z.read(i))); dt=pd.to_datetime(d.iloc[:,0],errors='coerce')
    print('ZIP',name,d.shape,dt.min().date(),dt.max().date(),dt.dt.year.value_counts().sort_index().to_dict())
