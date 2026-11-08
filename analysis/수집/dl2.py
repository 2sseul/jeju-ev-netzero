import json, re, subprocess, sys, os, zipfile, io, glob
import pandas as pd
R=sys.argv[1]; UA="Mozilla/5.0"
def get(id_, uddi, out):
    if os.path.exists(out) and os.path.getsize(out)>100: return 'cached'
    m=json.loads(subprocess.run(['curl','-s','-m','40','-A',UA,'-X','POST',f'https://www.data.go.kr/tcs/dss/selectFileDataDownload.do?publicDataPk={id_}&publicDataDetailPk={uddi}&fileDetailSn=1'],capture_output=True,text=True).stdout)
    info=m.get('dataSetFileDetailInfo') or {}
    a=m.get('atchFileId') or info.get('atchFileId'); sn=m.get('fileDetailSn') or info.get('fileDetailSn') or 1
    if not a: return 'no atch '+str(info.get('dataNm'))
    return subprocess.run(['curl','-s','-m','240','-A',UA,'-L','-o',out,'-w','%{http_code} %{size_download}',f'https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={a}&fileDetailSn={sn}'],capture_output=True,text=True).stdout
os.makedirs(f'{R}/dem',exist_ok=True); os.makedirs(f'{R}/gen',exist_ok=True)
sw=json.loads(subprocess.run(['curl','-s','-m','30','https://infuser.odcloud.kr/oas/docs?namespace=15065239/v1'],capture_output=True,text=True).stdout)
for p,v in sw['paths'].items():
    name=v['get']['summary'].split('_')[-1]; uddi=p.split('/')[-1]
    print('dem',name,get('15065239',uddi,f'{R}/dem/{name}.bin'))
print('gen24',get('15065269','uddi:57e66d86-ea1d-4efb-9eb3-a6d575336841',f'{R}/gen/2024.bin'))
print('gen25',get('15065269','uddi:76850689-e74c-402e-bfec-9142976b96c8',f'{R}/gen/2025.bin'))
def rd(f,**k):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,**k)
        except Exception: pass
fr=[]
for f in sorted(glob.glob(f'{R}/dem/*.bin')):
    if open(f,'rb').read(2)==b'PK': print(' zip',f); continue
    d=rd(f)
    if d is None or d.shape[1]<25: print(' skip',f, None if d is None else d.shape); continue
    d=d.iloc[:,:25]; d.columns=['date']+[f'h{i}' for i in range(1,25)]; d['date']=pd.to_datetime(d['date'],errors='coerce'); d['ver']=os.path.basename(f)
    print(' ',os.path.basename(f),d.shape,d.date.min(),d.date.max(),'median h12 %.3g'%pd.to_numeric(d.h12,errors='coerce').median()); fr.append(d)
dem=pd.concat(fr).dropna(subset=['date']).sort_values(['date','ver']).drop_duplicates('date',keep='last')
hs=[f'h{i}' for i in range(1,25)]; dem[hs]=dem[hs].apply(pd.to_numeric,errors='coerce')
print('dem total',dem.shape,dem.date.min(),dem.date.max(), dem.groupby(dem.date.dt.year).h12.median().round(1).to_dict())
dem.to_csv(f'{R}/demand_all.csv',index=False)
g=[]
for y in ['2024','2025']:
    d=rd(f'{R}/gen/{y}.bin',dtype=str); print(y,d.shape,list(d.columns),d.iloc[:,2].unique()[:25],d.iloc[:,3].unique()); g.append(d)
g=pd.concat(g); g.columns=['date','h','region','fuel','mwh']; j=g[g.region.str.contains('제주')].copy(); j['mwh']=pd.to_numeric(j.mwh,errors='coerce'); j['date']=pd.to_datetime(j.date); j['h']=j.h.astype(int)
print(j.groupby([j.date.dt.year,'fuel']).mwh.agg(['count','sum','max']).round(0))
j.to_csv(f'{R}/gen_jeju_2425.csv',index=False)
z=zipfile.ZipFile(f'{R}/f_15132422.bin')
for i in z.infolist():
    name=i.filename.encode('cp437').decode('cp949','ignore')
    d=rd(io.BytesIO(z.read(i)))
    c=d.columns[0]; dt=pd.to_datetime(d[c],errors='coerce')
    print('ZIP',name,d.shape,dt.min(),dt.max(),dt.dt.year.value_counts().sort_index().to_dict())
    if '제어지역' in d: print('   지역',d['제어지역'].value_counts().head(8).to_dict())
