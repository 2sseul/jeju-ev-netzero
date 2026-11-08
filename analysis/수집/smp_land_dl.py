"""EPSIS 육지 시간별 SMP 월별 수집 → raw/smp_land.csv. 사용: python 수집/smp_land_dl.py raw"""
import subprocess,sys,time,re,calendar,datetime as dt
import pandas as pd
R=sys.argv[1]; UA='Mozilla/5.0'; ck=f'{R}/ck.txt'; T=dt.date.today()
subprocess.run(['curl','-s','-m','30','-A',UA,'-c',ck,'-o','/dev/null','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202'])
rows=[]
for y in range(2024,T.year+1):
    for m in range(1,13):
        if (y,m)>(T.year,T.month): break
        b,e=f'{y}{m:02d}01',f'{y}{m:02d}{calendar.monthrange(y,m)[1]}'
        s=subprocess.run(['curl','-s','-m','60','-A',UA,'-b',ck,'-X','POST','-d',f'beginDate={b}&endDate={e}&selYear=&selMonth=&selKind=land&locale=','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'],capture_output=True,text=True).stdout
        for x in re.finditer(r'((?:c\d+\s*=\s*textFormmat\("[^"]*",count\);\s*)+)(.*?)gridData\.push\(\{"Date":"(\d{4}/\d{2}/\d{2})"',s,re.S):
            v=dict((int(a),c) for a,c in re.findall(r'c(\d+)\s*=\s*textFormmat\("([^"]*)"',x.group(1))); rows.append((x.group(3),*[float(v[k].replace(',','')) for k in range(1,25)]))
        time.sleep(1.5)
d=pd.DataFrame(rows,columns=['date']+[f'c{i}' for i in range(1,25)]); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date'); d.to_csv(f'{R}/smp_land.csv',index=False); print(d.shape,d.date.min().date(),d.date.max().date(),'<=0 낮',int((d[[f'c{i}' for i in range(11,17)]].min(axis=1)<=0).sum()))
