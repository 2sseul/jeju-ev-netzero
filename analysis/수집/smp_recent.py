"""2026-11-01 이후 제주 SMP 수집. 사용: python 수집/smp_recent.py raw"""
import subprocess,re,sys
import pandas as pd
import datetime as dt
END=(dt.date.today()+dt.timedelta(days=1)).strftime('%Y%m%d')
R=sys.argv[1]; UA='Mozilla/5.0'; ck=R+'/ck.txt'
subprocess.run(['curl','-s','-m','30','-A',UA,'-c',ck,'-o','/dev/null','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202'])
s=subprocess.run(['curl','-s','-m','60','-A',UA,'-b',ck,'-X','POST','-d',f'beginDate=20261101&endDate={END}&selYear=&selMonth=&selKind=jeju&locale=','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'],capture_output=True,text=True).stdout
open(R+'/202611.txt','w').write(s)
rows=[]
for m in re.finditer(r'((?:c\d+\s*=\s*textFormmat\("[^"]*",count\);\s*)+)(.*?)gridData\.push\(\{"Date":"(\d{4}/\d{2}/\d{2})"',s,re.S):
    v=dict((int(a),b) for a,b in re.findall(r'c(\d+)\s*=\s*textFormmat\("([^"]*)"',m.group(1))); rows.append((m.group(3),*[float(v[k]) for k in range(1,25)]))
d=pd.DataFrame(rows,columns=['date']+[f'c{i}' for i in range(1,25)]); print(d.date.tolist()); print(d[['date','c12','c13','c14','c15']].to_string())
d['date']=pd.to_datetime(d.date); d.sort_values('date').to_csv(R+'/smp_recent.csv',index=False)
