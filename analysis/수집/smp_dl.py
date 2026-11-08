import subprocess,sys,time,re,os,calendar,json
R=sys.argv[1]; UA='Mozilla/5.0'; ck=f'{R}/ck.txt'
subprocess.run(['curl','-s','-m','30','-A',UA,'-c',ck,'-o','/dev/null','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202'])
def q(b,e):
    return subprocess.run(['curl','-s','-m','60','-A',UA,'-b',ck,'-X','POST','-d',f'beginDate={b}&endDate={e}&selYear=&selMonth=&selKind=jeju&locale=','https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'],capture_output=True,text=True).stdout
s=q('20240601','20240602'); print('sample',len(s)); print(s[:900])
open(f'{R}/sample.txt','w').write(s)
if len(sys.argv)>2:
    for y in range(2021,2027):
        for m in range(1,13):
            if (y,m)>(2026,10): break
            f=f'{R}/{y}{m:02d}.txt'
            if os.path.exists(f) and os.path.getsize(f)>200: continue
            o=q(f'{y}{m:02d}01',f'{y}{m:02d}{calendar.monthrange(y,m)[1]}'); open(f,'w').write(o); time.sleep(1.5)
    print('done',len(os.listdir(R)))
