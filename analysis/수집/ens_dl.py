"""제주 4개 지점, 5개 예보 모델의 1·2·3일 전 예보 수집. 사용: python 수집/ens_dl.py wx/ens"""
import subprocess,time,os,json,sys
O=sys.argv[1]
LAT="33.5141,33.2938,33.3868,33.2461"; LON="126.5297,126.1628,126.8802,126.5653"
V="temperature_2m,precipitation,wind_speed_10m,shortwave_radiation,cloud_cover".split(',')
P=[('2024-05-01','2024-12-31'),('2025-01-01','2025-06-30'),('2025-07-01','2025-12-31'),('2026-01-01','2026-06-30'),('2026-07-01','2026-11-10')]
pv=','.join(f'{v}_previous_day{L}' for L in (1,2,3) for v in V)
for m in ['best_match','ecmwf_ifs025','gfs_global','icon_global','gem_global']:
    for a,b in P:
        f=f'{O}/{m}_{a}.json'
        if os.path.exists(f):
            try:
                if len(json.load(open(f)))==4: continue
            except Exception: pass
        for t in range(5):
            r=subprocess.run(['curl','-s','-m','290','-o',f,'-w','%{http_code}',f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pv}&start_date={a}&end_date={b}&timezone=Asia%2FSeoul&models={m}'],capture_output=True,text=True).stdout
            ok=False
            try: ok=r=='200' and len(json.load(open(f)))==4
            except Exception: pass
            print(m,a,r,ok,flush=True)
            if ok: break
            time.sleep(40)
        time.sleep(3)
print('done')
