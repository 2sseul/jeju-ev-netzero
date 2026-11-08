"""육지(전남·광주) 4개 지점 2일 전 예보 수집"""
import subprocess,time,os,json,sys
O=sys.argv[1]
LAT="34.8167,34.7393,35.1729,34.5533"; LON="126.3815,127.7406,126.8916,126.5691"  # 목포, 여수, 광주, 해남
V="temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,shortwave_radiation,cloud_cover".split(',')
P=[('2024-01-01','2024-06-30'),('2024-07-01','2024-12-31'),('2025-01-01','2025-06-30'),('2025-07-01','2025-12-31'),('2026-01-01','2026-06-30')]
pv=','.join(f'{v}_previous_day2' for v in V)
for a,b in P:
    f=f'{O}/d2_{a}.json'
    for t in range(4):
        r=subprocess.run(['curl','-s','-m','280','-o',f,'-w','%{http_code}',f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pv}&start_date={a}&end_date={b}&timezone=Asia%2FSeoul'],capture_output=True,text=True).stdout
        ok=False
        try: ok=r=='200' and len(json.load(open(f)))==4
        except Exception: pass
        print(a,r,ok,flush=True)
        if ok: break
        time.sleep(30)
    time.sleep(3)
print('done')
