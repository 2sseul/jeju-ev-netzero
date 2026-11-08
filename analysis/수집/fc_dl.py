import subprocess,time,os,json,sys
S=sys.argv[1]; os.makedirs(f'{S}/wx/ch',exist_ok=True)
LAT="33.5141,33.2938,33.3868,33.2461"; LON="126.5297,126.1628,126.8802,126.5653"
V="temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,cloud_cover".split(',')
P=[('2024-01-01','2024-06-30'),('2024-07-01','2024-12-31'),('2025-01-01','2025-06-30'),('2025-07-01','2025-12-31'),('2026-01-01','2026-06-30'),('2026-07-01','2026-10-31')]
for L in [1,2]:
    for a,b in P:
        f=f'{S}/wx/ch/d{L}_{a}.json'
        if os.path.exists(f) and os.path.getsize(f)>1000: continue
        pv=','.join(f'{v}_previous_day{L}' for v in V)
        for t in range(4):
            r=subprocess.run(['curl','-s','-m','280','-o',f,'-w','%{http_code}',f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pv}&start_date={a}&end_date={b}&timezone=Asia%2FSeoul'],capture_output=True,text=True).stdout
            print(L,a,r,os.path.getsize(f) if os.path.exists(f) else 0,flush=True)
            if r=='200': break
            time.sleep(30)
        time.sleep(3)
print('done')
