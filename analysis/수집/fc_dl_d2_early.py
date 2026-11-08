"""제주 4개 지점 2일 전 예보 (2022-06~2023-12). 순부하 평가 기간을 늘리기 위해 추가 수집. 인자: 추가분석 폴더 경로."""
import json, os, subprocess, sys, time
S = sys.argv[1]; LAT = "33.5141,33.2938,33.3868,33.2461"; LON = "126.5297,126.1628,126.8802,126.5653"
V = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,cloud_cover".split(',')
pv = ','.join(f'{v}_previous_day2' for v in V)
for a, b in [('2022-06-01', '2022-12-31'), ('2023-01-01', '2023-06-30'), ('2023-07-01', '2023-12-31')]:
    f = f'{S}/wx/ch/d2_{a}.json'
    for t in range(4):
        r = subprocess.run(['curl', '-s', '-m', '280', '-o', f, '-w', '%{http_code}', f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly={pv}&start_date={a}&end_date={b}&timezone=Asia%2FSeoul'], capture_output=True, text=True).stdout
        try: ok = r == '200' and len(json.load(open(f))) == 4
        except Exception: ok = False
        print(a, r, os.path.getsize(f) if os.path.exists(f) else 0, ok, flush=True)
        if ok: break
        time.sleep(30)
    time.sleep(3)
