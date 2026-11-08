"""기대 효과 시나리오. extras_model.py를 먼저 실행해야 한다. 참여율·이행률은 가정값이다."""
import os
import pandas as pd
A = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(f'{A}/out_smp/daily_calibrated.csv', parse_dates=['date']).set_index('date')['2025-11-01':'2026-10-31']
EV, KWH, COMPLY = 49335, 43.0 / 5, 0.5   # 누적 보급 대수, 제주 일평균 43.0km(전 차종) / 5km/kWh, 추천일 이행률
EF_NAT, EF_SFOC = 0.4173, 11740 / 24032   # 국가 전력배출계수, 기후솔루션 제주 시나리오 내재값(tCO2/MWh)
SURPLUS = 26197 + 6489   # 2023년 풍력 실측 + 태양광 추정(MWh)
hit = d[(d.yh == 1) & (d.y == 1)]; miss = d[(d.yh == 1) & (d.y == 0)]; off = d[d.offday == 1]; offhit = off[off.y == 1]
print(f'최근 12개월 저가일 {int(d.y.sum())}일 | 모델 추천 {int(d.yh.sum())}일(적중 {len(hit)}) | 주말·공휴일 {len(off)}일(적중 {len(offhit)})')
print(f'차량 1대 일평균 {KWH:.1f} kWh, 전체 {EV * KWH / 1000:.0f} MWh/일, 2023 제한일 평균 잉여 {SURPLUS / 120:.0f} MWh/일')
spread = (d.eve - d.w1114); print(f'저녁-11~14시 SMP 차이: 적중일 {spread[hit.index].mean():.1f}, 오경보일 {spread[miss.index].mean():.1f}, 주말 적중일 {spread[offhit.index].mean():.1f} 원/kWh')
rows = []
for part in [.01, .05, .10, .20]:
    per_day = EV * part * KWH * COMPLY / 1000
    e_hit = per_day * len(hit); e_rec = per_day * d.yh.sum()
    won = (spread[d[d.yh == 1].index].sum() * per_day * 1000) / 1e6
    e_off = per_day * len(offhit)
    rows.append({'참여율': f'{part:.0%}', '참여 대수': int(EV * part), '하루 이동(MWh)': round(per_day, 1), '저가일 흡수(MWh/년)': round(e_hit), '주말 고정 흡수(MWh/년)': round(e_off), '2023 잉여 대비': f'{e_hit / SURPLUS:.1%}', 'CO2 감축 상한(t/년, 국가~제주)': f'{e_hit * EF_NAT:.0f}~{e_hit * EF_SFOC:.0f}', '도매가 차이(백만원/년)': round(won, 1)})
t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f'{A}/out_smp/impact.csv', index=False, encoding='utf-8-sig')
print(f'내재 배출계수 {EF_SFOC:.3f}')
