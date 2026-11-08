# 남는 재생에너지를 충전으로: AI 기상 예보 기반 제주 전기차 충전 할인일 제안

**Turning Surplus Renewables into EV Charging: AI Weather Forecast-Based Discount Days for Jeju**

서비스 이름: 제주 충전 예보

AI융합콜로퀴움 연계 지역문제해결 경진대회 (글로벌 넷제로 분야)

**시연 페이지:** https://2sseul.github.io/jeju-ev-netzero/

## 개요

제주는 재생에너지 설비가 약 750MW에 이르러 낮에는 전력이 남고 저녁에는 수요가 몰린다. 2024년 6월 재생에너지 입찰제가 도입된 뒤로는 남는 전력이 계통한계가격(SMP, 시간별 전력 도매가격)에 드러나며, 낮 SMP가 0원 이하인 날이 2025년에만 56일 나타났다.

정부는 봄·가을 주말과 공휴일 낮에 전기차 충전요금을 할인하고 있지만, 전력이 남는 날의 62%는 평일이었다. 이 프로젝트는 2~3일 전 기상 예보로 제주의 전력 여유를 예측하고, 이를 바탕으로 충전일과 할인일을 정하는 방법을 제안한다.

## 주요 결과

| 질문 | 결과 |
|---|---|
| 수요-태양광은 잉여를 나타내는가 | 공식 출력제어일(2023.01~2024.05, 517일 중 172일)을 AUC 0.916으로 구분 |
| 2일 전 수요 예측은 정확한가 | MAPE 3.18%. 전력거래소 하루 전 공개 예측 5.10%(편향 보정 4.71%) |
| 3일 중 하루를 고르면 | 충전 시간의 수요-태양광이 주말 우선 방식보다 66.7MW 낮음(95% CI 48.1~87.4) |
| 현행 할인 기간에 날짜만 바꾸면 | 2026년 봄·가을 할인 기간에서 저가일 포착 38.1% → 81.0% |

자세한 수치는 `analysis/out_grid/`의 CSV 파일에서 확인할 수 있다.

## 방법

- **데이터:** 전력거래소의 제주 수요, 발전량, 출력제어, SMP, 하루전 발전계획용 수요예측 자료(2019~2026년)와 Open-Meteo의 4개 기상 예보(ECMWF, GFS, ICON, GEM)의 1·2·3일 전 실행분
- **모델:** HistGradientBoosting. 실제 운영처럼 기상 예보와 달력 정보만 입력하고, 하이퍼파라미터는 조정하지 않았다.
- **검증:** 매달 그 이전 자료로만 다시 학습하는 walk-forward 방식. 2026년 시험은 2025년까지의 자료로 모델과 기준값을 고정했다.
- **비교와 통계:** 전력거래소 공개 예측, 선형 회귀, 주말 고정·맑은 날 규칙과 비교하고, Diebold-Mariano 검정과 21일 블록 부트스트랩 95% 신뢰구간으로 차이를 확인했다.

## 폴더 구성

```text
index.html                 시연 페이지 (GitHub Pages)
data/                      전력거래소 발전량·출력제어, 기상청 ASOS 관측자료
analysis/
├── raw/                   전력거래소 수요·SMP·공개 수요예측 자료
├── 수집/                  기상 예보와 SMP를 내려받는 스크립트
├── ens_price.py           다중 예보 저가일 예측, 충전 시뮬레이션
├── grid_forecast.py       수요·태양광·풍력·순부하 예측
├── robustness*.py         기준 모델 비교, 공식 예측 비교, 강건성 검증
├── forecast_today.py      매일 예측을 실측 전에 기록하고 시연 페이지 갱신
├── charts_*.py            그림 생성
├── out_grid/, out_price/  결과 표
├── 그림/                  결과 그림
├── 실시간기록/            실측 공개 전에 저장한 예측
└── 시연/                  시연 페이지 원본
```

## 실행 방법

기상 예보 원자료(약 80MB)는 용량 때문에 올리지 않았다. `analysis/수집/`의 스크립트로 내려받은 뒤 실행한다.

```bash
pip install -r requirements.txt
cd analysis
python 수집/ens_dl.py wx/ens          # 4개 예보 모델의 1·2·3일 전 예보
python 수집/fc_dl.py .                # 제주 4개 지점 1·2일 전 예보
python 수집/fc_dl_d2_early.py .       # 2022~2023년 2일 전 예보
python 수집/fc_dl_d3.py .             # 3일 전 예보
python grid_forecast.py               # 수급 예측
python robustness4.py                 # 최종 검증 (robustness2.py 먼저)
python forecast_today.py              # 오늘 예측 기록, 시연 페이지 갱신
```

## 데이터 출처

| 데이터 | 출처 |
|---|---|
| 제주 시간별 전력수요 | 한국전력거래소, 공공데이터포털 |
| 제주 태양광·풍력 발전량 | 한국전력거래소, 공공데이터포털 |
| 제주 출력제어 기록 | 한국전력거래소, 공공데이터포털 |
| 제주 시간별 SMP | 한국전력거래소, 전력통계정보시스템(EPSIS) |
| 제주 하루전 발전계획용 수요예측, 하루전 SMP | 한국전력거래소 |
| 1·2·3일 전 기상 예보, ERA5 재분석 | Open-Meteo |
| ASOS 시간자료 | 기상청 기상자료개방포털 |

## 참고문헌

1. 한국전력거래소, “제주 시범사업 관련 주요 Q&A,” 2024.
2. 한국전력거래소, “제주 시범사업 개요,” 2024.
3. 기후에너지환경부, “전기차 충전요금 가을철 주말·공휴일 낮시간대 할인 개시,” 정책브리핑, 2026.
4. F. X. Diebold and R. S. Mariano, “Comparing Predictive Accuracy,” Journal of Business & Economic Statistics, 13(3), 1995.
5. M. R. Bailey et al., “Show Me the Money! A Field Experiment on Electric Vehicle Charge Timing,” AEJ: Economic Policy, 17(2), 2025.
6. A. La Nauze et al., “Can electric vehicles aid the renewable transition? Evidence from a field experiment incentivizing midday charging,” Journal of Public Economics, 260, 2026.
