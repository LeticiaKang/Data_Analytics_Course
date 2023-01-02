import pandas as pd
import pandas as pd
import numpy as np
df = pd.read_csv('221118_health.csv')
df.columns = ['Date','Click'] 
# 이동평균법은 주기와 추세를 쉽게 구할 수 있다.
# 이동평균법은 t시점의 에러를 구하는 것을 목표로 한다.
# 잔차만 잘 예측하면 우리는 좋은 예측을 할 수 있다.

# 시계열 데이터를 다루는 set_index
df = df.set_index("Date")
print(df)

# print(type(df.index))
df.index = pd.to_datetime(df.index)
# print(type(df.index))

# 등간척도 확인
# df.index[1] = df.index[0]

import matplotlib.pyplot as plt

#시계열 그래프 시각화 하기, x축은 날짜, y축은 클릭양
plt.plot(df.index, df["Click"])
plt.show()

# 추세(트랜드) 구하기
# 주기를 먼저 정해야 하는데, 임의로 7로 설정
season = 7
trend = []
for a in range(len(df["Click"])):
    # 슬라이싱을 통해서 앞, 뒤 3개를 불러옴
    df_slice = df['Click'][(a-(season//2)) : (a+(season//2)+1)]
    # 평균 구하기
    df_mean = np.mean(df_slice)
    # 트렌드에 추가하기
    trend.append(df_mean)
print('trend\n',trend)

# n을 기준으로 앞뒤에 주기를 볼때, 빈값이 있다면 mean을 구할 수 없다.
# 그래서 nan을 넣어주는 것.
for b in range(season//2):
    trend[(b+1)*(-1)] = np.nan
print('trend\n',trend)

# 주기성과 잔차를 배제하고 보기 위함임.
plt.plot(df.index, trend)
plt.show()

detrend = [] # 주기성 + 잔차
for k, n in enumerate(df['Click']):
    # detrend는 원본데이터에서 트렌드를 뺀거
    detrend.append(n-trend[k])
print('detrend\n', detrend)

# 테스트
plt.plot(df.index, detrend)
plt.show()

# 주기 구하기(평균구하기)
seasonal = []
for i in range(season):
    imsi = []
    # i 부터 7씩 건너뛰어서 하나씩 반환
    for w in detrend[i::season]:
        # nan이 아니면 imsi에 추가하기
        if np.isnan(w) == False:
            imsi.append(w)
        else:
            pass
    seasonal.append(np.mean(imsi))

print('seasonal\n',seasonal)

# 요일별로 추세를 알 수 있고, 토요일에 낮다는 시각정 정보를 알 수 있다.
# 월요일에 급격하게 높아지는데, 월요병 때문에 사람들이 쇼핑한다는 것을 알 수 있다.
plt.plot(['tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'mon'], seasonal)
plt.show()

seasonal_subtract = []
# season은 주기데이터
for i in range(len(detrend)//len(seasonal)):
    for w in seasonal:
        seasonal_subtract.append(w)
for i in seasonal[:len(detrend)%len(seasonal)]:
    seasonal_subtract.append(i)
print('seasonal_subtract\n',seasonal_subtract)

# 잔차 구하기
residual = []
for a,i in enumerate(seasonal_subtract):
    residual.append(detrend[a]-i)
print('residual\n',residual)

