# ( 시애틀 자전거 이동량 분석 )

# ( 시계열 데이터 다루기 )
# -. 타임스탬프 -> 특정 시점
# -. 시간 간격과 기간 -> 특정 시작점과 종료점 사이의 시간,
# 기간은 일반적으로 각 간격이 일정하고 서로 겹치지 않는 특별한 경우의 시간 간격
# -. 시간 델타(time delta)나 지속 기간(duration)은 정확한 시간 길이

import pandas as pd
import numpy as np

# 파이썬 날짜와 시간
from datetime import datetime
# print(datetime(year=2018, month=11, day=12)) #시분초를 지정하지 않아 시분초는 00으로 출력됨
# tm = datetime(year=2015, month=7, day=4); print(tm) #시분초를 지정하지 않아 시분초는 00으로 출력됨

# dateutil을 이용해 다양한 문자열 형태로부터 날짜를 해석
# parsersms 다양한 날짜와 시간의 문자열 값을 해석해서 datetime으로 인식한다.
# from dateutil import parser
# date=parser.parse("12th of Nov, 2018") #오타가 있으면 인식 못함
# print(date)
# print(date.strftime('%Y-%m-%d-%H-%M-%S-%A'))
#strftime은 문자열에서 각각 해당하는 정보를 가져온다.
# Y(연도), m(월), d(일), H(시), M(분), S(초), A(요일)

# NumPy의 dataetime64
import numpy as np
date = np.array('2018-11-12', dtype=np.datetime64)
# print(date)
# print(date + np.arange(12))
#0부터 12까지의 수로 이루어진 데이터를 date에 더해서 2018-11-12부터 순차로 날짜를 만든다.
# pd.date_range(시작날짜, 끝날짜)를 입력하면 알아서 인덱스를 생성해준다.
# print(pd.date_range('2018-11-12', '2018-11-25'))

# print(np.datetime64('2018-11-12'))
# print(np.datetime64('2018-11-13 12:00'))

# ( Pandas 시계열 데이터 조작 )
index = pd.DatetimeIndex(['2017-10-12', '2017-11-12',
                          '2018-10-12', '2018-11-12'])
data = pd.Series([0, 1, 2, 3], index=index)
# 날짜를 인덱스로 가진 DataFrame이 만들어짐
# print(data)
# print(data['2017-10-12':'2017-11-12'])
# # 날짜 인덱스 슬라이싱 (일반적인 방법)
# print(data['2018'])
# # 날짜의 연도만 입력해도 2018년에 해당하는 row를 추출할 수 있다.
#
# # https://pandas.pydata.org/pandas-docs/stable/timeseries.htm
# # 데이터를 직관적으로 구성하고 접근하기 위해 날짜와 시간을 인덱스로 사용하는 능력은 pandas 시계열도구
# # 중요한 부분.
# # 인덱스를 가진 데이터의 이점 (연산하는 동안 자동 정렬, 직관저인 데이터 슬라이싱 및 접근)
# # pandas는 주로 금융환경에서 개발 되었기 때문에 몇몇 금융 데이터에 특호된 전용도구 포함
#
# # https://wikidocs.net/4373
import pandas_datareader.data as web
import datetime
start = datetime.datetime(2016, 2, 19)
end = datetime.datetime(2016, 3, 4)
#
# gs = web.DataReader("078930.KS", "yahoo", start, end) #datareader API에서 야후의 078930.KS의 데이터 정보를 가져옴
# gs = web.get_data_yahoo("078930.KS", start, end)
# print(gs.head())
# print(gs.info())

import matplotlib.pyplot as plt
# gs = web.DataReader("078930.KS", "yahoo")
# plt.plot(gs['Adj Close'])
# plt.show()
# # print(gs.index) # 날짜 순으로 되어있음
# plt.plot(gs.index, gs['Adj Close'])
# plt.show()
#
# ( 시애틀 자전거 수 시각화 )
# https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD
# FremontBridge.csv --> goo.gl/o3FkTM
import pandas as pd
data = pd.read_csv("FremontBridge.csv", index_col = 'Date', parse_dates=True)
# print(data.head(5))

# 열이름을 단축하고 total 추가
# data.columns = ['West','East']  # west와 east를 바꿔서 넣어 놓았음.
data.columns = ['East','West']
data['Total'] = data.eval('West+East')
# print(data.head(5))
#요약 통계
# print(data.describe())

# # %matplotlib inline #주피터에서 사용함 plt.show()대신
import seaborn
seaborn.set()
# data.plot()
# plt.ylabel('Hourly Bicycle Count');
# plt.show()
#
# # 25,000개의 시간별 표본 이해하기 어려움
# # 데이터를 주 단위로 리 샘플링
# weekly = data.resample('W').sum()
# weekly.plot(style=[':','--','-'])
# #plt.ylabel('Weekly bicycle count')
# plt.show()
# # 겨울보다 여름에 자전거를 더 많이 타며
# # 특정 계절에는 자전거 사용 횟수가 주 마다 달라진다.

# # 30일 이동 평균(rolling)
# daily = data.resample('D').sum()
# daily.rolling(30, center=True).sum().plot(style=[':','--','-'])
# plt.ylabel('mean hourly count')
# plt.show()
#
# # 가우스평활(Gaussian smoothing) 적용
# # 가우스 윈도우(Gaussian window) 같은 윈도우 함수를 사용해 롤링 평균을 부드럽게 표현
# # 윈도우 폭(50일)과 윈도우내 가우스 폭(10) 지정
# daily.rolling(50, center=True,
#  win_type='gaussian').sum(std=10).plot(style=[':','--','-'])
#
#
# # 하루의 시간대를 기준으로 한 함수로 평균 통행량을 보고 싶을때
# by_time = data.groupby(data.index.time).mean()
# hourly_ticks = 4*60*60*np.arange(6)
# by_time.plot(xticks=hourly_ticks, style=[':','--','-'])
# plt.show()
# # 아침 8시, 저녁 5시 무렵에 많이 사용
# # 동서가 확연하게 나누어짐.. 출근 사용량
#
# # 요일별 통행량은?
# by_weekday = data.groupby(data.index.dayofweek).mean()
# by_weekday.index = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
# by_weekday.plot(style=[':','--','-'])
# # 주중에 많고.. 주말에 적음.. 출퇴근 용 가능성..
#
# # 주중과 주말의 시간대별 추이
# # 데이터를 주말을 표시하는 플래그와 시간대별로 분류
# weekend = np.where(data.index.weekday <5, 'Weekday', 'Weekend')
# np.where(조건, 참인경우 변경, 거짓인 경우 변경)
# print(weekend) ; print(data)
# # ?? 138
# by_time = data.groupby([weekend, data.index.time]).mean()

# # 다중 서브플롯
# import matplotlib.pyplot as plt
# fgs, ax = plt.subplots(1, 2, figsize=(14,5))
# by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays',
#  xticks=hourly_ticks, style=[':','--','-'])
# by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',
#  xticks=hourly_ticks, style=[':','--','-'])
# plt.show()
# 주중에는 출퇴근시간에 주말에는 낮시간에 이용량이 많다는 것을 알 수 있음
# 출퇴근 패턴에 영향을 미치는 날씨, 온도, 연중 시기 등 기타 요인 분석이 필요함

# # < 시애틀의 자전거 통행량 예측하기 >
counts = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('BicycleWeather.csv', index_col='DATE', parse_dates=True)
# print(counts.head(3)); print(weather.head(3))

# 일별 총 자전거 통행량을 계산하여 별도의 DataFrame에 넣기
daily = counts.resample('d').sum()
# day를 기준으로 사용시간을 모두 sum
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']] # total만 남기고 싶음 - 어차피 합친 거니까
# 요일을 나타내는 열 추가
# 처리속도를 빠르게 하기 위해 1,0으로 구분하여 요일을 인코딩함
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
 daily[days[i]] = (daily.index.dayofweek == i).astype(float)
# print(daily)

# 휴일에 자전거를 타는 사람
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016') #2012년부터 2016년까지의 휴일 가져오기
daily = daily.join(pd.Series(1, index=holidays, name='holiday')) 
# 휴일을 Series로 만들고 value에 1 넣기 
# 휴일 = 1인 series를 daily와 합치면 휴일이 아닌 곳은 NaN이 들어감
daily['holiday'].fillna(0, inplace=True)
# NaN인 곳을 0로 채우기
print(daily)

# # 17페이지
# # 일조시간에 자전거를 타는 사람
def hours_of_daylight(date, axis=23.44, latitude=47.61):
#  # 해당 날짜의 일조시간 계산
 days = (date - pd.datetime(2000, 12, 21)).days
 m = (1. -np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
 return 24 * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.
daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
# daily[['daylight_hrs']].plot()
# plt.ylim(8, 17)
# plt.show()

# 데이터에 평균 기온과 전체 강수량 추가
# 인치 단위의 강수량과 더불어 날이 건조했는지(강수량이 0) 알려주는 플래그 추가
# 기온은 섭씨 1/10도 단위, 섭씨 1도 단위로 변환
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (c)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
# 강수량은 1/10mm 단위; 인치 단위로 변환
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
daily = daily.join(weather[['PRCP', 'Temp (c)', 'dry day']])
print(daily.head(4))