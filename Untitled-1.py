# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from gp.messunsicherheiten import Value

# %% [markdown]
# 1.Aufgabe(Impedanz)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일의 경로
file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/1.Aufgabe/impedanz.csv' 

# CSV 파일 읽기
data = pd.read_csv(file_path, sep='\t', skiprows = 1 ,engine='python')

# 데이터가 제대로 읽혔는지 확인
# print(data.head())

# # 참고!! 첫 번째 행(단위 행) 제거 및 숫자로 변환
# data = data.iloc[1:].astype(float)

# 주파수(Frequency)와 임피던스(Impedance) 추출
frequency = data['Hz']
impedance = data['Ohm.1']

# print(frequency)
# print(impedance)

# 주어진 값
Rs = 10


# 함수 정의
def fit_function(f, Rs, Rp, C):
    Z = np.sqrt((Rs**2 + Rp**2 + 2*Rs*Rp + 4 * Rs**2 * Rp**2 * np.pi**2 * f**2 * C**2) / ((4 * Rp**2 * np.pi**2 * f**2 * C**2)+1))
    return Z

# 초기 추정값
# p0 = [Rs, 1E+4, 20*1e-12]

# curve_fit을 사용하여 피팅
popt, pcov = curve_fit(fit_function, frequency, impedance, p0=[Rs, 2E+5, 2.1*1e-12])


# 최적 파라미터
Rp = popt[1]
C = popt[2]

# 오차
Rp_err = np.sqrt(pcov[1, 1])
C_err = np.sqrt(pcov[2, 2])


# 결과 출력
print(f'Rp = {Rp:.4e} ± {Rp_err:.4e}')
print(f'C = {C:.4e} ± {C_err:.4e}')

# 피팅된 임피던스 계산
fitted_impedance = fit_function(frequency, Rs, Rp, C)

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(frequency, impedance, marker='o', color='b', label='Messdaten')  # 원래 데이터
plt.plot(frequency, fitted_impedance, 'r-', label='Fit-Funktion')  # 피팅된 함수


# 범위 설정
# plt.xlim(-0.05, 200000)

# x축을 로그 스케일로 설정
plt.xscale('log')

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Frequenz (Hz)')  # x축 레이블
plt.ylabel('Impedanz (Ohm)')  # y축 레이블

# 범례 추가
plt.legend()

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})
# 격자 추가
plt.grid(True)

# 결과 저장
plt.savefig('Impedanz.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %%
# import pandas as pd
# import matplotlib.pyplot as plt

# # CSV 파일의 경로
# file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/1.Aufgabe/impedanz.csv'  # CSV 파일 경로

# # CSV 파일을 읽습니다
# data = pd.read_csv(file_path)

# # 데이터 프레임의 열 이름을 출력
# print("Columns in the data:", data.columns)

# # 데이터의 첫 몇 행을 출력
# print("First few rows:")
# print(data.head())



# %% [markdown]
# 2. IV-Messungen

# %% [markdown]
# Rot-LED

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_rot.csv'  # CSV 파일 경로

data = pd.read_csv(file_path, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage = data['V.1']
current = data['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1 = voltage[voltage <= 1.5]  
y1 = current[voltage <= 1.5]

# 두번 째 fit 함수를 위해 데이터 분리
x2 = voltage[voltage >= 1.5]  
y2 = current[voltage >= 1.5]

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1, _ = curve_fit(fit_function, x1, y1)  # 첫 번째 직선 피팅
popt2, _ = curve_fit(fit_function, x2, y2)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1, b1 = popt1
m2, b2 = popt2

# 오차 계산
m1_err = np.sqrt(_[0, 0])
b1_err = np.sqrt(_[1, 1])
m2_err = np.sqrt(_[0, 0])
b2_err = np.sqrt(_[1, 1])

# 결과 출력
print(f"y1 = ({m1:.10f} ± {m1_err:.10f}) * x + ({b1:.10f} ± {b1_err:.10f})")
print(f"y2 = ({m2:.10f} ± {m2_err:.10f}) * x + ({b2:.10f} ± {b2_err:.10f})")

val_m1 = Value(m1, m1_err)
val_b1 = Value(b1, b1_err)
val_m2 = Value(m2, m2_err)
val_b2 = Value(b2, b2_err)
# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection = (val_b2 - val_b1) / (val_m1 - val_m2)
y_intersection = fit_function(x_intersection, m1, b1)

# 교차점 출력
print( f'Us = {x_intersection}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit = np.linspace(0, 1.57, 100)
y1_fit = fit_function1(x1_fit, m1, b1)


# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage, current, marker='o', color='r', label='rot')  # 원래 데이터
plt.plot(x1_fit, y1_fit, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
plt.plot(x2, fit_function(x2, m2, b2), 'g-', label='Fitlinie 2')  # 두 번째 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})


# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_rot.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# log IV - Rot

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_rot.csv'  # CSV 파일 경로

data = pd.read_csv(file_path, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage = data['V.1']
current = data['A.1']

# voltage >= 0.5 그리고 log_current <= 0
filtered_data = data[(voltage >= 0.6) & (current > 0)]

# 조건을 만족하는 데이터 추출
voltage_filtered = filtered_data['V.1']
log_current_filtered = np.log(filtered_data['A.1'])


# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt, pcov = curve_fit(fit_function, voltage_filtered, log_current_filtered)  

# 피팅된 기울기와 절편
m, b = popt

# 오차 계산
m_err = np.sqrt(pcov[0, 0])
b_err = np.sqrt(pcov[1, 1])

# 결과 출력
print(f"y = ({m:.10f} ± {m_err:.10f}) * x + ({b:.10f} ± {b_err:.10f})")

val_m = Value(m, m_err)
val_b = Value(b, b_err)


# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered, log_current_filtered, marker='o', color='r', label='rot')  # 원래 데이터
plt.plot(voltage_filtered, fit_function(voltage_filtered, m, b), 'b-', label='Fitlinie')  # 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 결과 저장
plt.savefig('IV_rot_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# IV für alle LED

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_rot = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_rot.csv'  
file_orange = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_orange.csv'
# file_gruen = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_gruen.csv'
file_weiss = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_weiss.csv'

data_rot = pd.read_csv(file_rot, sep='\t', skiprows = 1 ,engine='python')
data_orange = pd.read_csv(file_orange, sep='\t', skiprows = 1 ,engine='python')
# data_gruen = pd.read_csv(file_gruen, sep='\t', skiprows = 1 ,engine='python')
data_weiss = pd.read_csv(file_weiss, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_rot = data_rot['V.1']
current_rot = data_rot['A.1']
voltage_orange = data_orange['V.1']
current_orange = data_orange['A.1']
# voltage_gruen = data_gruen['V.1']
# current_gruen = data_gruen['A.1']
voltage_weiss = data_weiss['V.1']
current_weiss = data_weiss['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1_r = voltage_rot[voltage_rot <= 1.5]  
y1_r = current_rot[voltage_rot <= 1.5]
x1_o = voltage_orange[voltage_orange <= 1.6]
y1_o = current_orange[voltage_orange <= 1.6]
# x1_g = voltage_gruen[voltage_gruen <= 1.5]
# y1_g = current_gruen[voltage_gruen <= 1.5]
x1_w = voltage_weiss[voltage_weiss <= 2.5]
y1_w = current_weiss[voltage_weiss <= 2.5]

# 두번 째 fit 함수를 위해 데이터 분리
x2_r = voltage_rot[voltage_rot >= 1.5]
y2_r = current_rot[voltage_rot >= 1.5]
x2_o = voltage_orange[voltage_orange >= 1.6]
y2_o = current_orange[voltage_orange >= 1.6]
# x2_g = voltage_gruen[voltage_gruen >= 1.5]
# y2_g = current_gruen[voltage_gruen >= 1.5]
x2_w = voltage_weiss[voltage_weiss >= 2.5]
y2_w = current_weiss[voltage_weiss >= 2.5]

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1_r, c1_r = curve_fit(fit_function, x1_r, y1_r)  # 첫 번째 직선 피팅
popt2_r, c2_r = curve_fit(fit_function, x2_r, y2_r)  # 두 번째 직선 피팅
popt1_o, c1_o = curve_fit(fit_function, x1_o, y1_o)  # 첫 번째 직선 피팅
popt2_o, c2_o = curve_fit(fit_function, x2_o, y2_o)  # 두 번째 직선 피팅
# popt1_g, _ = curve_fit(fit_function, x1_g, y1_g)  # 첫 번째 직선 피팅
# popt2_g, _ = curve_fit(fit_function, x2_g, y2_g)  # 두 번째 직선 피팅
popt1_w, c1_w = curve_fit(fit_function, x1_w, y1_w)  # 첫 번째 직선 피팅
popt2_w, c2_w = curve_fit(fit_function, x2_w, y2_w)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1_r, b1_r = popt1_r
m2_r, b2_r = popt2_r
m1_o, b1_o = popt1_o
m2_o, b2_o = popt2_o
# m1_g, b1_g = popt1_g
# m2_g, b2_g = popt2_g
m1_w, b1_w = popt1_w
m2_w, b2_w = popt2_w

# 오차 계산
m1_err_r = np.sqrt(c1_r[0, 0])
b1_err_r = np.sqrt(c1_r[1, 1])
m2_err_r = np.sqrt(c2_r[0, 0])
b2_err_r = np.sqrt(c2_r[1, 1])
m1_err_o = np.sqrt(c1_o[0, 0])
b1_err_o = np.sqrt(c1_o[1, 1])
m2_err_o = np.sqrt(c2_o[0, 0])
b2_err_o = np.sqrt(c2_o[1, 1])
# m1_err_g = np.sqrt(_[0, 0])
# b1_err_g = np.sqrt(_[1, 1])
# m2_err_g = np.sqrt(_[0, 0])
# b2_err_g = np.sqrt(_[1, 1])
m1_err_w = np.sqrt(c1_w[0, 0])
b1_err_w = np.sqrt(c1_w[1, 1])
m2_err_w = np.sqrt(c2_w[0, 0])
b2_err_w = np.sqrt(c2_w[1, 1])


# 결과 출력
print(f"First line y1_r = ({m1_r:.10f} ± {m1_err_r:.10f}) * x + ({b1_r:.10f} ± {b1_err_r:.10f})")
print(f"Second line y2_r = ({m2_r:.10f} ± {m2_err_r:.10f}) * x + ({b2_r:.10f} ± {b2_err_r:.10f})")
print(f"First line y1_o = ({m1_o:.10f} ± {m1_err_o:.10f}) * x + ({b1_o:.10f} ± {b1_err_o:.10f})")
print(f"Second line y2_o = ({m2_o:.10f} ± {m2_err_o:.10f}) * x + ({b2_o:.10f} ± {b2_err_o:.10f})")
# print(f"First line y1_g = ({m1_g:.10f} ± {m1_err_g:.10f}) * x + ({b1_g:.10f} ± {b1_err_g:.10f})")
# print(f"Second line y2_g = ({m2_g:.10f} ± {m2_err_g:.10f}) * x + ({b2_g:.10f} ± {b2_err_g:.10f})")
print(f"First line y1_w = ({m1_w:.10f} ± {m1_err_w:.10f}) * x + ({b1_w:.10f} ± {b1_err_w:.10f})")
print(f"Second line y2_w = ({m2_w:.10f} ± {m2_err_w:.10f}) * x + ({b2_w:.10f} ± {b2_err_w:.10f})")

val_m1_r = Value(m1_r, m1_err_r)
val_b1_r = Value(b1_r, b1_err_r)
val_m2_r = Value(m2_r, m2_err_r)
val_b2_r = Value(b2_r, b2_err_r)
val_m1_o = Value(m1_o, m1_err_o)
val_b1_o = Value(b1_o, b1_err_o)
val_m2_o = Value(m2_o, m2_err_o)
val_b2_o = Value(b2_o, b2_err_o)
# val_m1_g = Value(m1_g, m1_err_g)
# val_b1_g = Value(b1_g, b1_err_g)
# val_m2_g = Value(m2_g, m2_err_g)
# val_b2_g = Value(b2_g, b2_err_g)
val_m1_w = Value(m1_w, m1_err_w)
val_b1_w = Value(b1_w, b1_err_w)
val_m2_w = Value(m2_w, m2_err_w)
val_b2_w = Value(b2_w, b2_err_w)


# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection_r = (val_b2_r - val_b1_r) / (val_m1_r - val_m2_r)
y_intersection_r = fit_function(x_intersection_r, m1_r, b1_r)
x_intersection_o = (val_b2_o - val_b1_o) / (val_m1_o - val_m2_o)
y_intersection_o = fit_function(x_intersection_o, m1_o, b1_o)
# x_intersection_g = (val_b2_g - val_b1_g) / (val_m1_g - val_m2_g)
# y_intersection_g = fit_function(x_intersection_g, m1_g, b1_g)
x_intersection_w = (val_b2_w - val_b1_w) / (val_m1_w - val_m2_w)
y_intersection_w = fit_function(x_intersection_w, m1_w, b1_w)

# 교차점 출력
print( f'Us_r = {x_intersection_r}')
print( f'Us_o = {x_intersection_o}')
# print( f'Us_g = {x_intersection_g}')
print( f'Us_w = {x_intersection_w}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit_r = np.linspace(0, 1.57, 100)
y1_fit_r = fit_function1(x1_fit_r, m1_r, b1_r)
x1_fit_o = np.linspace(0, 1.6, 100)
y1_fit_o = fit_function1(x1_fit_o, m1_o, b1_o)
# x1_fit_g = np.linspace(0, 1.57, 100)
# y1_fit_g = fit_function1(x1_fit_g, m1_g, b1_g)
x1_fit_w = np.linspace(0, 2.67, 100)
y1_fit_w = fit_function1(x1_fit_w, m1_w, b1_w)


# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_rot, current_rot, marker='o', color='r', label='rot')  # 원래 데이터
plt.plot(x1_fit_r, y1_fit_r, 'b-')  # 첫 번째 피팅된 직선
plt.plot(x2_r, fit_function(x2_r, m2_r, b2_r), 'g-')  # 두 번째 피팅된 직선
plt.scatter(voltage_orange, current_orange, marker='o', color='orange', label='orange')  # 원래 데이터
plt.plot(x1_fit_o, y1_fit_o, 'b-')  # 첫 번째 피팅된 직선
plt.plot(x2_o, fit_function(x2_o, m2_o, b2_o), 'g-')  # 두 번째 피팅된 직선
# plt.scatter(voltage_gruen, current_gruen, marker='o', color='g', label='gruen')  # 원래 데이터
# plt.plot(x1_fit_g, y1_fit_g, 'b-', label='Fitlinie 1 gruen')  # 첫 번째 피팅된 직선
# plt.plot(x2_g, fit_function(x2_g, m2_g, b2_g), 'g-', label='Fitlinie 2 gruen')  # 두 번째 피팅된 직선
plt.scatter(voltage_weiss, current_weiss, marker='o', color='black', label='weiß')  # 원래 데이터
plt.plot(x1_fit_w, y1_fit_w, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
plt.plot(x2_w, fit_function(x2_w, m2_w, b2_w), 'g-', label='Fitlinie 2')  # 두 번째 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.0005, 0.023)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_alleled.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()




# %% [markdown]
# log IV für alle LED

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_path_r = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_rot.csv'  # CSV 파일 경로
file_path_o = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_orange.csv'  # CSV 파일 경로
# file_path_g = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_gruen.csv'  # CSV 파일 경로
file_path_w = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_weiss.csv'  # CSV 파일 경로

data_r = pd.read_csv(file_path_r, sep='\t', skiprows = 1 ,engine='python')
data_o = pd.read_csv(file_path_o, sep='\t', skiprows = 1 ,engine='python')
# data_g = pd.read_csv(file_path_g, sep='\t', skiprows = 1 ,engine='python')
data_w = pd.read_csv(file_path_w, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_r = data_r['V.1']
current_r = data_r['A.1']
voltage_o = data_o['V.1']
current_o = data_o['A.1']
# voltage_g = data_g['V.1']
# current_g = data_g['A.1']
voltage_w = data_w['V.1']
current_w = data_w['A.1']

# voltage >= 0.5 그리고 log_current <= 0
filtered_data_r = data_r[(voltage_r >= 0.6) & (current_r > 0)]
filtered_data_o = data_o[(voltage_o > 0) & (current_o > 0)]
# filtered_data_g = data_g[(voltage_g >= 0.6) & (current_g > 0)]
filtered_data_w = data_w[(voltage_w > 0.5) & (current_w > 0)]

# 조건을 만족하는 데이터 추출
voltage_filtered_r = filtered_data_r['V.1']
log_current_filtered_r = np.log(filtered_data_r['A.1'])
voltage_filtered_o = filtered_data_o['V.1']
log_current_filtered_o = np.log(filtered_data_o['A.1'])
# voltage_filtered_g = filtered_data_g['V.1']
# log_current_filtered_g = np.log(filtered_data_g['A.1'])
voltage_filtered_w = filtered_data_w['V.1']
log_current_filtered_w = np.log(filtered_data_w['A.1'])


# fit 함수를 위한 데이터 분리
x_r = data_r[(voltage_r < 1.6) & (voltage_r > 0.6) & (current_r > 0)]['V.1']
y_r = np.log(data_r[(voltage_r < 1.6) & (voltage_r >= 0.6) & (current_r > 0)]['A.1'])
x_o = data_o[(voltage_o < 1.7) & (voltage_o > 1.2) & (current_o > 0)]['V.1']
y_o = np.log(data_o[(voltage_o < 1.7) & (voltage_o > 1.2) & (current_o > 0)]['A.1'])
# x_g = data_g[(voltage_g >= 0.6) & (current_g > 0)]['V.1']
# y_g = np.log(data_g[(voltage_g >= 0.6) & (current_g > 0)]['A.1'])
x_w = data_w[(voltage_w < 2.65) & (voltage_w > 1.95) & (current_w > 0)]['V.1']
y_w = np.log(data_w[(voltage_w < 2.65) & (voltage_w > 1.95) & (current_w > 0)]['A.1'])



# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt_r, pcov_r = curve_fit(fit_function, x_r, y_r)  # 첫 번째 직선 피팅
popt_o, pcov_o = curve_fit(fit_function, x_o, y_o)  # 두 번째 직선 피팅
# popt_g, pcov_g = curve_fit(fit_function, x_g, y_g)  # 첫 번째 직선 피팅
popt_w, pcov_w = curve_fit(fit_function, x_w, y_w)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m_r, b_r = popt_r
m_o, b_o = popt_o
# m_g, b_g = popt_g
m_w, b_w = popt_w

# 오차 계산
m_r_err = np.sqrt(pcov_r[0, 0])
b_r_err = np.sqrt(pcov_r[1, 1])
m_o_err = np.sqrt(pcov_o[0, 0])
b_o_err = np.sqrt(pcov_o[1, 1])
# m_g_err = np.sqrt(pcov_g[0, 0])
# b_g_err = np.sqrt(pcov_g[1, 1])
m_w_err = np.sqrt(pcov_w[0, 0])
b_w_err = np.sqrt(pcov_w[1, 1])

# 결과 출력

print(f"y_r = ({m_r:.10f} ± {m_r_err:.10f}) * x + ({b_r:.10f} ± {b_r_err:.10f})")
print(f"y_o = ({m_o:.10f} ± {m_o_err:.10f}) * x + ({b_o:.10f} ± {b_o_err:.10f})")
# print(f"y_g = ({m_g:.10f} ± {m_g_err:.10f}) * x + ({b_g:.10f} ± {b_g_err:.10f})")
print(f"y_w = ({m_w:.10f} ± {m_w_err:.10f}) * x + ({b_w:.10f} ± {b_w_err:.10f})")

val_m_r = Value(m_r, m_r_err)
val_b_r = Value(b_r, b_r_err)
val_m_o = Value(m_o, m_o_err)
val_b_o = Value(b_o, b_o_err)
# val_m_g = Value(m_g, m_g_err)
# val_b_g = Value(b_g, b_g_err)
val_m_w = Value(m_w, m_w_err)
val_b_w = Value(b_w, b_w_err)


# Sätttigungsstrom & Idealitätsfaktor 계산
# Sättigungsstrom Is = np.exp(val_b)
Is_r = np.exp(b_r)
Is_o = np.exp(b_o)
# Is_g = np.exp(b_g)
Is_w = np.exp(b_w)

# Sättigungsstrom 오차 계산
Is_r_err = np.abs(Is_r * b_r_err)
Is_o_err = np.abs(Is_o * b_o_err)
# Is_g_err = np.abs(Is_g * b_g_err)
Is_w_err = np.abs(Is_w * b_w_err)


# Idealitätsfaktor n = e / (k * T * val_m)
e = 1.6 * 1e-19 
k = 1.38 * 1e-23 
T = 298 

n_r = e / (k * T * m_r)
n_o = e / (k * T * m_o)
# n_g = e / (k * T * val_m_g)
n_w = e / (k * T * m_w)

# Idealitätsfaktor 오차 계산
n_r_err = np.abs(n_r * m_r_err)
n_o_err = np.abs(n_o * m_o_err)
# n_g_err = np.abs(n_g * m_g_err)
n_w_err = np.abs(n_w * m_w_err)

# 결과 출력
print(f"Is_r = {Is_r:.10e} ± {Is_r_err:.10e}")
print(f"Is_o = {Is_o:.10e} ± {Is_o_err:.10e}")
# print(f"Is_g = {Is_g:.10e} ± {Is_g_err:.10e}")
print(f"Is_w = {Is_w:.10e} ± {Is_w_err:.10e}")
print(f"n_r = {n_r:.10f} ± {n_r_err:.10f}")
print(f"n_o = {n_o:.10f} ± {n_o_err:.10f}")
# print(f"n_g = {n_g:.10f} ± {n_g_err:.10f}")
print(f"n_w = {n_w:.10f} ± {n_w_err:.10f}")

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered_r, log_current_filtered_r, marker='o', color='r', label='rot')  # 원래 데이터
plt.plot(x_r, fit_function(x_r, m_r, b_r), 'b-')  # 피팅된 직선
plt.scatter(voltage_filtered_o, log_current_filtered_o, marker='o', color='orange', label='orange')  # 원래 데이터
plt.plot(x_o, fit_function(x_o, m_o, b_o), 'b-')  # 피팅된 직선
# plt.scatter(voltage_filtered_g, log_current_filtered_g, marker='o', color='g', label='gruen')  # 원래 데이터
# plt.plot(x_g, fit_function(x_g, m_g, b_g), 'b-', label='Fitlinie gruen')  # 피팅된 직선
plt.scatter(voltage_filtered_w, log_current_filtered_w, marker='o', color='black', label='weiß')  # 원래 데이터
plt.plot(x_w, fit_function(x_w, m_w, b_w), 'b-', label='Fitlinie')  # 피팅된 직선


# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})


# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_alleled_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()



# %% [markdown]
# grüne LED

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_gruen = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_gruen.csv'

data_gruen = pd.read_csv(file_gruen, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_gruen = data_gruen['V.1']
current_gruen = data_gruen['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1_g = voltage_gruen[voltage_gruen <= 1.5]
y1_g = current_gruen[voltage_gruen <= 1.5]

# 두번 째 fit 함수를 위해 데이터 분리
x2_g = voltage_gruen[voltage_gruen >= 1.5]
y2_g = current_gruen[voltage_gruen >= 1.5]

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1_g, c1_g = curve_fit(fit_function, x1_g, y1_g)  # 첫 번째 직선 피팅
popt2_g, c2_g = curve_fit(fit_function, x2_g, y2_g)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1_g, b1_g = popt1_g
m2_g, b2_g = popt2_g

# 오차 계산
m1_err_g = np.sqrt(c1_g[0, 0])
b1_err_g = np.sqrt(c1_g[1, 1])
m2_err_g = np.sqrt(c2_g[0, 0])
b2_err_g = np.sqrt(c2_g[1, 1])


# 결과 출력
print(f"y1 = ({m1_g:.10f} ± {m1_err_g:.10f}) * x + ({b1_g:.10f} ± {b1_err_g:.10f})")
print(f"y2 = ({m2_g:.10f} ± {m2_err_g:.10f}) * x + ({b2_g:.10f} ± {b2_err_g:.10f})")

val_m1_g = Value(m1_g, m1_err_g)
val_b1_g = Value(b1_g, b1_err_g)
val_m2_g = Value(m2_g, m2_err_g)
val_b2_g = Value(b2_g, b2_err_g)


# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection_g = (val_b2_g - val_b1_g) / (val_m1_g - val_m2_g)
y_intersection_g = fit_function(x_intersection_g, m1_g, b1_g)

# 교차점 출력
print( f'Us_g = {x_intersection_g}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit_g = np.linspace(0, 2.1, 100)
y1_fit_g = fit_function1(x1_fit_g, m1_g, b1_g)

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_gruen, current_gruen, marker='o', color='g', label='grün')  # 원래 데이터
plt.plot(x1_fit_g, y1_fit_g, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
plt.plot(x2_g, fit_function(x2_g, m2_g, b2_g), 'g-', label='Fitlinie 2')  # 두 번째 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.00001, 0.000065)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_green.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()



# %% [markdown]
# log IV - green

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/2.Aufgabe/IV_gruen.csv'  # CSV 파일 경로

data = pd.read_csv(file_path, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage = data['V.1']
current = data['A.1']

# voltage >= 0.5 그리고 log_current <= 0
filtered_data = data[(voltage >= 0.6) & (current > 0)]

# 조건을 만족하는 데이터 추출
voltage_filtered = filtered_data['V.1']
log_current_filtered = np.log(filtered_data['A.1'])


# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt, pcov = curve_fit(fit_function, voltage_filtered, log_current_filtered)  

# 피팅된 기울기와 절편
m, b = popt

# 오차 계산
m_err = np.sqrt(pcov[0, 0])
b_err = np.sqrt(pcov[1, 1])

# 결과 출력
print(f"y = ({m:.10f} ± {m_err:.10f}) * x + ({b:.10f} ± {b_err:.10f})")

val_m = Value(m, m_err)
val_b = Value(b, b_err)

#Sätttigungsstrom & Idealitätsfaktor 계산
# Sättigungsstrom Is = np.exp(val_b)
Is = np.exp(b)
# Sättigungsstrom 오차 계산
Is_err = np.abs(Is * b_err)
# Idealitätsfaktor n = e / (k * T * val_m)
e = 1.6 * 1e-19
k = 1.38 * 1e-23
T = 298
n = e / (k * T * m)
# Idealitätsfaktor 오차 계산
n_err = np.abs(n * m_err)

# 결과 출력
print(f"Is = {Is:.10e} ± {Is_err:.10e}")
print(f"n = {n:.10f} ± {n_err:.10f}")

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered, log_current_filtered, marker='o', color='g', label='grün')  # 원래 데이터
plt.plot(voltage_filtered, fit_function(voltage_filtered, m, b), 'b-', label='Fitlinie')  # 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_green_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# Tabelle von IV-Messungen der LEDs für Latex

# %%
import pandas as pd

# 데이터: (LED 색상, Schwellenspannung 값, Sättigungsstrom 값, Idealitätsfaktor 값)
data = [
    ('LED-Rot', (1.55, 0.15), (9.7780917798e-14, 6.6983634197e-14), (2.55, 1.31)),
    ('LED-Orange', (1.67, 0.08), (3.7089339634e-21, 3.3188996226e-21), (1.61, 0.93)),
    ('LED-Grün', (2.09, 0.15), (2.0597171124e-06, 7.7222724144e-08), (58.89, 0.71)),
    ('LED-Weiß', (2.63, 0.18), (2.2273523664e-30, 3.8958962736e-30), (1.64, 1.20)),
]

# 과학적 표기법을 사용하여 불확실성을 \pm 기호로 표시하는 함수
def format_with_uncertainty(value):
    nominal, uncertainty = value
    
    # 값이 1 이상이면 일반 소수점 두 자리 표현
    if nominal >= 1:
        return f"{nominal:.2f} ± {uncertainty:.2f}"
    
    # 값이 1 미만이면 과학적 표기법 사용
    nominal_str = f"{nominal:.2e}"
    nominal_base = float(nominal_str.split('e')[0])
    nominal_exponent = int(nominal_str.split('e')[-1])
    
    uncertainty_str = f"{uncertainty:.2e}"
    uncertainty_base = float(uncertainty_str.split('e')[0])
    uncertainty_exponent = int(uncertainty_str.split('e')[-1])
    
    if nominal_exponent == uncertainty_exponent:
        return f"({nominal_base:.2f} \\pm {uncertainty_base:.2f}) \\cdot 10^{{{nominal_exponent}}}"
    else:
        return f"{nominal_base:.2f} \\cdot 10^{{{nominal_exponent}}} \\pm {uncertainty_base:.2f} \\cdot 10^{{{uncertainty_exponent}}}"

# 데이터프레임 생성
df = pd.DataFrame(data, columns=['LED', 'Schwellenspannung $U_\\mathrm{S}$ (V)', 'Sättigungsstrom $I_\\mathrm{S}$ (A)', 'Idealitätsfaktor n'])

# 각 열에 대해 불확실성 포맷팅 적용
df['Schwellenspannung $U_\\mathrm{S}$ (V)'] = df['Schwellenspannung $U_\\mathrm{S}$ (V)'].apply(format_with_uncertainty)
df['Sättigungsstrom $I_\\mathrm{S}$ (A)'] = df['Sättigungsstrom $I_\\mathrm{S}$ (A)'].apply(format_with_uncertainty)
df['Idealitätsfaktor n'] = df['Idealitätsfaktor n'].apply(format_with_uncertainty)

# LaTex 표 생성
latex_table = df.to_latex(
    index=False, 
    escape=False,  # LaTex 특수문자 이스케이프 방지
    column_format='c' * len(df.columns)  # 모든 열 가운데 정렬
)

# LaTex 표 시작 및 끝 추가
latex_table = "\\begin{table}[H]\n    \\centering\n    \\caption[IV-Messungen]{verschiedene Proben}\n" + latex_table + "\n\\end{table}"

# 결과 출력
print(latex_table)


# %% [markdown]
# IV/log-IV - verschiedene Proben

# %% [markdown]
# B3297-IV

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_klein.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_gross.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_mittel.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')


# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1_k = voltage_klein[voltage_klein <= 3.7]
y1_k = current_klein[voltage_klein <= 3.7]
x1_g = voltage_gross[voltage_gross <= 3.2]
y1_g = current_gross[voltage_gross <= 3.2]
x1_m = voltage_mittel[voltage_mittel <= 3.3]
y1_m = current_mittel[voltage_mittel <= 3.3]
x1_L = voltage_L[voltage_L <= 3.2]
y1_L = current_L[voltage_L <= 3.2]

# 두번 째 fit 함수를 위해 데이터 분리
x2_k = voltage_klein[voltage_klein >= 3.7]
y2_k = current_klein[voltage_klein >= 3.7]
x2_g = voltage_gross[voltage_gross >= 3.2]
y2_g = current_gross[voltage_gross >= 3.2]
x2_m = voltage_mittel[voltage_mittel >= 3.3]
y2_m = current_mittel[voltage_mittel >= 3.3]
x2_L = voltage_L[voltage_L >= 3.2]
y2_L = current_L[voltage_L >= 3.2]

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1_k, c1_k = curve_fit(fit_function, x1_k, y1_k)  # 첫 번째 직선 피팅
popt2_k, c2_k = curve_fit(fit_function, x2_k, y2_k)  # 두 번째 직선 피팅
popt1_g, c1_g = curve_fit(fit_function, x1_g, y1_g)  # 첫 번째 직선 피팅
popt2_g, c2_g = curve_fit(fit_function, x2_g, y2_g)  # 두 번째 직선 피팅
popt1_m, c1_m = curve_fit(fit_function, x1_m, y1_m)  # 첫 번째 직선 피팅
popt2_m, c2_m = curve_fit(fit_function, x2_m, y2_m)  # 두 번째 직선 피팅
popt1_L, c1_L = curve_fit(fit_function, x1_L, y1_L)  # 첫 번째 직선 피팅
popt2_L, c2_L = curve_fit(fit_function, x2_L, y2_L)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1_k, b1_k = popt1_k
m2_k, b2_k = popt2_k
m1_g, b1_g = popt1_g
m2_g, b2_g = popt2_g
m1_m, b1_m = popt1_m
m2_m, b2_m = popt2_m
m1_L, b1_L = popt1_L
m2_L, b2_L = popt2_L

# 오차 계산
m1_err_k = np.sqrt(c1_k[0, 0])
b1_err_k = np.sqrt(c1_k[1, 1])
m2_err_k = np.sqrt(c2_k[0, 0])
b2_err_k = np.sqrt(c2_k[1, 1])
m1_err_g = np.sqrt(c1_g[0, 0])
b1_err_g = np.sqrt(c1_g[1, 1])
m2_err_g = np.sqrt(c2_g[0, 0])
b2_err_g = np.sqrt(c2_g[1, 1])
m1_err_m = np.sqrt(c1_m[0, 0])
b1_err_m = np.sqrt(c1_m[1, 1])
m2_err_m = np.sqrt(c2_m[0, 0])
b2_err_m = np.sqrt(c2_m[1, 1])
m1_err_L = np.sqrt(c1_L[0, 0])
b1_err_L = np.sqrt(c1_L[1, 1])
m2_err_L = np.sqrt(c2_L[0, 0])
b2_err_L = np.sqrt(c2_L[1, 1])

# 결과 출력
print(f"y1_k = ({m1_k:.10f} ± {m1_err_k:.10f}) * x + ({b1_k:.10f} ± {b1_err_k:.10f})")
print(f"y2_k = ({m2_k:.10f} ± {m2_err_k:.10f}) * x + ({b2_k:.10f} ± {b2_err_k:.10f})")
print(f"y1_g = ({m1_g:.10f} ± {m1_err_g:.10f}) * x + ({b1_g:.10f} ± {b1_err_g:.10f})")
print(f"y2_g = ({m2_g:.10f} ± {m2_err_g:.10f}) * x + ({b2_g:.10f} ± {b2_err_g:.10f})")
print(f"y1_m = ({m1_m:.10f} ± {m1_err_m:.10f}) * x + ({b1_m:.10f} ± {b1_err_m:.10f})")
print(f"y2_m = ({m2_m:.10f} ± {m2_err_m:.10f}) * x + ({b2_m:.10f} ± {b2_err_m:.10f})")
print(f"y1_L = ({m1_L:.10f} ± {m1_err_L:.10f}) * x + ({b1_L:.10f} ± {b1_err_L:.10f})")
print(f"y2_L = ({m2_L:.10f} ± {m2_err_L:.10f}) * x + ({b2_L:.10f} ± {b2_err_L:.10f})")

val_m1_k = Value(m1_k, m1_err_k)
val_b1_k = Value(b1_k, b1_err_k)
val_m2_k = Value(m2_k, m2_err_k)
val_b2_k = Value(b2_k, b2_err_k)
val_m1_g = Value(m1_g, m1_err_g)
val_b1_g = Value(b1_g, b1_err_g)
val_m2_g = Value(m2_g, m2_err_g)
val_b2_g = Value(b2_g, b2_err_g)
val_m1_m = Value(m1_m, m1_err_m)
val_b1_m = Value(b1_m, b1_err_m)
val_m2_m = Value(m2_m, m2_err_m)
val_b2_m = Value(b2_m, b2_err_m)
val_m1_L = Value(m1_L, m1_err_L)
val_b1_L = Value(b1_L, b1_err_L)
val_m2_L = Value(m2_L, m2_err_L)
val_b2_L = Value(b2_L, b2_err_L)

# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection_k = (val_b2_k - val_b1_k) / (val_m1_k - val_m2_k)
y_intersection_k = fit_function(x_intersection_k, m1_k, b1_k)
x_intersection_g = (val_b2_g - val_b1_g) / (val_m1_g - val_m2_g)
y_intersection_g = fit_function(x_intersection_g, m1_g, b1_g)
x_intersection_m = (val_b2_m - val_b1_m) / (val_m1_m - val_m2_m)
y_intersection_m = fit_function(x_intersection_m, m1_m, b1_m)
x_intersection_L = (val_b2_L - val_b1_L) / (val_m1_L - val_m2_L)
y_intersection_L = fit_function(x_intersection_L, m1_L, b1_L)

# 교차점 출력
print( f'Us_k = {x_intersection_k}')
print( f'Us_g = {x_intersection_g}')
print( f'Us_m = {x_intersection_m}')
print( f'Us_L = {x_intersection_L}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit_k = np.linspace(0, 3.7, 100)
y1_fit_k = fit_function1(x1_fit_k, m1_k, b1_k)
x1_fit_g = np.linspace(0, 3.3, 100)
y1_fit_g = fit_function1(x1_fit_g, m1_g, b1_g)
x1_fit_m = np.linspace(0, 3.3, 100)
y1_fit_m = fit_function1(x1_fit_m, m1_m, b1_m)
x1_fit_L = np.linspace(0, 3.2, 100)
y1_fit_L = fit_function1(x1_fit_L, m1_L, b1_L)

# 두 번째 fit 함수 연장 시켜 그리기
x2_fit_k = np.linspace(3.72, 3.9, 100)
y2_fit_k = fit_function(x2_fit_k, m2_k, b2_k)
x2_fit_g = np.linspace(3.22, 4.0, 100)
y2_fit_g = fit_function(x2_fit_g, m2_g, b2_g)
x2_fit_m = np.linspace(3.35, 3.8, 100)
y2_fit_m = fit_function(x2_fit_m, m2_m, b2_m)
x2_fit_L = np.linspace(3.2, 3.6, 100)
y2_fit_L = fit_function(x2_fit_L, m2_L, b2_L)


# 그래프 생성
plt.figure(figsize=(8, 6))

plt.scatter(voltage_gross, current_gross, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x1_fit_g, y1_fit_g, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_g, fit_function(x2_g, m2_g, b2_g), 'g-', label='Fitlinie 2 gross')  # 두 번째 피팅된 직선
plt.plot(x2_fit_g, y2_fit_g, 'g-')  # 두

plt.scatter(voltage_mittel, current_mittel, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x1_fit_m, y1_fit_m, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_m, fit_function(x2_m, m2_m, b2_m), 'g-', label='Fitlinie 2 mittel')  # 두 번째 피팅된 직선
plt.plot(x2_fit_m, y2_fit_m, 'g-')  # 두

plt.scatter(voltage_klein, current_klein, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x1_fit_k, y1_fit_k, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_k, fit_function(x2_k, m2_k, b2_k), 'g-', label='Fitlinie 2 klein')  # 두 번째 피팅된 직선
plt.plot(x2_fit_k, y2_fit_k, 'g-')  # 두

plt.scatter(voltage_L, current_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x1_fit_L, y1_fit_L, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
# plt.plot(x2_L, fit_function(x2_L, m2_L, b2_L), 'g-', label='Fitlinie 2 L')  # 두 번째 피팅된 직선
plt.plot(x2_fit_L, y2_fit_L, 'g-', label='Fitlinie 2')  # 두

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.001, 0.0150)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B3297.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# B3297-log-IV

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_klein.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_gross.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_mittel.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/4.Aufgabe/IV_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# voltage 그리고 log_current 필터링
voltage_filtered_k = data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['V.1']
log_current_filtered_k = np.log(data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['A.1'])
voltage_filtered_g = data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['V.1']
log_current_filtered_g = np.log(data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['A.1'])
voltage_filtered_m = data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['V.1']
log_current_filtered_m = np.log(data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['A.1'])
voltage_filtered_L = data_L[(voltage_L >= 0.6) & (current_L > 0)]['V.1']
log_current_filtered_L = np.log(data_L[(voltage_L >= 0.6) & (current_L > 0)]['A.1'])


# fit 함수를 위한 데이터 분리
x_k = data_klein[(voltage_klein < 3.7) & (voltage_klein > 2) & (current_klein > 0)]['V.1']
y_k = np.log(data_klein[(voltage_klein < 3.7) & (voltage_klein > 2) & (current_klein > 0)]['A.1'])
x_g = data_gross[(voltage_gross < 3.2) & (voltage_gross > 2) & (current_gross > 0)]['V.1']
y_g = np.log(data_gross[(voltage_gross < 3.2) & (voltage_gross > 2) & (current_gross > 0)]['A.1'])
x_m = data_mittel[(voltage_mittel < 3.5) & (voltage_mittel > 2.2) & (current_mittel > 0)]['V.1']
y_m = np.log(data_mittel[(voltage_mittel < 3.5) & (voltage_mittel > 2.2) & (current_mittel > 0)]['A.1'])
x_L = data_L[(voltage_L < 3.2) & (voltage_L > 2) & (current_L > 0)]['V.1']
y_L = np.log(data_L[(voltage_L < 3.2) & (voltage_L > 2) & (current_L > 0)]['A.1'])

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt_k, pcov_k = curve_fit(fit_function, x_k, y_k)  
popt_g, pcov_g = curve_fit(fit_function, x_g, y_g)  
popt_m, pcov_m = curve_fit(fit_function, x_m, y_m)  
popt_L, pcov_L = curve_fit(fit_function, x_L, y_L)  


# 피팅된 기울기와 절편
m_k, b_k = popt_k
m_g, b_g = popt_g
m_m, b_m = popt_m
m_L, b_L = popt_L

# 오차 계산
m_k_err = np.sqrt(pcov_k[0, 0])
b_k_err = np.sqrt(pcov_k[1, 1])
m_g_err = np.sqrt(pcov_g[0, 0])
b_g_err = np.sqrt(pcov_g[1, 1])
m_m_err = np.sqrt(pcov_m[0, 0])
b_m_err = np.sqrt(pcov_m[1, 1])
m_L_err = np.sqrt(pcov_L[0, 0])
b_L_err = np.sqrt(pcov_L[1, 1])

# 결과 출력
print(f"y_k = ({m_k:.10f} ± {m_k_err:.10f}) * x + ({b_k:.10f} ± {b_k_err:.10f})")
print(f"y_g = ({m_g:.10f} ± {m_g_err:.10f}) * x + ({b_g:.10f} ± {b_g_err:.10f})")
print(f"y_m = ({m_m:.10f} ± {m_m_err:.10f}) * x + ({b_m:.10f} ± {b_m_err:.10f})")
print(f"y_L = ({m_L:.10f} ± {m_L_err:.10f}) * x + ({b_L:.10f} ± {b_L_err:.10f})")

val_m_k = Value(m_k, m_k_err)
val_b_k = Value(b_k, b_k_err)
val_m_g = Value(m_g, m_g_err)
val_b_g = Value(b_g, b_g_err)
val_m_m = Value(m_m, m_m_err)
val_b_m = Value(b_m, b_m_err)
val_m_L = Value(m_L, m_L_err)
val_b_L = Value(b_L, b_L_err)


# Sätttigungsstrom & Idealitätsfaktor 계산
# Sättigungsstrom Is = np.exp(val_b)
Is_k = np.exp(b_k)
Is_g = np.exp(b_g)
Is_m = np.exp(b_m)
Is_L = np.exp(b_L)
# Sättigungsstrom 오차 계산
Is_k_err = np.abs(Is_k * b_k_err)
Is_g_err = np.abs(Is_g * b_g_err)
Is_m_err = np.abs(Is_m * b_m_err)
Is_L_err = np.abs(Is_L * b_L_err)
# Idealitätsfaktor n = e / (k * T * val_m)
e = 1.6 * 1e-19
k = 1.38 * 1e-23
T = 298
n_k = e / (k * T * m_k)
n_g = e / (k * T * m_g)
n_m = e / (k * T * m_m)
n_L = e / (k * T * m_L)
# Idealitätsfaktor 오차 계산
n_k_err = np.abs(n_k * m_k_err)
n_g_err = np.abs(n_g * m_g_err)
n_m_err = np.abs(n_m * m_m_err)
n_L_err = np.abs(n_L * m_L_err)

# 결과 출력
print(f"Is_k = {Is_k:.10e} ± {Is_k_err:.10e}")
print(f"Is_g = {Is_g:.10e} ± {Is_g_err:.10e}")
print(f"Is_m = {Is_m:.10e} ± {Is_m_err:.10e}")
print(f"Is_L = {Is_L:.10e} ± {Is_L_err:.10e}")
print(f"n_k = {n_k:.10f} ± {n_k_err:.10f}")
print(f"n_g = {n_g:.10f} ± {n_g_err:.10f}")
print(f"n_m = {n_m:.10f} ± {n_m_err:.10f}")
print(f"n_L = {n_L:.10f} ± {n_L_err:.10f}")



# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered_g, log_current_filtered_g, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x_g, fit_function(x_g, m_g, b_g), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_m, log_current_filtered_m, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x_m, fit_function(x_m, m_m, b_m), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_k, log_current_filtered_k, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x_k, fit_function(x_k, m_k, b_k), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_L, log_current_filtered_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x_L, fit_function(x_L, m_L, b_L), 'b-', label='Fitlinie')  # 피팅된 직선


# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B3297_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()



# %% [markdown]
# B2396 - IV

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_c.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_a.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_b.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')


# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1_k = voltage_klein[voltage_klein <= 2.7]
y1_k = current_klein[voltage_klein <= 2.7]
x1_g = voltage_gross[voltage_gross <= 2.7]
y1_g = current_gross[voltage_gross <= 2.7]
x1_m = voltage_mittel[voltage_mittel <= 2.7]
y1_m = current_mittel[voltage_mittel <= 2.7]
x1_L = voltage_L[voltage_L <= 2.7]
y1_L = current_L[voltage_L <= 2.7]

# 두번 째 fit 함수를 위해 데이터 분리
x2_k = voltage_klein[voltage_klein >= 3.1]
y2_k = current_klein[voltage_klein >= 3.1]
x2_g = voltage_gross[voltage_gross >= 2.6]
y2_g = current_gross[voltage_gross >= 2.6]
x2_m = voltage_mittel[voltage_mittel >= 2.9]
y2_m = current_mittel[voltage_mittel >= 2.9]
x2_L = voltage_L[voltage_L >= 2.8]
y2_L = current_L[voltage_L >= 2.8]


# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1_k, c1_k = curve_fit(fit_function, x1_k, y1_k)  # 첫 번째 직선 피팅
popt2_k, c2_k = curve_fit(fit_function, x2_k, y2_k)  # 두 번째 직선 피팅
popt1_g, c1_g = curve_fit(fit_function, x1_g, y1_g)  # 첫 번째 직선 피팅
popt2_g, c2_g = curve_fit(fit_function, x2_g, y2_g)  # 두 번째 직선 피팅
popt1_m, c1_m = curve_fit(fit_function, x1_m, y1_m)  # 첫 번째 직선 피팅
popt2_m, c2_m = curve_fit(fit_function, x2_m, y2_m)  # 두 번째 직선 피팅
popt1_L, c1_L = curve_fit(fit_function, x1_L, y1_L)  # 첫 번째 직선 피팅
popt2_L, c2_L = curve_fit(fit_function, x2_L, y2_L)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1_k, b1_k = popt1_k
m2_k, b2_k = popt2_k
m1_g, b1_g = popt1_g
m2_g, b2_g = popt2_g
m1_m, b1_m = popt1_m
m2_m, b2_m = popt2_m
m1_L, b1_L = popt1_L
m2_L, b2_L = popt2_L

# 오차 계산
m1_err_k = np.sqrt(c1_k[0, 0])
b1_err_k = np.sqrt(c1_k[1, 1])
m2_err_k = np.sqrt(c2_k[0, 0])
b2_err_k = np.sqrt(c2_k[1, 1])
m1_err_g = np.sqrt(c1_g[0, 0])
b1_err_g = np.sqrt(c1_g[1, 1])
m2_err_g = np.sqrt(c2_g[0, 0])
b2_err_g = np.sqrt(c2_g[1, 1])
m1_err_m = np.sqrt(c1_m[0, 0])
b1_err_m = np.sqrt(c1_m[1, 1])
m2_err_m = np.sqrt(c2_m[0, 0])
b2_err_m = np.sqrt(c2_m[1, 1])
m1_err_L = np.sqrt(c1_L[0, 0])
b1_err_L = np.sqrt(c1_L[1, 1])
m2_err_L = np.sqrt(c2_L[0, 0])
b2_err_L = np.sqrt(c2_L[1, 1])

# 결과 출력

print(f"y1_k = ({m1_k:.10f} ± {m1_err_k:.10f}) * x + ({b1_k:.10f} ± {b1_err_k:.10f})")
print(f"y2_k = ({m2_k:.10f} ± {m2_err_k:.10f}) * x + ({b2_k:.10f} ± {b2_err_k:.10f})")
print(f"y1_g = ({m1_g:.10f} ± {m1_err_g:.10f}) * x + ({b1_g:.10f} ± {b1_err_g:.10f})")
print(f"y2_g = ({m2_g:.10f} ± {m2_err_g:.10f}) * x + ({b2_g:.10f} ± {b2_err_g:.10f})")
print(f"y1_m = ({m1_m:.10f} ± {m1_err_m:.10f}) * x + ({b1_m:.10f} ± {b1_err_m:.10f})")
print(f"y2_m = ({m2_m:.10f} ± {m2_err_m:.10f}) * x + ({b2_m:.10f} ± {b2_err_m:.10f})")
print(f"y1_L = ({m1_L:.10f} ± {m1_err_L:.10f}) * x + ({b1_L:.10f} ± {b1_err_L:.10f})")
print(f"y2_L = ({m2_L:.10f} ± {m2_err_L:.10f}) * x + ({b2_L:.10f} ± {b2_err_L:.10f})")

val_m1_k = Value(m1_k, m1_err_k)
val_b1_k = Value(b1_k, b1_err_k)
val_m2_k = Value(m2_k, m2_err_k)
val_b2_k = Value(b2_k, b2_err_k)
val_m1_g = Value(m1_g, m1_err_g)
val_b1_g = Value(b1_g, b1_err_g)
val_m2_g = Value(m2_g, m2_err_g)
val_b2_g = Value(b2_g, b2_err_g)
val_m1_m = Value(m1_m, m1_err_m)
val_b1_m = Value(b1_m, b1_err_m)
val_m2_m = Value(m2_m, m2_err_m)
val_b2_m = Value(b2_m, b2_err_m)
val_m1_L = Value(m1_L, m1_err_L)
val_b1_L = Value(b1_L, b1_err_L)
val_m2_L = Value(m2_L, m2_err_L)
val_b2_L = Value(b2_L, b2_err_L)

# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection_k = (val_b2_k - val_b1_k) / (val_m1_k - val_m2_k)
y_intersection_k = fit_function(x_intersection_k, m1_k, b1_k)
x_intersection_g = (val_b2_g - val_b1_g) / (val_m1_g - val_m2_g)
y_intersection_g = fit_function(x_intersection_g, m1_g, b1_g)
x_intersection_m = (val_b2_m - val_b1_m) / (val_m1_m - val_m2_m)
y_intersection_m = fit_function(x_intersection_m, m1_m, b1_m)
x_intersection_L = (val_b2_L - val_b1_L) / (val_m1_L - val_m2_L)
y_intersection_L = fit_function(x_intersection_L, m1_L, b1_L)

# 교차점 출력
print( f'Us_k = {x_intersection_k}')
print( f'Us_g = {x_intersection_g}')
print( f'Us_m = {x_intersection_m}')
print( f'Us_L = {x_intersection_L}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit_k = np.linspace(0, 3.3, 100)
y1_fit_k = fit_function1(x1_fit_k, m1_k, b1_k)
x1_fit_g = np.linspace(0, 3, 100)
y1_fit_g = fit_function1(x1_fit_g, m1_g, b1_g)
x1_fit_m = np.linspace(0, 3.2, 100)
y1_fit_m = fit_function1(x1_fit_m, m1_m, b1_m)
x1_fit_L = np.linspace(0, 3.1, 100)
y1_fit_L = fit_function1(x1_fit_L, m1_L, b1_L)

# 두 번째 fit 함수 연장 시켜 그리기
x2_fit_k = np.linspace(3.1, 3.5, 100)
y2_fit_k = fit_function(x2_fit_k, m2_k, b2_k)
x2_fit_g = np.linspace(2.9, 3.3, 100)
y2_fit_g = fit_function(x2_fit_g, m2_g, b2_g)
x2_fit_m = np.linspace(2.98, 3.2, 100)
y2_fit_m = fit_function(x2_fit_m, m2_m, b2_m)
x2_fit_L = np.linspace(2.9, 3.4, 100)
y2_fit_L = fit_function(x2_fit_L, m2_L, b2_L)


# 그래프 생성
plt.figure(figsize=(8, 6))

plt.scatter(voltage_gross, current_gross, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x1_fit_g, y1_fit_g, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_g, fit_function(x2_g, m2_g, b2_g), 'g-', label='Fitlinie 2 gross')  # 두 번째 피팅된 직선
plt.plot(x2_fit_g, y2_fit_g, 'g-')  # 두

plt.scatter(voltage_mittel, current_mittel, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x1_fit_m, y1_fit_m, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_m, fit_function(x2_m, m2_m, b2_m), 'g-', label='Fitlinie 2 mittel')  # 두 번째 피팅된 직선
plt.plot(x2_fit_m, y2_fit_m, 'g-')  # 두

plt.scatter(voltage_klein, current_klein, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x1_fit_k, y1_fit_k, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_k, fit_function(x2_k, m2_k, b2_k), 'g-', label='Fitlinie 2 klein')  # 두 번째 피팅된 직선
plt.plot(x2_fit_k, y2_fit_k, 'g-')  # 두

plt.scatter(voltage_L, current_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x1_fit_L, y1_fit_L, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
# plt.plot(x2_L, fit_function(x2_L, m2_L, b2_L), 'g-', label='Fitlinie 2 L')  # 두 번째 피팅된 직선
plt.plot(x2_fit_L, y2_fit_L, 'g-', label='Fitlinie 2')  # 두

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.001, 0.020)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B2396.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# B2396 - IV-log

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_c.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_a.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_b.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B2396_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# voltage 그리고 log_current 필터링
voltage_filtered_k = data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['V.1']
log_current_filtered_k = np.log(data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['A.1'])
voltage_filtered_g = data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['V.1']
log_current_filtered_g = np.log(data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['A.1'])
voltage_filtered_m = data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['V.1']
log_current_filtered_m = np.log(data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['A.1'])
voltage_filtered_L = data_L[(voltage_L >= 0.6) & (current_L > 0)]['V.1']
log_current_filtered_L = np.log(data_L[(voltage_L >= 0.6) & (current_L > 0)]['A.1'])


# fit 함수를 위한 데이터 분리
x_k = data_klein[(voltage_klein < 3.2) & (voltage_klein > 2.4) & (current_klein > 0)]['V.1']
y_k = np.log(data_klein[(voltage_klein < 3.2) & (voltage_klein > 2.4) & (current_klein > 0)]['A.1'])
x_g = data_gross[(voltage_gross < 3.1) & (voltage_gross > 2.6) & (current_gross > 0)]['V.1']
y_g = np.log(data_gross[(voltage_gross < 3.1) & (voltage_gross > 2.6) & (current_gross > 0)]['A.1'])
x_m = data_mittel[(voltage_mittel < 3.2) & (voltage_mittel > 2.5) & (current_mittel > 0)]['V.1']
y_m = np.log(data_mittel[(voltage_mittel < 3.2) & (voltage_mittel > 2.5) & (current_mittel > 0)]['A.1'])
x_L = data_L[(voltage_L < 3) & (voltage_L > 2.3) & (current_L > 0)]['V.1']
y_L = np.log(data_L[(voltage_L < 3) & (voltage_L > 2.3) & (current_L > 0)]['A.1'])

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt_k, pcov_k = curve_fit(fit_function, x_k, y_k)  
popt_g, pcov_g = curve_fit(fit_function, x_g, y_g)  
popt_m, pcov_m = curve_fit(fit_function, x_m, y_m)  
popt_L, pcov_L = curve_fit(fit_function, x_L, y_L)  


# 피팅된 기울기와 절편
m_k, b_k = popt_k
m_g, b_g = popt_g
m_m, b_m = popt_m
m_L, b_L = popt_L

# 오차 계산
m_k_err = np.sqrt(pcov_k[0, 0])
b_k_err = np.sqrt(pcov_k[1, 1])
m_g_err = np.sqrt(pcov_g[0, 0])
b_g_err = np.sqrt(pcov_g[1, 1])
m_m_err = np.sqrt(pcov_m[0, 0])
b_m_err = np.sqrt(pcov_m[1, 1])
m_L_err = np.sqrt(pcov_L[0, 0])
b_L_err = np.sqrt(pcov_L[1, 1])

# 결과 출력
print(f"y_k = ({m_k:.10f} ± {m_k_err:.10f}) * x + ({b_k:.10f} ± {b_k_err:.10f})")
print(f"y_g = ({m_g:.10f} ± {m_g_err:.10f}) * x + ({b_g:.10f} ± {b_g_err:.10f})")
print(f"y_m = ({m_m:.10f} ± {m_m_err:.10f}) * x + ({b_m:.10f} ± {b_m_err:.10f})")
print(f"y_L = ({m_L:.10f} ± {m_L_err:.10f}) * x + ({b_L:.10f} ± {b_L_err:.10f})")

val_m_k = Value(m_k, m_k_err)
val_b_k = Value(b_k, b_k_err)
val_m_g = Value(m_g, m_g_err)
val_b_g = Value(b_g, b_g_err)
val_m_m = Value(m_m, m_m_err)
val_b_m = Value(b_m, b_m_err)
val_m_L = Value(m_L, m_L_err)
val_b_L = Value(b_L, b_L_err)


# Sätttigungsstrom & Idealitätsfaktor 계산
# Sättigungsstrom Is = np.exp(val_b)
Is_k = np.exp(b_k)
Is_g = np.exp(b_g)
Is_m = np.exp(b_m)
Is_L = np.exp(b_L)
# Sättigungsstrom 오차 계산
Is_k_err = np.abs(Is_k * b_k_err)
Is_g_err = np.abs(Is_g * b_g_err)
Is_m_err = np.abs(Is_m * b_m_err)
Is_L_err = np.abs(Is_L * b_L_err)
# Idealitätsfaktor n = e / (k * T * val_m)
e = 1.6 * 1e-19
k = 1.38 * 1e-23
T = 298
n_k = e / (k * T * m_k)
n_g = e / (k * T * m_g)
n_m = e / (k * T * m_m)
n_L = e / (k * T * m_L)
# Idealitätsfaktor 오차 계산
n_k_err = np.abs(n_k * m_k_err)
n_g_err = np.abs(n_g * m_g_err)
n_m_err = np.abs(n_m * m_m_err)
n_L_err = np.abs(n_L * m_L_err)

# 결과 출력
print(f"Is_k = {Is_k:.10e} ± {Is_k_err:.10e}")
print(f"Is_g = {Is_g:.10e} ± {Is_g_err:.10e}")
print(f"Is_m = {Is_m:.10e} ± {Is_m_err:.10e}")
print(f"Is_L = {Is_L:.10e} ± {Is_L_err:.10e}")
print(f"n_k = {n_k:.10f} ± {n_k_err:.10f}")
print(f"n_g = {n_g:.10f} ± {n_g_err:.10f}")
print(f"n_m = {n_m:.10f} ± {n_m_err:.10f}")
print(f"n_L = {n_L:.10f} ± {n_L_err:.10f}")

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered_g, log_current_filtered_g, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x_g, fit_function(x_g, m_g, b_g), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_m, log_current_filtered_m, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x_m, fit_function(x_m, m_m, b_m), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_k, log_current_filtered_k, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x_k, fit_function(x_k, m_k, b_k), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_L, log_current_filtered_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x_L, fit_function(x_L, m_L, b_L), 'b-', label='Fitlinie')  # 피팅된 직선


# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B2396_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()



# %% [markdown]
# B3204A - IV

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_c.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_a_V2.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_b.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')


# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# 첫번 째 fit 함수를 위해 데이터 분리
x1_k = voltage_klein[voltage_klein <= 3.3]
y1_k = current_klein[voltage_klein <= 3.3]
x1_g = voltage_gross[voltage_gross <= 3]
y1_g = current_gross[voltage_gross <= 3]
x1_m = voltage_mittel[voltage_mittel <= 2.9]
y1_m = current_mittel[voltage_mittel <= 2.9]
x1_L = voltage_L[voltage_L <= 2.7]
y1_L = current_L[voltage_L <= 2.7]

# 두번 째 fit 함수를 위해 데이터 분리
x2_k = voltage_klein[voltage_klein >= 3.2]
y2_k = current_klein[voltage_klein >= 3.2]
x2_g = voltage_gross[voltage_gross >= 2.9]
y2_g = current_gross[voltage_gross >= 2.9]
x2_m = voltage_mittel[voltage_mittel >= 3]
y2_m = current_mittel[voltage_mittel >= 3]
x2_L = voltage_L[voltage_L >= 2.8]
y2_L = current_L[voltage_L >= 2.8]


# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt1_k, c1_k = curve_fit(fit_function, x1_k, y1_k)  # 첫 번째 직선 피팅
popt2_k, c2_k = curve_fit(fit_function, x2_k, y2_k)  # 두 번째 직선 피팅
popt1_g, c1_g = curve_fit(fit_function, x1_g, y1_g)  # 첫 번째 직선 피팅
popt2_g, c2_g = curve_fit(fit_function, x2_g, y2_g)  # 두 번째 직선 피팅
popt1_m, c1_m = curve_fit(fit_function, x1_m, y1_m)  # 첫 번째 직선 피팅
popt2_m, c2_m = curve_fit(fit_function, x2_m, y2_m)  # 두 번째 직선 피팅
popt1_L, c1_L = curve_fit(fit_function, x1_L, y1_L)  # 첫 번째 직선 피팅
popt2_L, c2_L = curve_fit(fit_function, x2_L, y2_L)  # 두 번째 직선 피팅

# 피팅된 기울기와 절편
m1_k, b1_k = popt1_k
m2_k, b2_k = popt2_k
m1_g, b1_g = popt1_g
m2_g, b2_g = popt2_g
m1_m, b1_m = popt1_m
m2_m, b2_m = popt2_m
m1_L, b1_L = popt1_L
m2_L, b2_L = popt2_L

# 오차 계산
m1_err_k = np.sqrt(c1_k[0, 0])
b1_err_k = np.sqrt(c1_k[1, 1])
m2_err_k = np.sqrt(c2_k[0, 0])
b2_err_k = np.sqrt(c2_k[1, 1])
m1_err_g = np.sqrt(c1_g[0, 0])
b1_err_g = np.sqrt(c1_g[1, 1])
m2_err_g = np.sqrt(c2_g[0, 0])
b2_err_g = np.sqrt(c2_g[1, 1])
m1_err_m = np.sqrt(c1_m[0, 0])
b1_err_m = np.sqrt(c1_m[1, 1])
m2_err_m = np.sqrt(c2_m[0, 0])
b2_err_m = np.sqrt(c2_m[1, 1])
m1_err_L = np.sqrt(c1_L[0, 0])
b1_err_L = np.sqrt(c1_L[1, 1])
m2_err_L = np.sqrt(c2_L[0, 0])
b2_err_L = np.sqrt(c2_L[1, 1])

# 결과 출력

print(f"y1_k = ({m1_k:.10f} ± {m1_err_k:.10f}) * x + ({b1_k:.10f} ± {b1_err_k:.10f})")
print(f"y2_k = ({m2_k:.10f} ± {m2_err_k:.10f}) * x + ({b2_k:.10f} ± {b2_err_k:.10f})")
print(f"y1_g = ({m1_g:.10f} ± {m1_err_g:.10f}) * x + ({b1_g:.10f} ± {b1_err_g:.10f})")
print(f"y2_g = ({m2_g:.10f} ± {m2_err_g:.10f}) * x + ({b2_g:.10f} ± {b2_err_g:.10f})")
print(f"y1_m = ({m1_m:.10f} ± {m1_err_m:.10f}) * x + ({b1_m:.10f} ± {b1_err_m:.10f})")
print(f"y2_m = ({m2_m:.10f} ± {m2_err_m:.10f}) * x + ({b2_m:.10f} ± {b2_err_m:.10f})")
print(f"y1_L = ({m1_L:.10f} ± {m1_err_L:.10f}) * x + ({b1_L:.10f} ± {b1_err_L:.10f})")
print(f"y2_L = ({m2_L:.10f} ± {m2_err_L:.10f}) * x + ({b2_L:.10f} ± {b2_err_L:.10f})")

val_m1_k = Value(m1_k, m1_err_k)
val_b1_k = Value(b1_k, b1_err_k)
val_m2_k = Value(m2_k, m2_err_k)
val_b2_k = Value(b2_k, b2_err_k)
val_m1_g = Value(m1_g, m1_err_g)
val_b1_g = Value(b1_g, b1_err_g)
val_m2_g = Value(m2_g, m2_err_g)
val_b2_g = Value(b2_g, b2_err_g)
val_m1_m = Value(m1_m, m1_err_m)
val_b1_m = Value(b1_m, b1_err_m)
val_m2_m = Value(m2_m, m2_err_m)
val_b2_m = Value(b2_m, b2_err_m)
val_m1_L = Value(m1_L, m1_err_L)
val_b1_L = Value(b1_L, b1_err_L)
val_m2_L = Value(m2_L, m2_err_L)
val_b2_L = Value(b2_L, b2_err_L)

# 교차점 계산
# val_m1 * x + val_b1 = val_m2 * x + val_b2
# (val_m1 - val_m2) * x = val_b2 - val_b1
# x = (val_b2 - val_b1) / (val_m1 - val_m2)
x_intersection_k = (val_b2_k - val_b1_k) / (val_m1_k - val_m2_k)
y_intersection_k = fit_function(x_intersection_k, m1_k, b1_k)
x_intersection_g = (val_b2_g - val_b1_g) / (val_m1_g - val_m2_g)
y_intersection_g = fit_function(x_intersection_g, m1_g, b1_g)
x_intersection_m = (val_b2_m - val_b1_m) / (val_m1_m - val_m2_m)
y_intersection_m = fit_function(x_intersection_m, m1_m, b1_m)
x_intersection_L = (val_b2_L - val_b1_L) / (val_m1_L - val_m2_L)
y_intersection_L = fit_function(x_intersection_L, m1_L, b1_L)

# 교차점 출력
print( f'Us_k = {x_intersection_k}')
print( f'Us_g = {x_intersection_g}')
print( f'Us_m = {x_intersection_m}')
print( f'Us_L = {x_intersection_L}')

# 첫 번째 fit 함수 연장 시켜 그리기
def fit_function1(x, a, b):
    y = a * x + b
    return y

x1_fit_k = np.linspace(0, 3.3, 100)
y1_fit_k = fit_function1(x1_fit_k, m1_k, b1_k)
x1_fit_g = np.linspace(0, 3, 100)
y1_fit_g = fit_function1(x1_fit_g, m1_g, b1_g)
x1_fit_m = np.linspace(0, 3.2, 100)
y1_fit_m = fit_function1(x1_fit_m, m1_m, b1_m)
x1_fit_L = np.linspace(0, 3.1, 100)
y1_fit_L = fit_function1(x1_fit_L, m1_L, b1_L)

# 두 번째 fit 함수 연장 시켜 그리기
x2_fit_k = np.linspace(3.2, 3.8, 100)
y2_fit_k = fit_function(x2_fit_k, m2_k, b2_k)
x2_fit_g = np.linspace(3, 3.5, 100)
y2_fit_g = fit_function(x2_fit_g, m2_g, b2_g)
x2_fit_m = np.linspace(3.1, 3.7, 100)
y2_fit_m = fit_function(x2_fit_m, m2_m, b2_m)
x2_fit_L = np.linspace(2.9, 3.4, 100)
y2_fit_L = fit_function(x2_fit_L, m2_L, b2_L)


# 그래프 생성
plt.figure(figsize=(8, 6))

plt.scatter(voltage_gross, current_gross, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x1_fit_g, y1_fit_g, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_g, fit_function(x2_g, m2_g, b2_g), 'g-', label='Fitlinie 2 gross')  # 두 번째 피팅된 직선
plt.plot(x2_fit_g, y2_fit_g, 'g-')  # 두

plt.scatter(voltage_mittel, current_mittel, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x1_fit_m, y1_fit_m, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_m, fit_function(x2_m, m2_m, b2_m), 'g-', label='Fitlinie 2 mittel')  # 두 번째 피팅된 직선
plt.plot(x2_fit_m, y2_fit_m, 'g-')  # 두

plt.scatter(voltage_klein, current_klein, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x1_fit_k, y1_fit_k, 'b-')  # 첫 번째 피팅된 직선
# plt.plot(x2_k, fit_function(x2_k, m2_k, b2_k), 'g-', label='Fitlinie 2 klein')  # 두 번째 피팅된 직선
plt.plot(x2_fit_k, y2_fit_k, 'g-')  # 두

plt.scatter(voltage_L, current_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x1_fit_L, y1_fit_L, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
# plt.plot(x2_L, fit_function(x2_L, m2_L, b2_L), 'g-', label='Fitlinie 2 L')  # 두 번째 피팅된 직선
plt.plot(x2_fit_L, y2_fit_L, 'g-', label='Fitlinie 2')  # 두

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

plt.ylim(-0.001, 0.020)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B3204A.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# B3204A - IV-log

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

file_klein = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_c.csv'
file_gross = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_a_V2.csv'
file_mittel = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_b.csv'
file_L = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/I-V_B3204A_L.csv'

data_klein = pd.read_csv(file_klein, sep='\t', skiprows = 1 ,engine='python')
data_gross = pd.read_csv(file_gross, sep='\t', skiprows = 1 ,engine='python')
data_mittel = pd.read_csv(file_mittel, sep='\t', skiprows = 1 ,engine='python')
data_L = pd.read_csv(file_L, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# 데이터 추출
voltage_klein = data_klein['V.1']
current_klein = data_klein['A.1']
voltage_gross = data_gross['V.1']
current_gross = data_gross['A.1']
voltage_mittel = data_mittel['V.1']
current_mittel = data_mittel['A.1']
voltage_L = data_L['V.1']
current_L = data_L['A.1']

# voltage 그리고 log_current 필터링
voltage_filtered_k = data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['V.1']
log_current_filtered_k = np.log(data_klein[(voltage_klein >= 0.6) & (current_klein > 0)]['A.1'])
voltage_filtered_g = data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['V.1']
log_current_filtered_g = np.log(data_gross[(voltage_gross >= 0.6) & (current_gross > 0)]['A.1'])
voltage_filtered_m = data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['V.1']
log_current_filtered_m = np.log(data_mittel[(voltage_mittel >= 0.6) & (current_mittel > 0)]['A.1'])
voltage_filtered_L = data_L[(voltage_L >= 0.6) & (current_L > 0)]['V.1']
log_current_filtered_L = np.log(data_L[(voltage_L >= 0.6) & (current_L > 0)]['A.1'])


# fit 함수를 위한 데이터 분리
x_k = data_klein[(voltage_klein < 3.2) & (voltage_klein > 2.4) & (current_klein > 0)]['V.1']
y_k = np.log(data_klein[(voltage_klein < 3.2) & (voltage_klein > 2.4) & (current_klein > 0)]['A.1'])
x_g = data_gross[(voltage_gross < 3.1) & (voltage_gross > 2.6) & (current_gross > 0)]['V.1']
y_g = np.log(data_gross[(voltage_gross < 3.1) & (voltage_gross > 2.6) & (current_gross > 0)]['A.1'])
x_m = data_mittel[(voltage_mittel < 3.2) & (voltage_mittel > 2.5) & (current_mittel > 0)]['V.1']
y_m = np.log(data_mittel[(voltage_mittel < 3.2) & (voltage_mittel > 2.5) & (current_mittel > 0)]['A.1'])
x_L = data_L[(voltage_L < 3) & (voltage_L > 2.3) & (current_L > 0)]['V.1']
y_L = np.log(data_L[(voltage_L < 3) & (voltage_L > 2.3) & (current_L > 0)]['A.1'])

# 함수 정의
def fit_function(x, a, b):
    y = a * x + b
    return y

# 각 직선에 대한 피팅
popt_k, pcov_k = curve_fit(fit_function, x_k, y_k)  
popt_g, pcov_g = curve_fit(fit_function, x_g, y_g)  
popt_m, pcov_m = curve_fit(fit_function, x_m, y_m)  
popt_L, pcov_L = curve_fit(fit_function, x_L, y_L)  


# 피팅된 기울기와 절편
m_k, b_k = popt_k
m_g, b_g = popt_g
m_m, b_m = popt_m
m_L, b_L = popt_L

# 오차 계산
m_k_err = np.sqrt(pcov_k[0, 0])
b_k_err = np.sqrt(pcov_k[1, 1])
m_g_err = np.sqrt(pcov_g[0, 0])
b_g_err = np.sqrt(pcov_g[1, 1])
m_m_err = np.sqrt(pcov_m[0, 0])
b_m_err = np.sqrt(pcov_m[1, 1])
m_L_err = np.sqrt(pcov_L[0, 0])
b_L_err = np.sqrt(pcov_L[1, 1])

# 결과 출력
print(f"y_k = ({m_k:.10f} ± {m_k_err:.10f}) * x + ({b_k:.10f} ± {b_k_err:.10f})")
print(f"y_g = ({m_g:.10f} ± {m_g_err:.10f}) * x + ({b_g:.10f} ± {b_g_err:.10f})")
print(f"y_m = ({m_m:.10f} ± {m_m_err:.10f}) * x + ({b_m:.10f} ± {b_m_err:.10f})")
print(f"y_L = ({m_L:.10f} ± {m_L_err:.10f}) * x + ({b_L:.10f} ± {b_L_err:.10f})")

val_m_k = Value(m_k, m_k_err)
val_b_k = Value(b_k, b_k_err)
val_m_g = Value(m_g, m_g_err)
val_b_g = Value(b_g, b_g_err)
val_m_m = Value(m_m, m_m_err)
val_b_m = Value(b_m, b_m_err)
val_m_L = Value(m_L, m_L_err)
val_b_L = Value(b_L, b_L_err)


# Sätttigungsstrom & Idealitätsfaktor 계산
# Sättigungsstrom Is = np.exp(val_b)
Is_k = np.exp(b_k)
Is_g = np.exp(b_g)
Is_m = np.exp(b_m)
Is_L = np.exp(b_L)
# Sättigungsstrom 오차 계산
Is_k_err = np.abs(Is_k * b_k_err)
Is_g_err = np.abs(Is_g * b_g_err)
Is_m_err = np.abs(Is_m * b_m_err)
Is_L_err = np.abs(Is_L * b_L_err)
# Idealitätsfaktor n = e / (k * T * val_m)
e = 1.6 * 1e-19
k = 1.38 * 1e-23
T = 298
n_k = e / (k * T * m_k)
n_g = e / (k * T * m_g)
n_m = e / (k * T * m_m)
n_L = e / (k * T * m_L)
# Idealitätsfaktor 오차 계산
n_k_err = np.abs(n_k * m_k_err)
n_g_err = np.abs(n_g * m_g_err)
n_m_err = np.abs(n_m * m_m_err)
n_L_err = np.abs(n_L * m_L_err)

# 결과 출력
print(f"Is_k = {Is_k:.10e} ± {Is_k_err:.10e}")
print(f"Is_g = {Is_g:.10e} ± {Is_g_err:.10e}")
print(f"Is_m = {Is_m:.10e} ± {Is_m_err:.10e}")
print(f"Is_L = {Is_L:.10e} ± {Is_L_err:.10e}")
print(f"n_k = {n_k:.10f} ± {n_k_err:.10f}")
print(f"n_g = {n_g:.10f} ± {n_g_err:.10f}")
print(f"n_m = {n_m:.10f} ± {n_m_err:.10f}")
print(f"n_L = {n_L:.10f} ± {n_L_err:.10f}")

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage_filtered_g, log_current_filtered_g, marker='o', color='orange', label='A_groß')  # 원래 데이터
plt.plot(x_g, fit_function(x_g, m_g, b_g), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_m, log_current_filtered_m, marker='o', color='g', label='B_mittel')  # 원래 데이터
plt.plot(x_m, fit_function(x_m, m_m, b_m), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_k, log_current_filtered_k, marker='o', color='r', label='C_klein')  # 원래 데이터
plt.plot(x_k, fit_function(x_k, m_k, b_k), 'b-')  # 피팅된 직선

plt.scatter(voltage_filtered_L, log_current_filtered_L, marker='o', color='black', label='L')  # 원래 데이터
plt.plot(x_L, fit_function(x_L, m_L, b_L), 'b-', label='Fitlinie')  # 피팅된 직선


# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('log Strom (A)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})



# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
plt.savefig('IV_B3204A_log.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()



# %% [markdown]
# Tabelle von IV-Messungen der verschiedenen Proben

# %%
import pandas as pd

# 데이터: (Probe 종류, Schwellenspannung 값, Sättigungsstrom 값, Idealitätsfaktor 값)
# A =: gross B =: mittel C =: klein
data = [
    ('B3297_A', (3.26, 0.28), (9.1623625507e-16, 1.4031558933e-15), (4.3546977984, 2.4704438190)),
    ('B3297_B', (3.4, 0.4), (1.1579517404e-13, 7.9573597481e-14), (5.5572204432, 1.2746538682)),
    ('B3297_C', (3.73, 0.23), (3.6299721378e-13, 3.7533549944e-13), (6.1811898705, 2.1323245322)),
    ('B3297_L', (3.3, 0.4), (3.2286753603e-15, 3.3263967607e-15), (4.5034314998, 1.7021519685)),
    
    ('B2396_A', (3.0, 0.5), (1.3451952944e-18, 9.5262815262e-19), (3.3330594904, 0.8019479918)),
    ('B2396_B', (3.00, 0.26), (1.0030456648e-17, 6.5934301145e-18), (3.6126709057, 0.7931602665)),
    ('B2396_C', (3.06, 0.16), (4.8241348620e-19, 4.0301884810e-19), (3.3871122821, 0.9694163362)),
    ('B2396_L', (2.86, 0.18), (1.4106993662e-18, 8.1083009752e-19), (3.3039543319, 0.6922862242)),
    
    ('B3204A_A', (3.00, 0.30), (1.6190668559e-11, 2.9842409177e-12), (6.3451451510, 0.4038996984)),
    ('B3204A_B', (3.16, 0.26), (8.0746538809e-12, 2.0696393889e-12), (6.3278035240, 0.5550231275)),
    ('B3204A_C', (3.23, 0.23), (1.2711654735e-12, 3.3348943150e-13), (5.9528332793, 0.5439746927)),
    ('B3204A_L', (2.86, 0.28), (4.0782181624e-12, 2.1182189811e-12), (5.6619884839, 1.0717075936)),
]


# 과학적 표기법을 사용하여 불확실성을 \pm 기호로 표시하는 함수
def format_with_uncertainty(value):
    nominal, uncertainty = value
    
    # 값이 1 이상이면 일반 소수점 두 자리 표현
    if nominal >= 1:
        return f"{nominal:.2f} ± {uncertainty:.2f}"
    
    # 값이 1 미만이면 과학적 표기법 사용
    nominal_str = f"{nominal:.2e}"
    nominal_base = float(nominal_str.split('e')[0])
    nominal_exponent = int(nominal_str.split('e')[-1])
    
    uncertainty_str = f"{uncertainty:.2e}"
    uncertainty_base = float(uncertainty_str.split('e')[0])
    uncertainty_exponent = int(uncertainty_str.split('e')[-1])
    
    if nominal_exponent == uncertainty_exponent:
        return f"({nominal_base:.2f} \\pm {uncertainty_base:.2f}) \\cdot 10^{{{nominal_exponent}}}"
    else:
        return f"{nominal_base:.2f} \\cdot 10^{{{nominal_exponent}}} \\pm {uncertainty_base:.2f} \\cdot 10^{{{uncertainty_exponent}}}"

# 데이터프레임 생성
df = pd.DataFrame(data, columns=['Probe', 'Schwellenspannung $U_\\mathrm{S}$ (V)', 'Sättigungsstrom $I_\\mathrm{S}$ (A)', 'Idealitätsfaktor n'])

# 각 열에 대해 불확실성 포맷팅 적용
df['Schwellenspannung $U_\\mathrm{S}$ (V)'] = df['Schwellenspannung $U_\\mathrm{S}$ (V)'].apply(format_with_uncertainty)
df['Sättigungsstrom $I_\\mathrm{S}$ (A)'] = df['Sättigungsstrom $I_\\mathrm{S}$ (A)'].apply(format_with_uncertainty)
df['Idealitätsfaktor n'] = df['Idealitätsfaktor n'].apply(format_with_uncertainty)

# LaTex 표 생성
latex_table = df.to_latex(
    index=False, 
    escape=False,  # LaTex 특수문자 이스케이프 방지
    column_format='c' * len(df.columns)  # 모든 열 가운데 정렬
)

# LaTex 표 시작 및 끝 추가
latex_table = "\\begin{table}[H]\n    \\centering\n    \\caption[IV-Messungen]{verschiedene Proben}\n" + latex_table + "\n\\end{table}"

# 결과 출력
print(latex_table)


# %% [markdown]
# Tabelle von gesamten Messdaten

# %% [markdown]
# \begin{table}[H]
#     \centering
#     \caption[IV-Messungen]{Die ermitteleten Messdaten der verschiedenen Proben bei IV-Messungen}
# \begin{tabular}{cccc}
# \toprule
# Probe & Schwellenspannung $U_\mathrm{S}$ (V) & Sättigungsstrom $I_\mathrm{S}$ (A) & Idealitätsfaktor n \\
# \midrule
# LED-Rot & 1.55 ± 0.15 & (9.78 \pm 6.70) \cdot 10^{-14} & 2.55 ± 1.31 \\
# LED-Orange & 1.67 ± 0.08 & (3.71 \pm 3.32) \cdot 10^{-21} & 1.61 ± 0.93 \\
# LED-Grün & 2.09 ± 0.15 & 2.06 \cdot 10^{-6} \pm 7.72 \cdot 10^{-8} & 58.89 ± 0.71 \\
# LED-Weiß & 2.63 ± 0.18 & (2.23 \pm 3.90) \cdot 10^{-30} & 1.64 ± 1.20 \\
# B3297_A & 3.26 ± 0.28 & 9.16 \cdot 10^{-16} \pm 1.40 \cdot 10^{-15} & 4.35 ± 2.47 \\
# B3297_B & 3.40 ± 0.40 & 1.16 \cdot 10^{-13} \pm 7.96 \cdot 10^{-14} & 5.56 ± 1.27 \\
# B3297_C & 3.73 ± 0.23 & (3.63 \pm 3.75) \cdot 10^{-13} & 6.18 ± 2.13 \\
# B3297_L & 3.30 ± 0.40 & (3.23 \pm 3.33) \cdot 10^{-15} & 4.50 ± 1.70 \\
# B2396_A & 3.00 ± 0.50 & 1.35 \cdot 10^{-18} \pm 9.53 \cdot 10^{-19} & 3.33 ± 0.80 \\
# B2396_B & 3.00 ± 0.26 & 1.00 \cdot 10^{-17} \pm 6.59 \cdot 10^{-18} & 3.61 ± 0.79 \\
# B2396_C & 3.06 ± 0.16 & (4.82 \pm 4.03) \cdot 10^{-19} & 3.39 ± 0.97 \\
# B2396_L & 2.86 ± 0.18 & 1.41 \cdot 10^{-18} \pm 8.11 \cdot 10^{-19} & 3.30 ± 0.69 \\
# B3204A_A & 3.00 ± 0.30 & 1.62 \cdot 10^{-11} \pm 2.98 \cdot 10^{-12} & 6.35 ± 0.40 \\
# B3204A_B & 3.16 ± 0.26 & (8.07 \pm 2.07) \cdot 10^{-12} & 6.33 ± 0.56 \\
# B3204A_C & 3.23 ± 0.23 & 1.27 \cdot 10^{-12} \pm 3.33 \cdot 10^{-13} & 5.95 ± 0.54 \\
# B3204A_L & 2.86 ± 0.28 & (4.08 \pm 2.12) \cdot 10^{-12} & 5.66 ± 1.07 \\
# \bottomrule
# \end{tabular}
# 
# \end{table}

# %% [markdown]
# CV-Messungen

# %% [markdown]
# CV-rot

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value
from formatter import format_with_uncertainty  as form

file_path = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/3.Aufgabe/CV_rot.csv'  # CSV 파일 경로

data = pd.read_csv(file_path, sep='\t', skiprows = 1 ,engine='python')

# # 데이터가 제대로 읽혔는지 확인
# print(data.head())

# # 데이터 추출
voltage = data['V.1']
capacitance = data['F'] 

yc = 1/capacitance**2

# 직선 피팅
p, residuals, _, _, _ = np.polyfit(voltage, yc, 1, full=True)
m, b = p  # 기울기와 절편 추출

# 피팅된 직선에 대한 잔차 구하기
residual_sum_of_squares = residuals[0]  # 잔차의 제곱합
degrees_of_freedom = len(voltage) - 2  # 자유도

# 잔차 표준 오차 계산
rse = np.sqrt(residual_sum_of_squares / degrees_of_freedom)

# 기울기와 절편의 오차 계산
x_mean = np.mean(voltage)
S_xx = np.sum((voltage - x_mean) ** 2)

m_error = rse / np.sqrt(S_xx)  # 기울기의 표준 오차
b_error = rse * np.sqrt(1 / len(voltage) + (x_mean ** 2) / S_xx)  # 절편의 표준 오차

# 결과 출력
print(f"Steigung: {m:.2e} ± {m_error:.2e}")
print(f"y Abschnitt : {b:.2e} ± {b_error:.2e}")

val_m_mit = Value(m, m_error)
val_b_mit = Value(b, b_error)

# Kontaktfläche von roter LED
A = 921.600 * 1e-12  # m^2
e = 1.602176634e-19
epsilon_0 = 8.854187817*1e-12  # F/m
epsilon_r = 13.1
epsilon = epsilon_0 * epsilon_r

# N_D 계산
N_D = 2 / (A**2 * e * epsilon * m)
# 편미분에 따른 오차 계산
partial_derivative = -2 / (A**2 * e * epsilon * m**2)
N_D_error = abs(partial_derivative) * m_error

# 결과 출력
print(f"N_D: {N_D:.2e} ± {N_D_error:.2e}")



# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(voltage, yc, marker='o', color='r', label='rot')  # 원래 데이터
plt.plot(x_fit, y_fit, 'r-', label=f'Fitlinie: y = {m:.2e}x + {b:.2e}')
# plt.plot(x1_fit, y1_fit, 'b-', label='Fitlinie 1')  # 첫 번째 피팅된 직선
# plt.plot(x2, fit_function(x2, m2, b2), 'g-', label='Fitlinie 2')  # 두 번째 피팅된 직선

# 그래프에 제목과 축 레이블 추가
plt.xlabel('Spannung (V)')  # x축 레이블
plt.ylabel('Kehrwert der quadrierten Kapazität (1/pF$^2$)')  # y축 레이블

# 범례 추가
plt.legend()

# plt.ylim(-0.0005, 0.026)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})


# 격자 추가
plt.grid(True)

# 그래프 pdf로 저장
# plt.savefig('CV_rot.pdf', dpi = 300, bbox_inches='tight')

# 그래프 표시
plt.show()


# %% [markdown]
# CV-Messung

# %% [markdown]
# CV-LEDs

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/3.Aufgabe'

dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

# Zuordnung von Dateinamen zu Farben
info = {
    'CV_rot.csv': {'farbe': 'red', 'label': 'LED Rot'},
    'CV_orange.csv': {'farbe': 'orange', 'label': 'LED Orange'},
    'CV_gruen.csv': {'farbe': 'green', 'label': 'LED Grün'},   
    'CV_weiss.csv': {'farbe': 'black', 'label': 'LED Weiß'}
}

for datei in dateien:
    # Vollständiger Pfad zur Datei
    voller_pfad = os.path.join(ordner_pfad, datei)
    
    # Lese die CSV-Datei, überspringe die ersten zwei Zeilen und setze den Delimiter auf Tab
    daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
    
    # Spaltennamen manuell
    daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
    
    # Berechne 1/C^2 
    daten['1/C^2'] = 1 / daten['Capacitance']**2

    farbe = info[datei]['farbe']
    label = info[datei]['label']
    
    # 라인
    #plt.plot(daten['VDC'], daten['1/C^2'], color=farbe)
    
    # 점점이 알파 값으로 투명도
    plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'{label}')


# lineare Regression
    fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
    a, b = fit_params[0], fit_params[1]
    fit_line = a * daten['VDC'] + b

        # Zeichne die Fit-Linie
    plt.plot(daten['VDC'], fit_line, color=info[datei]['farbe'])

        # Ausgabe der Fit-Parameter
    print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}")



# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)

# 결과 저장
plt.savefig('CV_kom_LED_ND.pdf', dpi = 300, bbox_inches='tight')
plt.show()




# %% [markdown]
# CV-Probe 3297

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/4.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv') and f in info]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

info = {
    'CV_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CV_gross.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CV_mittel.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CV_klein.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

for datei in dateien:
    # Vollständiger Pfad zur Datei
    voller_pfad = os.path.join(ordner_pfad, datei)
    
    # Lese die CSV-Datei, überspringe die ersten zwei Zeilen und setze den Delimiter auf Tab
    daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
    
    # Spaltennamen manuell
    daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
    
    # Berechne 1/C^2 
    daten['1/C^2'] = 1 / daten['Capacitance']**2

    farbe = info[datei]['farbe']
    label = info[datei]['label']
    
    # 라인
    #plt.plot(daten['VDC'], daten['1/C^2'], color=farbe)
    
    # 점점이 알파 값으로 투명도
    plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'{label}')


# lineare Regression
    fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
    a, b = fit_params[0], fit_params[1]
    fit_line = a * daten['VDC'] + b

        # Zeichne die Fit-Linie
    plt.plot(daten['VDC'], fit_line, color=info[datei]['farbe'])

        # Ausgabe der Fit-Parameter
    print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}")



# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)

# 결과 저장
plt.savefig('CV_3297_ND.pdf', dpi = 300, bbox_inches='tight')
plt.show()




# %% [markdown]
# CV-Probe 2396

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv') and f in info]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

info = {
    'CVP_B2396_2mhz_L.csv': {'farbe': 'tomato', 'label': '2396 L'},
    'CVP_B2396_2mhz_a.csv': {'farbe': 'limegreen', 'label': '2396 A'},
    'CVP_B2396_2mhz_b.csv': {'farbe': 'pink', 'label': '2396 B'},
    'CVP_B2396_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '2396 C'}
}

for datei in dateien:
    # Vollständiger Pfad zur Datei
    voller_pfad = os.path.join(ordner_pfad, datei)
    
    # Lese die CSV-Datei, überspringe die ersten zwei Zeilen und setze den Delimiter auf Tab
    daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
    
    # Spaltennamen manuell
    daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
    
    # Berechne 1/C^2 
    daten['1/C^2'] = 1 / daten['Capacitance']**2

    farbe = info[datei]['farbe']
    label = info[datei]['label']
    
    # 라인
    #plt.plot(daten['VDC'], daten['1/C^2'], color=farbe)
    
    # 점점이 알파 값으로 투명도
    plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'{label}')


# lineare Regression
    fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
    a, b = fit_params[0], fit_params[1]
    fit_line = a * daten['VDC'] + b

        # Zeichne die Fit-Linie
    plt.plot(daten['VDC'], fit_line, color=info[datei]['farbe'])

        # Ausgabe der Fit-Parameter
    print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}")



# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)

# 결과 저장
plt.savefig('CV_2396_ND.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# CV-Probe 3204

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv') and f in info]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3204 L'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3204 A'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'pink', 'label': '3204 B'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3204 C'}
}

for datei in dateien:
    # Vollständiger Pfad zur Datei
    voller_pfad = os.path.join(ordner_pfad, datei)
    
    # Lese die CSV-Datei, überspringe die ersten zwei Zeilen und setze den Delimiter auf Tab
    daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
    
    # Spaltennamen manuell
    daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
    
    # Berechne 1/C^2 
    daten['1/C^2'] = 1 / daten['Capacitance']**2

    farbe = info[datei]['farbe']
    label = info[datei]['label']
    
    # 라인
    #plt.plot(daten['VDC'], daten['1/C^2'], color=farbe)
    
    # 점점이 알파 값으로 투명도
    plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'{label}')


# lineare Regression
    fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
    a, b = fit_params[0], fit_params[1]
    fit_line = a * daten['VDC'] + b

        # Zeichne die Fit-Linie
    plt.plot(daten['VDC'], fit_line, color=info[datei]['farbe'])

        # Ausgabe der Fit-Parameter
    print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}")



# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)

# 결과 저장
plt.savefig('CV_3204_ND.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# ND(V) 추가해서 그리기

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from formatter import format_with_uncertainty 

ordner_pfad = '/Users/estern/FP-SS24/FP_Messdaten/F13/Messdaten/3.Aufgabe'

dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

# Zuordnung von Dateinamen zu Farben
info = {
    'CV_rot.csv': {'farbe': 'red', 'label': 'Rot'},
    'CV_orange.csv': {'farbe': 'orange', 'label': 'Orange'},
    'CV_gruen.csv': {'farbe': 'green', 'label': 'Grün'},   
    'CV_weiss.csv': {'farbe': 'black', 'label': 'Weiß'}
}

for datei in dateien:
    # Vollständiger Pfad zur Datei
    voller_pfad = os.path.join(ordner_pfad, datei)
    
    # Lese die CSV-Datei, überspringe die ersten zwei Zeilen und setze den Delimiter auf Tab
    daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
    
    # Spaltennamen manuell
    daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
    
    # Berechne 1/C^2 
    daten['1/C^2'] = 1 / daten['Capacitance']**2

    farbe = info[datei]['farbe']
    label = info[datei]['label']
    
    # 라인
    plt.plot(daten['VDC'], daten['1/C^2'], color=farbe)
    
    # 점점이 알파 값으로 투명도
    plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'{label}')


    # fit 함수 정의
    def fit_function(x, a, b):
        y = a * x + b
        return y

    # 각 직선에 대한 피팅
    popt, pcov = curve_fit(fit_function, daten['VDC'], daten['1/C^2'])  # 직선 피팅

    # 피팅된 기울기와 절편
    m, b = popt

    # 오차 계산
    m_err = np.sqrt(pcov[0, 0])
    b_err = np.sqrt(pcov[1, 1])

    # 결과 출력
    print(f"y = ({m:.10e} ± {m_err:.10e}) * x + ({b:.10e} ± {b_err:.10e})")

    # m_ges = (m, m_err)
    # b_ges = (b, b_err)

    # val_m = format_with_uncertainty(m_ges)
    # val_b = format_with_uncertainty(b_ges)

    # print(f"y = {val_m} * x + {val_b}")






# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})


plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# CV- Kontaktflächen- Proben

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from gp.messunsicherheiten import Value

A_kurz = 200.80*1e-6
A_lang = 211.24*1e-6

A_area = A_kurz ** 2
A_out_area = A_lang ** 2
A_area_err = A_out_area - A_area

A = Value(A_area, A_area_err)

B_kurz = 150.96*1e-6
B_lang = 161.88*1e-6

B_area = B_kurz ** 2
B_out_area = B_lang ** 2
B_area_err = B_out_area - B_area

B = Value(B_area, B_area_err)

C_kurz = 102.07*1e-6
C_lang = 112.08*1e-6

C_area = C_kurz ** 2
C_out_area = C_lang ** 2
C_area_err = C_out_area - C_area

C = Value(C_area, C_area_err)

L_kurz_breite_o = 372.66*1e-6
L_lang_breite_o = 383.14*1e-6
L_kurz_breite_u = 155.25*1e-6
L_lang_breite_u = 166.19*1e-6
L_kurz_hoehe_l = 374.07*1e-6
L_lang_hoehe_l = 382.62*1e-6
L_kurz_hoehe_r = 156.19*1e-6
L_lang_hoehe_r = 165.20*1e-6

L_area = (L_kurz_breite_o * L_kurz_hoehe_l) - (L_kurz_breite_o-L_kurz_breite_u)*(L_kurz_hoehe_l-L_kurz_hoehe_r)
L_out_area = (L_lang_breite_o * L_lang_hoehe_l) - (L_kurz_breite_o-L_kurz_breite_u)*(L_kurz_hoehe_l-L_kurz_hoehe_r)
L_area_err = L_out_area - L_area

L = Value(L_area, L_area_err)


# print(f"Area A = {A*1e12} μm^2")
# print(f"Area B = {B*1e12} μm^2")
# print(f"Area C = {C*1e12} μm^2")
# print(f"Area L = {L*1e12} μm^2")

# latex tabelle

import pandas as pd

# 데이터: (Probe 종류, Kontaktflächen 값)
# A =: gross B =: mittel C =: klein
data = [
    ('A', (A_area*1e12, A_area_err*1e12)),
    ('B', (B_area*1e12, B_area_err*1e12)),
    ('C', (C_area*1e12, C_area_err*1e12)),
    ('L', (L_area*1e12, L_area_err*1e12))
]


# 과학적 표기법을 사용하여 불확실성을 \pm 기호로 표시하는 함수
def format_with_uncertainty(value):
    nominal, uncertainty = value
    
    # 값이 1 이상이면 일반 소수점 두 자리 표현
    if nominal >= 1:
        return f"{nominal:.2f} ± {uncertainty:.2f}"
    
    # 값이 1 미만이면 과학적 표기법 사용
    nominal_str = f"{nominal:.2e}"
    nominal_base = float(nominal_str.split('e')[0])
    nominal_exponent = int(nominal_str.split('e')[-1])
    
    uncertainty_str = f"{uncertainty:.2e}"
    uncertainty_base = float(uncertainty_str.split('e')[0])
    uncertainty_exponent = int(uncertainty_str.split('e')[-1])
    
    if nominal_exponent == uncertainty_exponent:
        return f"({nominal_base:.2f} \\pm {uncertainty_base:.2f}) \\cdot 10^{{{nominal_exponent}}}"
    else:
        return f"{nominal_base:.2f} \\cdot 10^{{{nominal_exponent}}} \\pm {uncertainty_base:.2f} \\cdot 10^{{{uncertainty_exponent}}}"


# 데이터프레임 생성
df = pd.DataFrame(data, columns=['Probe', 'Kontaktfläche ($μm^2$)'])

# 각 열에 대해 불확실성 포맷팅 적용
df['Kontaktfläche ($μm^2$)'] = df['Kontaktfläche ($μm^2$)'].apply(format_with_uncertainty)

# LaTex 표 생성
latex_table = df.to_latex(
    index=False, 
    escape=False,  # LaTex 특수문자 이스케이프 방지
    column_format='c' * len(df.columns)  # 모든 열 가운데 정렬
)

# LaTex 표 시작 및 끝 추가
latex_table = "\\begin{table}[H]\n    \\centering\n    \\caption[Kontaktflächen]{verschiedene Proben}\n" + latex_table + "\n\\end{table}"
    
# 결과 출력
print(latex_table)





# %%


# %% [markdown]
# N_D2 Berechnung mit Differentialquotienten statt mittl. Änderungsrate

# %% [markdown]
# N diff kom. LED

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter
areas = {
    'CV_rot.csv': 921600 * 1e-12,
    'CV_orange.csv': 1000000 * 1e-12,
    'CV_gruen.csv': 2250000 * 1e-12,
    'CV_weiss.csv': 2250000 * 1e-12
}

# Materialzuordnung
material_map = {
    'CV_rot.csv': 'GaAs',
    'CV_orange.csv': 'GaAs',
    'CV_gruen.csv': 'GaN',
    'CV_weiss.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/3.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(10, 6))

# Für jeden Datensatz die Ergebnisse berechnen und speichern
results = []

for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Berechnung von 1/C^2
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        # Berechnung der numerischen Ableitung von 1/C^2 bezüglich VDC
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        
        # Berechnung von N für jeden Punkt
        area = areas[datei]
        epsilon = permittivity[material_map[datei]]
        
        daten['N'] = -2 / (area**2 * e * epsilon * d_one_over_c2_dv)

        # Ergebnisse sammeln für Ausgabe
        for index, row in daten.iterrows():
            results.append({
                'LED': datei,
                'VDC': row['VDC'],
                'N (m^-3)': row['N'],
                'd(1/C^2)/dV': row['1/C^2']
            })

# DataFrame mit den Ergebnissen erstellen
results_df = pd.DataFrame(results)
print(results_df)

plt.title('Änderung von N über die Spannung mit Diff')
plt.xlabel('Spannung VDC (V)')
plt.ylabel('N (m^-3)')
plt.grid(True)
plt.show()

# Optional: Ergebnisse als CSV speichern
#results_df.to_csv('N_Variation_Results.csv', index=False)



# %% [markdown]
# N diff Probe 3297

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

areas = {
    'CV_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CV_gross.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CV_mittel.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CV_klein.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CV_L.csv': 'GaN',
    'CV_gross.csv': 'GaN',
    'CV_mittel.csv': 'GaN',
    'CV_klein.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/4.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(10, 6))

info = {
    'CV_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CV_gross.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CV_mittel.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CV_klein.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

# Für jeden Datensatz die Ergebnisse berechnen und speichern
results = []

for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Berechnung von 1/C^2
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        # Berechnung der numerischen Ableitung von 1/C^2 bezüglich VDC
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        
        # Berechnung von N für jeden Punkt
        
        A = areas[datei]['area']  # Spezifische Fläche für die Datei
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne die Variable 'N' (ich nehme an, d_one_over_c2_dv ist bereits definiert)
        daten['N'] = -2 / (A**2 * e * epsilon * d_one_over_c2_dv)

        # Ergebnisse sammeln für Ausgabe
        for index, row in daten.iterrows():
            results.append({
                'LED': datei,
                'VDC': row['VDC'],
                'N (m^-3)': row['N'],
                'd(1/C^2)/dV': row['1/C^2']
            })

# DataFrame mit den Ergebnissen erstellen
results_df = pd.DataFrame(results)
print(results_df)

plt.title('Änderung von N über die Spannung mit Diff')
plt.xlabel('Spannung VDC (V)')
plt.ylabel('N (m^-3)')
plt.grid(True)
plt.show()

results_df.to_csv('N_Variation_Results_3297.csv', index=False)

# %% [markdown]
# N diff Probe 2396

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

areas = {
    'CVP_B2396_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B2396_2mhz_a.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B2396_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B2396_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B2396_2mhz_L.csv': 'GaN',
    'CVP_B2396_2mhz_a.csv': 'GaN',
    'CVP_B2396_2mhz_b.csv': 'GaN',
    'CVP_B2396_2mhz_c.csv' : 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B2396_2mhz_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CVP_B2396_2mhz_a.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CVP_B2396_2mhz_b.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CVP_B2396_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

# Für jeden Datensatz die Ergebnisse berechnen und speichern
results = []

for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Berechnung von 1/C^2
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        # Berechnung der numerischen Ableitung von 1/C^2 bezüglich VDC
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        
        # Berechnung von N für jeden Punkt
        
        A = areas[datei]['area']  # Spezifische Fläche für die Datei
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne die Variable 'N' (ich nehme an, d_one_over_c2_dv ist bereits definiert)
        daten['N'] = -2 / (A**2 * e * epsilon * d_one_over_c2_dv)

        # Ergebnisse sammeln für Ausgabe
        for index, row in daten.iterrows():
            results.append({
                'LED': datei,
                'VDC': row['VDC'],
                'N (m^-3)': row['N'],
                'd(1/C^2)/dV': row['1/C^2']
            })

# DataFrame mit den Ergebnissen erstellen
results_df = pd.DataFrame(results)
print(results_df)

plt.title('Änderung von N über die Spannung mit Diff')
plt.xlabel('Spannung VDC (V)')
plt.ylabel('N (m^-3)')
plt.grid(True)
plt.show()

results_df.to_csv('N_Variation_Results_2396.csv', index=False)

# %% [markdown]
# N diff Probe 3204

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

areas = {
    'CVP_B3204A_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B3204A_2mhz_a_V2.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B3204A_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B3204A_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B3204A_2mhz_L.csv': 'GaN',
    'CVP_B3204A_2mhz_a_V2.csv': 'GaN',
    'CVP_B3204A_2mhz_b.csv': 'GaN',
    'CVP_B3204A_2mhz_c.csv' : 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3204 L'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3204 A'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'pink', 'label': '3204 B'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3204 C'}
}

# Für jeden Datensatz die Ergebnisse berechnen und speichern
results = []

for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Berechnung von 1/C^2
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        # Berechnung der numerischen Ableitung von 1/C^2 bezüglich VDC
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        
        # Berechnung von N für jeden Punkt
        
        A = areas[datei]['area']  # Spezifische Fläche für die Datei
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne die Variable 'N' (ich nehme an, d_one_over_c2_dv ist bereits definiert)
        daten['N'] = -2 / (A**2 * e * epsilon * d_one_over_c2_dv)

        # Ergebnisse sammeln für Ausgabe
        for index, row in daten.iterrows():
            results.append({
                'LED': datei,
                'VDC': row['VDC'],
                'N (m^-3)': row['N'],
                'd(1/C^2)/dV': row['1/C^2']
            })

# DataFrame mit den Ergebnissen erstellen
results_df = pd.DataFrame(results)
print(results_df)

plt.title('Änderung von N über die Spannung mit Diff')
plt.xlabel('Spannung VDC (V)')
plt.ylabel('N (m^-3)')
plt.grid(True)
plt.show()

results_df.to_csv('N_Variation_Results_3204.csv', index=False)

# %% [markdown]
# N_D V_bi berechnen-----------------------------------------------------------------------

# %% [markdown]
# N_D, V_bi kom. LED

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,  # Permittivität von GaAs in F/m
    'GaN': 9 * 8.854e-12    # Permittivität von GaN in F/m
}

# Flächen in Quadratmeter umrechnen (von Quadratmikrometern)
areas = {
    'CV_rot.csv': 921600 * 1e-12,  # umgerechnet in m^2
    'CV_orange.csv': 1000000 * 1e-12,
    'CV_gruen.csv': 2250000 * 1e-12,
    'CV_weiss.csv': 2250000 * 1e-12
}

# Materialzuordnung
material_map = {
    'CV_rot.csv': 'GaAs',
    'CV_orange.csv': 'GaAs',
    'CV_gruen.csv': 'GaN',
    'CV_weiss.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/3.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

# Zuordnung von Dateinamen zu Farben und Labels
info = {
    'CV_rot.csv': {'farbe': 'red', 'label': 'LED Rot'},
    'CV_orange.csv': {'farbe': 'orange', 'label': 'LED Orange'},
    'CV_gruen.csv': {'farbe': 'green', 'label': 'LED Grün'},
    'CV_weiss.csv': {'farbe': 'black', 'label': 'LED Weiß'}
}

for datei in dateien:
    if datei in info:  # Überprüfen, ob die Datei in info vorhanden ist
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        daten['1/C^2'] = 1 / daten['Capacitance']**2

        farbe = info[datei]['farbe']
        label = info[datei]['label']

        # Plot lines and points
        plt.plot(daten['VDC'], daten['1/C^2'], color=farbe, label=label)
        plt.scatter(daten['VDC'], daten['1/C^2'], color=farbe, edgecolor='grey', alpha=0.5, s=20, label=f'Scatter {label}')

        # Perform linear regression
        fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
        a, b = fit_params[0], fit_params[1]
        fit_line = a * daten['VDC'] + b
        plt.plot(daten['VDC'], fit_line, label=f"Fit: {label}", color=farbe, linestyle='--')

        # Calculate V_bi and N
        V_bi = -b / a
        material = material_map[datei]
        epsilon = permittivity[material]
        area = areas[datei]
        N = -2 / (area**2 * e * epsilon * a)

        # Output the calculated values
        print(f"Fit für {label}: a = {a:.4f}, b = {b:.4e}, V_bi = {V_bi:.4f} V, N = {N:.4e} m^-3")

plt.title('C-V Messungen für kommerzielle LEDs')
plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# N_D V_bi Probe 3297

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,  # Permittivität von GaAs in F/m
    'GaN': 9 * 8.854e-12    # Permittivität von GaN in F/m
}

areas = {
    'CV_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CV_gross.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CV_mittel.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CV_klein.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CV_L.csv': 'GaN',
    'CV_gross.csv': 'GaN',
    'CV_mittel.csv': 'GaN',
    'CV_klein.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/4.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(10, 6))

info = {
    'CV_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CV_gross.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CV_mittel.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CV_klein.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        daten['1/C^2'] = 1 / daten['Capacitance']**2

        # Lineare Regression durchführen
        fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
        a, b = fit_params

        # Parameter zur Berechnung von N_D und V_bi
        A = areas[datei]['area']
        epsilon = permittivity[material_map[datei]]
        N_D = 2 / (e * epsilon * A**2 * a)
        V_bi = -b / a

        # Plotte Daten und Regressionslinie
        plt.plot(daten['VDC'], daten['1/C^2'] * A**2, label=f'Fit: {info[datei]["label"]}')
        plt.scatter(daten['VDC'], daten['1/C^2'] * A**2, label=f'Scatter {info[datei]["label"]}', alpha=0.5)

        print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}, V_bi = {V_bi:.4f} V, N_D = {N_D:.4e} m^-3")


plt.title('C-V Messungen für kommerzielle LEDs')
plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# N_D V_bi Probe 2396

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,  # Permittivität von GaAs in F/m
    'GaN': 9 * 8.854e-12    # Permittivität von GaN in F/m
}

areas = {
    'CVP_B2396_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B2396_2mhz_a.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B2396_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B2396_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B2396_2mhz_L.csv': 'GaN',
    'CVP_B2396_2mhz_a.csv': 'GaN',
    'CVP_B2396_2mhz_b.csv': 'GaN',
    'CVP_B2396_2mhz_c.csv' : 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B2396_2mhz_L.csv': {'farbe': 'tomato', 'label': '2396 L'},
    'CVP_B2396_2mhz_a.csv': {'farbe': 'limegreen', 'label': '2396 A'},
    'CVP_B2396_2mhz_b.csv': {'farbe': 'pink', 'label': '2396 B'},
    'CVP_B2396_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '2396 C'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        daten['1/C^2'] = 1 / daten['Capacitance']**2

        # Lineare Regression durchführen
        fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
        a, b = fit_params

        # Parameter zur Berechnung von N_D und V_bi
        A = areas[datei]['area']
        epsilon = permittivity[material_map[datei]]
        N_D = 2 / (e * epsilon * A**2 * a)
        V_bi = -b / a

        # Plotte Daten und Regressionslinie
        plt.plot(daten['VDC'], daten['1/C^2'] * A**2, label=f'Fit: {info[datei]["label"]}')
        plt.scatter(daten['VDC'], daten['1/C^2'] * A**2, label=f'Scatter {info[datei]["label"]}', alpha=0.5)

        print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}, V_bi = {V_bi:.4f} V, N_D = {N_D:.4e} m^-3")


plt.title('C-V Messungen für die Probe 2396')
plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# N_D V_bi Probe 3204

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,  # Permittivität von GaAs in F/m
    'GaN': 9 * 8.854e-12    # Permittivität von GaN in F/m
}

areas = {
    'CVP_B3204A_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B3204A_2mhz_a_V2.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B3204A_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B3204A_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B3204A_2mhz_L.csv': 'GaN',
    'CVP_B3204A_2mhz_a_V2.csv': 'GaN',
    'CVP_B3204A_2mhz_b.csv': 'GaN',
    'CVP_B3204A_2mhz_c.csv' : 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3204 L'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3204 A'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'pink', 'label': '3204 B'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3204 C'}
}


for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        daten['1/C^2'] = 1 / daten['Capacitance']**2

        # Lineare Regression durchführen
        fit_params = np.polyfit(daten['VDC'], daten['1/C^2'], 1)
        a, b = fit_params

        # Parameter zur Berechnung von N_D und V_bi
        A = areas[datei]['area']
        epsilon = permittivity[material_map[datei]]
        N_D = 2 / (e * epsilon * A**2 * a)
        V_bi = -b / a

        # Plotte Daten und Regressionslinie
        plt.plot(daten['VDC'], daten['1/C^2'] * A**2, label=f'Fit: {info[datei]["label"]}')
        plt.scatter(daten['VDC'], daten['1/C^2'] * A**2, label=f'{info[datei]["label"]}', alpha=0.5)

        print(f"Fit für {info[datei]['label']}: a = {a:.4f}, b = {b:.4e}, V_bi = {V_bi:.4f} V, N_D = {N_D:.4e} m^-3")


plt.title('C-V Messungen für die Probe 3204')
plt.xlabel('Spannung (V)')
plt.ylabel('1/C² (1/F²)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# W(V) Messung LED---------------------------------------------------------------------------

# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CV_rot.csv': 921600 * 1e-12,  # umgerechnet in m^2
    'CV_orange.csv': 1000000 * 1e-12,
    'CV_gruen.csv': 2250000 * 1e-12
}

# Materialzuordnung
material_map = {
    'CV_rot.csv': 'GaAs',
    'CV_orange.csv': 'GaAs',
    'CV_gruen.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/3.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))  # Erstelle eine Figure für die Plots

# Zuordnung von Dateinamen zu Farben und Labels
info = {
    'CV_rot.csv': {'farbe': 'red', 'label': 'Rot'},
    'CV_orange.csv': {'farbe': 'orange', 'label': 'Orange'},
    'CV_gruen.csv': {'farbe': 'green', 'label': 'Grün'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
       
        A = areas[datei]
        
        #delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * A / daten['Capacitance']) * 1e6  # W in Mikrometern
        #daten['delta_W'] = daten['W'] * (delta_A / A)  # Fehlerfortpflanzung für W
        plt.scatter(daten['VDC'], daten['W'], color=info[datei]['farbe'], edgecolor='grey', label=info[datei]['label'])

        # Plot mit Fehlerbalken
        

#plt.title('Probe 3297')
plt.xlabel('Spannung (V)')
plt.ylabel('Breite der Raumladunszone W (µm)')

plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()

# Definiere die gewünschte Reihenfolge der Labels
desired_order = ['Rot', 'Orange', 'Grün']

# Ordne die Handles basierend auf der gewünschten Reihenfolge neu an
ordered_handles = [handles[labels.index(label)] for label in desired_order if label in labels]

# Erstelle die Legende mit den neu geordneten Handles
plt.legend(ordered_handles, desired_order)


# 결과 저장
plt.savefig('WV_kom_LED.pdf', dpi = 300, bbox_inches='tight')
plt.show()

# %% [markdown]
# W(V) Messung Probe 3297

# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CV_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CV_gross.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CV_mittel.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CV_klein.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CV_L.csv': 'GaN',
    'CV_gross.csv': 'GaN',
    'CV_mittel.csv': 'GaN',
    'CV_klein.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/4.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CV_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CV_gross.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CV_mittel.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CV_klein.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        A = areas[datei]['area']
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * A / daten['Capacitance']) * 1e6  # W in Mikrometern
        daten['delta_W'] = daten['W'] * (delta_A / A)  # Fehlerfortpflanzung für W
        plt.scatter(daten['VDC'], daten['W'], color=info[datei]['farbe'], edgecolor='grey')

        # Plot mit Fehlerbalken
        plt.errorbar(daten['VDC'], daten['W'], yerr=daten['delta_W'], fmt='o', alpha=0.7, capsize=3, color=info[datei]['farbe'], label=info[datei]['label'])

#plt.title('Probe 3297')
plt.xlabel('Spannung (V)')
plt.ylabel('Breite der Raumladunszone W (µm)')
plt.legend()
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['3297 A', '3297 B', '3297 C', '3297 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)
# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('WV_3297.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# W(v) Probe 2396

# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CVP_B2396_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B2396_2mhz_a.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B2396_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B2396_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B2396_2mhz_L.csv': 'GaN',
    'CVP_B2396_2mhz_a.csv': 'GaN',
    'CVP_B2396_2mhz_b.csv': 'GaN',
    'CVP_B2396_2mhz_c.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B2396_2mhz_L.csv': {'farbe': 'tomato', 'label': '2396 L'},
    'CVP_B2396_2mhz_a.csv': {'farbe': 'limegreen', 'label': '2396 A'},
    'CVP_B2396_2mhz_b.csv': {'farbe': 'pink', 'label': '2396 B'},
    'CVP_B2396_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '2396 C'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        A = areas[datei]['area']
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * A / daten['Capacitance']) * 1e6  # W in Mikrometern
        daten['delta_W'] = daten['W'] * (delta_A / A)  # Fehlerfortpflanzung für W
        
        plt.scatter(daten['VDC'], daten['W'], color=info[datei]['farbe'], edgecolor='grey')

        # Plot mit Fehlerbalken
        plt.errorbar(daten['VDC'], daten['W'], yerr=daten['delta_W'], fmt='o', color=info[datei]['farbe'], alpha=0.7, capsize=3, label=info[datei]['label'])

#plt.title('Probe 2396')

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['2396 A', '2396 B', '2396 C', '2396 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)

plt.xlabel('Spannung (V)')
plt.ylabel('Breite der Raumladunszone W (µm)')
plt.grid(True)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('WV_2396.pdf', dpi = 300, bbox_inches='tight')
plt.show()

# %% [markdown]
# W(v) Probe 3204
# 
# A und B schneiden sich fasst exakt zu einander. Für die bessere Übersicht ist der Graph noch mal mit dem anderen Code versucht worden.

# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CVP_B3204A_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B3204A_2mhz_a_V2.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B3204A_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B3204A_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B3204A_2mhz_L.csv': 'GaN',
    'CVP_B3204A_2mhz_a_V2.csv': 'GaN',
    'CVP_B3204A_2mhz_b.csv': 'GaN',
    'CVP_B3204A_2mhz_c.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        A = areas[datei]['area']
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * A / daten['Capacitance']) * 1e6  # W in Mikrometern
        daten['delta_W'] = daten['W'] * (delta_A / A)  # Fehlerfortpflanzung für W

        # Plot mit Fehlerbalken
        plt.errorbar(daten['VDC'], daten['W'], yerr=daten['delta_W'], fmt='o', color=info[datei]['farbe'], ecolor='gray', alpha=0.5, capsize=5, label=info[datei]['label'])

plt.title('Probe 3204')
plt.xlabel('Spannung (V)')
plt.ylabel('Breite der Raumladunszone W (µm)')
plt.legend()
plt.grid(True)
plt.show()

#글씨 크기랑 폼은 나중에 맞출게!


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CVP_B3204A_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B3204A_2mhz_a_V2.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B3204A_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B3204A_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B3204A_2mhz_L.csv': 'GaN',
    'CVP_B3204A_2mhz_a_V2.csv': 'GaN',
    'CVP_B3204A_2mhz_b.csv': 'GaN',
    'CVP_B3204A_2mhz_c.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))



info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3204 L', 'marker': 'o'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3204 A', 'marker': 's'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'purple', 'label': '3204 B', 'marker': '^'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3204 C', 'marker': 'D'}
}

jitter = {'3297 A': -0.02, '3297 B': 0.02}  # Beispielwerte für x-Achsen-Verschiebung

for datei in dateien:
    if datei in info:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        A = areas[datei]['area']
        delta_A = areas[datei]['delta_area']
        epsilon = permittivity[material_map[datei]]

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * A / daten['Capacitance']) * 1e6  # W in Mikrometern
        daten['delta_W'] = daten['W'] * (delta_A / A)  # Fehlerfortpflanzung für W
    # Datenvorbereitung und Berechnungen
        daten['VDC_jitter'] = daten['VDC'] + jitter.get(info[datei]['label'], 0)  # Jitter hinzufügen

        # Plot mit Fehlerbalken
        plt.errorbar(daten['VDC_jitter'], daten['W'], yerr=daten['delta_W'], fmt=info[datei]['marker'], color=info[datei]['farbe'], ecolor=info[datei]['farbe'], alpha=0.7, capsize=3, label=info[datei]['label'])

# 그래프 생성 후 레전드 순서 변경
handles, labels = plt.gca().get_legend_handles_labels()

# 원하는 순서에 따라 핸들과 라벨을 재정렬
desired_order = ['3204 A', '3204 B', '3204 C', '3204 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]

# 변경된 순서로 레전드를 재생성
plt.legend(ordered_handles, desired_order)


#plt.title('Probe 3294')
plt.xlabel('Spannung (V)')
plt.ylabel('Breite der Raumladunszone W (µm)')

plt.grid(True)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('WV_3204.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# N(W) mit N=N_diff

# %% [markdown]
# N(W) kom.LED

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten und Materialinformationen
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter
areas = {
    'CV_rot.csv': 921600 * 1e-12,
    'CV_orange.csv': 1000000 * 1e-12,
    'CV_gruen.csv': 2250000 * 1e-12,
}
info = {
    'CV_rot.csv': {'farbe': 'red', 'label': 'Rot'},
    'CV_orange.csv': {'farbe': 'orange', 'label': 'Orange'},
    'CV_gruen.csv': {'farbe': 'green', 'label': 'Grün'},   
}

material_map = {
    'CV_rot.csv': 'GaAs',
    'CV_orange.csv': 'GaAs',
    'CV_gruen.csv': 'GaN',
}

# Ordnerpfad und Dateien festlegen
ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/3.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(10, 6))

# Daten berechnen und plotten
for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # W und N berechnen
        epsilon = permittivity['GaAs' if 'GaAs' in datei else 'GaN']
        area = areas[datei]

        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        daten['W'] = epsilon * area / daten['Capacitance'] * 1e6  # Umrechnung in Mikrometer
        daten['N'] = -2 / (area**2 * e * epsilon * np.gradient(daten['1/C^2'], daten['VDC']))

        # Plot erstellen
        plt.scatter(daten['W'], daten['N'], label=info[datei]['label'], color=info[datei]['farbe'], alpha=0.6, edgecolor='grey')
#plt.title('Dotierungskonzentration N als Funktion der Breite der Raumladungszone W')
plt.xlabel('Breite der Raumladungszone W (µm)')
plt.ylabel('Dotierungskonzentration N (m^-3)')
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['Rot', 'Orange', 'Grün']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('NW_kom_LED.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# N(W) 3297

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten und Materialinformationen
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CV_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CV_gross.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CV_mittel.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CV_klein.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CV_L.csv': 'GaN',
    'CV_gross.csv': 'GaN',
    'CV_mittel.csv': 'GaN',
    'CV_klein.csv': 'GaN'
}

info = {
    'CV_L.csv': {'farbe': 'tomato', 'label': '3297 L'},
    'CV_gross.csv': {'farbe': 'limegreen', 'label': '3297 A'},
    'CV_mittel.csv': {'farbe': 'pink', 'label': '3297 B'},
    'CV_klein.csv': {'farbe': 'cornflowerblue', 'label': '3297 C'}
}
ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten/4.Aufgabe'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

# Daten berechnen und plotten
for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Wichtig: Berechnung von 1/C^2 sicherstellen
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        epsilon = permittivity['GaAs' if 'GaAs' in datei else 'GaN']
        area = areas[datei]['area']
        delta_area = areas[datei]['delta_area']

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * area / daten['Capacitance']) * 1e6  # Umrechnung in Mikrometer
        daten['delta_W'] = daten['W'] * (delta_area / area)  # Fehlerfortpflanzung

        # Berechne N und Fehler in N
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        daten['N'] = -2 / (area**2 * e * epsilon * d_one_over_c2_dv)
        daten['delta_N'] = daten['N'] * (delta_area / area)  # Vereinfachte Fehlerfortpflanzung

        # Plot mit Fehlerbalken
        plt.scatter(daten['W'], daten['N'], label=info[datei]['label'], alpha=0.6, edgecolor='grey')


#plt.title('Dotierungskonzentration N als Funktion der Breite der Raumladungszone W mit Unsicherheiten')
plt.xlabel('Breite der Raumladungszone W (µm)')
plt.ylabel('Dotierungskonzentration N (m^-3)')
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['3297 A', '3297 B', '3297 C', '3297 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('NW_3297.pdf', dpi = 300, bbox_inches='tight')
plt.show()



# %% [markdown]
# N(W) 2396

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten und Materialinformationen
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

# Flächen in Quadratmeter und ihre Unsicherheiten
areas = {
    'CVP_B2396_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B2396_2mhz_a.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B2396_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B2396_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B2396_2mhz_L.csv': 'GaN',
    'CVP_B2396_2mhz_a.csv': 'GaN',
    'CVP_B2396_2mhz_b.csv': 'GaN',
    'CVP_B2396_2mhz_c.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(12, 9))

info = {
    'CVP_B2396_2mhz_L.csv': {'farbe': 'tomato', 'label': '2396 L'},
    'CVP_B2396_2mhz_a.csv': {'farbe': 'limegreen', 'label': '2396 A'},
    'CVP_B2396_2mhz_b.csv': {'farbe': 'pink', 'label': '2396 B'},
    'CVP_B2396_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '2396 C'}
}

# Daten berechnen und plotten
for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Wichtig: Berechnung von 1/C^2 sicherstellen
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        epsilon = permittivity['GaAs' if 'GaAs' in datei else 'GaN']
        area = areas[datei]['area']
        delta_area = areas[datei]['delta_area']

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * area / daten['Capacitance']) * 1e6  # Umrechnung in Mikrometer
        daten['delta_W'] = daten['W'] * (delta_area / area)  # Fehlerfortpflanzung

        # Berechne N und Fehler in N
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        daten['N'] = -2 / (area**2 * e * epsilon * d_one_over_c2_dv)
        daten['delta_N'] = daten['N'] * (delta_area / area)  # Vereinfachte Fehlerfortpflanzung

        # Plot mit Fehlerbalken
        plt.scatter(daten['W'], daten['N'], label=info[datei]['label'], alpha=0.6, edgecolor='grey')


#plt.title('Dotierungskonzentration N als Funktion der Breite der Raumladungszone W mit Unsicherheiten')
plt.xlabel('Breite der Raumladungszone W (µm)')
plt.ylabel('Dotierungskonzentration N (m^-3)')
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['2396 A', '2396 B', '2396 C', '2396 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('NW_2396.pdf', dpi = 300, bbox_inches='tight')
plt.show()


# %% [markdown]
# N(W) 3204

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Konstanten und Materialinformationen
e = 1.602e-19  # Elementarladung in Coulomb
permittivity = {
    'GaAs': 13.1 * 8.854e-12,
    'GaN': 9 * 8.854e-12
}

areas = {
    'CVP_B3204A_2mhz_L.csv': {'area': 92000 * 1e-12, 'delta_area': 8000 * 1e-12},
    'CVP_B3204A_2mhz_a_V2.csv': {'area': 40000 * 1e-12, 'delta_area': 5000 * 1e-12},
    'CVP_B3204A_2mhz_b.csv': {'area': 23000 * 1e-12, 'delta_area': 4000 * 1e-12},
    'CVP_B3204A_2mhz_c.csv': {'area': 10000 * 1e-12, 'delta_area': 2200 * 1e-12}
}

material_map = {
    'CVP_B3204A_2mhz_L.csv': 'GaN',
    'CVP_B3204A_2mhz_a_V2.csv': 'GaN',
    'CVP_B3204A_2mhz_b.csv': 'GaN',
    'CVP_B3204A_2mhz_c.csv': 'GaN'
}

ordner_pfad = '/Users/naebaekhap/FP_Protokoll_Kim/FP_Messdaten/F13/Messdaten'
dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.csv')]

plt.figure(figsize=(8, 6))

info = {
    'CVP_B3204A_2mhz_L.csv': {'farbe': 'tomato', 'label': '3204 L'},
    'CVP_B3204A_2mhz_a_V2.csv': {'farbe': 'limegreen', 'label': '3204 A'},
    'CVP_B3204A_2mhz_b.csv': {'farbe': 'pink', 'label': '3204 B'},
    'CVP_B3204A_2mhz_c.csv': {'farbe': 'cornflowerblue', 'label': '3204 C'}
}

# Daten berechnen und plotten
for datei in dateien:
    if datei in areas:
        voller_pfad = os.path.join(ordner_pfad, datei)
        daten = pd.read_csv(voller_pfad, delimiter='\t', skiprows=2, header=None)
        daten.columns = ['Frequency', 'Capacitance', 'Resistance', 'VAC', 'IAC', 'VDC', 'IDC', 'Time', 'Impedance', 'Temperature']
        
        # Wichtig: Berechnung von 1/C^2 sicherstellen
        daten['1/C^2'] = 1 / daten['Capacitance']**2
        
        epsilon = permittivity['GaAs' if 'GaAs' in datei else 'GaN']
        area = areas[datei]['area']
        delta_area = areas[datei]['delta_area']

        # Berechne W und Fehler in W
        daten['W'] = (epsilon * area / daten['Capacitance']) * 1e6  # Umrechnung in Mikrometer
        daten['delta_W'] = daten['W'] * (delta_area / area)  # Fehlerfortpflanzung

        # Berechne N und Fehler in N
        d_one_over_c2_dv = np.gradient(daten['1/C^2'], daten['VDC'])
        daten['N'] = -2 / (area**2 * e * epsilon * d_one_over_c2_dv)
        daten['delta_N'] = daten['N'] * (delta_area / area)  # Vereinfachte Fehlerfortpflanzung

        # Plot mit Fehlerbalken
        plt.scatter(daten['W'], daten['N'], label=info[datei]['label'], alpha=0.6, edgecolor='grey')


#plt.title('Dotierungskonzentration N als Funktion der Breite der Raumladungszone W mit Unsicherheiten')
plt.xlabel('Breite der Raumladungszone W (µm)')
plt.ylabel('Dotierungskonzentration N (m^-3)')
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
desired_order = ['3204 A', '3204 B', '3204 C', '3204 L']
ordered_handles = [handles[labels.index(label)] for label in desired_order]
plt.legend(ordered_handles, desired_order)

# font size
plt.rcParams.update({
    'font.size': 15,  # 기본 글자 크기 설정
    'axes.labelsize': 15,  # x, y 축 레이블 글자 크기
    'xtick.labelsize': 15,  # x축 틱 레이블 글자 크기
    'ytick.labelsize': 15,  # y축 틱 레이블 글자 크기
    'legend.fontsize': 15  # 범례 글자 크기
})

# 결과 저장
plt.savefig('NW_3204.pdf', dpi = 300, bbox_inches='tight')
plt.show()



