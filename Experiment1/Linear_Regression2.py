import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
income = pd.read_csv(r"line-ext.csv")

print("=" * 50)
print("方法一：手动计算（最小二乘法公式）")
print("=" * 50)

n = income.shape[0]
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()

w_manual = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x**2 / n)
b_manual = income.Salary.mean() - w_manual * income.YearsExperience.mean()
print(f"回归参数 w：{w_manual}")
print(f"回归参数 b：{b_manual}")

pred_manual = w_manual * income.YearsExperience + b_manual
mse_manual = ((income.Salary - pred_manual) ** 2).mean()
print(f"MSE：{mse_manual}")

x_new_value = 0.8452
y_pred_manual = w_manual * x_new_value + b_manual
print(f"x=0.8452 时预测值：{y_pred_manual}")

print("\n" + "=" * 50)
print("方法二：statsmodels 计算")
print("=" * 50)

fit = sm.formula.ols("Salary ~ YearsExperience", data=income).fit()
print(f"回归参数：\n{fit.params}")
print(f"MSE：{((income.Salary - fit.predict(income)) ** 2).mean()}")

x_new_df = pd.DataFrame({'YearsExperience': [0.8452]})
y_pred_sm = fit.predict(x_new_df)
print(f"x=0.8452 时预测值：{y_pred_sm.values[0]}")

print("\n" + "=" * 50)
print("方法三：sklearn 计算")
print("=" * 50)

X = income[['YearsExperience']]
y = income['Salary']
model = LinearRegression()
model.fit(X, y)

print(f"回归参数 w：{model.coef_[0]}")
print(f"回归参数 b：{model.intercept_}")
print(f"MSE：{((y - model.predict(X)) ** 2).mean()}")

x_new_sk = pd.DataFrame({'YearsExperience': [0.8452]})
y_pred_sk = model.predict(x_new_sk)
print(f"x=0.8452 时预测值：{y_pred_sk[0]}")
