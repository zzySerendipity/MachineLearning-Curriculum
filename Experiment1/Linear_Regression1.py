# 导入第三方模块
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 导入数据集
income = pd.read_csv(r"line-ext.csv")

# # 绘制散点图
# sns.lmplot(x="YearsExperience", y="Salary", data=income, ci=None)
# # 显示图形
# plt.show()

# 样本量
n = income.shape[0]
# 计算自变量、因变量、自变量平方、自变量与因变量乘积的和
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()
# 根据公式计算回归模型的参数
w = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x**2 / n)
b = income.Salary.mean() - w * income.YearsExperience.mean()
# 打印出计算结果
print("回归参数 w 的值：", w)
print("回归参数 b 的值：", b)
print("模型表达式：f(x)=", w, "x+", b)
# 打印出均方误差
pred = w * income.YearsExperience + b
mse = ((income.Salary - pred) ** 2).mean()
print("mse:", mse)
# 打印出变量 x =0.8452 时，变量 y 的预测值
x_new = 0.8452
y_pred = w * x_new + b
print("当输入x=0.8452时，y_pred=", y_pred)

# 绘制散点图及回归线
sns.lmplot(x="YearsExperience", y="Salary", data=income, ci=None)
# 绘制新预测点（红色圆点）
plt.scatter(x_new, y_pred, color='red', s=80, zorder=5, label=f'predict (x={x_new})')
plt.legend()
plt.show()

# 导入第三方模块
import statsmodels.api as sm

# 利用收入数据集，构建回归模型
fit = sm.formula.ols("Salary ~ YearsExperience", data=income).fit()
# 返回模型的参数值
print("=== statsmodels 计算结果 ===")
print("回归参数：")
print(fit.params)
print("\n模型摘要：")
print(fit.summary())

# 验证：预测 x=0.8452 时的值
import pandas as pd
x_new = pd.DataFrame({'YearsExperience': [0.8452]})
y_pred_sm = fit.predict(x_new)
print(f"\n当 x=0.8452 时，y 的预测值为：{y_pred_sm.values[0]}")

# 计算 MSE
pred_all = fit.predict(income)
mse_sm = ((income.Salary - pred_all) ** 2).mean()
print(f"均方误差(MSE)：{mse_sm}")