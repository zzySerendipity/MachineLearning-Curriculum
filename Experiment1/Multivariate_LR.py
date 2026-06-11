# Predict to Profit.xlsx
# 多元线性回归 - 预测利润
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 添加这行，解决上标符号问题

# 读入数据（直接读取 Excel）
data = pd.read_excel("Predict to Profit.xlsx")  # 假设文件在代码同目录下
print(f"数据集大小: {data.shape}")
print(f"数据前5行:")
print(data.head())
print(f"描述性统计:")
print(data.describe())

# ==================== 数据预处理 ====================
X = data[['RD_Spend', 'Administration', 'Marketing_Spend', 'State']]
y = data['Profit']

# 数值特征标准化，类别特征独热编码
numeric_features = ['RD_Spend', 'Administration', 'Marketing_Spend']
categorical_features = ['State']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# 获取处理后的特征名
cat_encoder = preprocessor.named_transformers_['cat']
state_categories = cat_encoder.get_feature_names_out(['State'])
feature_names = numeric_features + list(state_categories)

print(f"处理后的特征名: {feature_names}")

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ==================== 建立多元线性回归模型 ====================
model = LinearRegression()
model.fit(X_train, y_train)

# ==================== 模型评估 ====================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)


print("\n模型评估结果")
print(f"{'指标':<15} {'训练集':<20} {'测试集':<20}")
print(f"{'R²':<15} {train_r2:<20.4f} {test_r2:<20.4f}")
print(f"{'MSE':<15} {train_mse:<20.2f} {test_mse:<20.2f}")
print(f"{'MAE':<15} {train_mae:<20.2f} {test_mae:<20.2f}")

# ==================== 回归系数分析 ====================
print("\n回归系数")
coef_df = pd.DataFrame({
    '特征': feature_names,
    '系数': model.coef_
})
coef_df['系数绝对值'] = coef_df['系数'].abs()
coef_df = coef_df.sort_values('系数绝对值', ascending=False)
print(coef_df.to_string(index=False))

print(f"截距 (b): {model.intercept_:.4f}")

# ==================== 特征重要性分析 ====================
print("\n特征重要性分析")
print("对利润影响最大的因素是：")
for i, row in coef_df.iterrows():
    if row['系数'] > 0:
        print(f"  {row['特征']}: 每增加1个标准差，利润增加 {row['系数']:.2f}")
    else:
        print(f"  {row['特征']}: 每增加1个标准差，利润减少 {abs(row['系数']):.2f}")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：预测值 vs 真实值
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.7, edgecolors='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel("真实值")
axes[0, 0].set_ylabel("预测值")
axes[0, 0].set_title(f"测试集: 预测值 vs 真实值 (R^2={test_r2:.4f})")
axes[0, 0].grid(True, alpha=0.3)

# 图2：残差图
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.7, edgecolors='k')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel("预测值")
axes[0, 1].set_ylabel("残差")
axes[0, 1].set_title("残差图")
axes[0, 1].grid(True, alpha=0.3)

# 图3：特征系数柱状图
colors = ['green' if c > 0 else 'red' for c in coef_df['系数']]
axes[1, 0].barh(range(len(coef_df)), coef_df['系数'], color=colors)
axes[1, 0].set_yticks(range(len(coef_df)))
axes[1, 0].set_yticklabels(coef_df['特征'])
axes[1, 0].set_xlabel("系数值")
axes[1, 0].set_title("特征系数（绿色=正相关，红色=负相关）")
axes[1, 0].grid(True, alpha=0.3)

# 图4：各地区平均利润对比
axes[1, 1].bar(data['State'].unique(),
               [data[data['State']==s]['Profit'].mean() for s in data['State'].unique()],
               color=['blue', 'orange', 'green'])
axes[1, 1].set_xlabel("销售市场")
axes[1, 1].set_ylabel("平均利润")
axes[1, 1].set_title("各销售市场平均利润对比")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("multiple_linear_regression.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n模型公式")
formula = f"Profit = {model.intercept_:.2f}"
for i, coef in enumerate(model.coef_):
    if coef >= 0:
        formula += f" + {coef:.2f} × {feature_names[i]}"
    else:
        formula += f" - {abs(coef):.2f} × {feature_names[i]}"
print(formula)