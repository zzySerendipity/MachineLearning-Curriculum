import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
df = pd.read_csv('HR-Employee-Attrition.csv')
print("数据预览:")
print(df.head())

# ==================== 数据预处理 ====================
print("\n各列唯一值数量(最少10列):")
print(df.nunique().nsmallest(10))

# 删除无效列值
df.drop(['StandardHours', 'EmployeeCount', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

print(f"\n缺失值统计: {df.isnull().sum().sum()}")
print(f"重复行数: {df[df.duplicated()].shape[0]}")

# ===== 编码所有非数字列（修正版）=====
# 先获取所有非数字列
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n需要编码的列: {object_cols}")

# 逐一编码
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("\n编码后数据预览:")
print(df.head())
print(f"\n所有列数据类型:\n{df.dtypes.value_counts()}")

# ==================== 查看各列相关性 ====================
corr = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            cmap='coolwarm', center=0, annot=False)
plt.title("特征相关性热力图")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# 删除高相关性特征
df.drop(['JobLevel', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsWithCurrManager',
         'PercentSalaryHike', 'StockOptionLevel'], axis=1, inplace=True)

# ==================== 特征提取 ====================
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# 标准化数据
sc = StandardScaler()
X = sc.fit_transform(X)
mean = np.mean(X, axis=0)
print('均值:')
print(mean)
standard_deviation = np.std(X, axis=0)
print('标准差:')
print(standard_deviation)

# ==================== (1) 划分数据集（25%测试集）====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集流失比例: {y_train.mean():.4f}")
print(f"测试集流失比例: {y_test.mean():.4f}")

# ==================== 训练SVM模型 ====================
svc = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# ==================== (2) 输出Accuracy Score ====================
acc = accuracy_score(y_test, y_pred)
print(f'\nAccuracy Score: {acc:.4f} ({acc*100:.2f}%)')

# ==================== 输出支持向量 ====================
support_vectors = svc.support_vectors_
print(f'支持向量个数: {support_vectors.shape[0]}')
print(f'支持向量 (前5个):\n{support_vectors[:5]}')

# ==================== (3) 获取混淆矩阵 ====================
cnf_matrix = confusion_matrix(y_test, y_pred)
print(f'\n混淆矩阵:\n{cnf_matrix}')

# 绘制混淆矩阵
class_names = ['未流失(0)', '流失(1)']
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cnf_matrix, index=class_names, columns=class_names),
            annot=True, cmap="YlGnBu", fmt='g', ax=ax)
ax.set_title('混淆矩阵', y=1.05)
ax.set_ylabel('真实标签')
ax.set_xlabel('预测标签')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 准确度、精确度、召回率和F1 ====================
print(f"\n{'='*50}")
print(f"模型评估指标")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

# ==================== (4) 输出F1_score ====================
f1 = f1_score(y_test, y_pred)
print(f"F1_score:  {f1:.4f}")

# ==================== (5) 绘制P-R曲线 ====================
y_scores = svc.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='P-R曲线')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('P-R曲线')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 绘制ROC曲线 ====================
fpr, tpr, threshold = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
plt.plot(fpr, tpr, color='black', lw=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.text(0.5, 0.3, f'ROC curve (AUC = {roc_auc:.2f})', fontsize=12)
plt.xlabel('1 - 特异性 (False Positive Rate)')
plt.ylabel('敏感性 (True Positive Rate)')
plt.title('ROC曲线')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*50}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"{'='*50}")
print(f"\n分析完成！共生成4张图片:")
print("  correlation_heatmap.png - 特征相关性热力图")
print("  confusion_matrix.png - 混淆矩阵")
print("  pr_curve.png - P-R曲线")
print("  roc_curve.png - ROC曲线")