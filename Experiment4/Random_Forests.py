# 导入库
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, auc,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 加载数据 ====================
data = pd.read_csv('Electrical Grid Data.csv', sep=',')
print("数据预览:")
print(data.head())
print(f"\n数据集大小: {data.shape}")

# ==================== 数据预处理（先编码非数值列）====================
# 编码 stabf 列
le = preprocessing.LabelEncoder()
data['stabf'] = le.fit_transform(data['stabf'])
print(f"stabf 编码: {le.classes_} -> {list(range(len(le.classes_)))}")

# 如果 stab 列也是非数值，同样编码
if data['stab'].dtype == 'object':
    data['stab'] = le.fit_transform(data['stab'])

# ==================== 查看各列相关性 ====================
corr = data.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            cmap='coolwarm', center=0, annot=True, fmt='.2f', linewidths=0.5)
plt.title("特征相关性热力图")
plt.tight_layout()
plt.savefig("2_Correlation.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 分离特征和标签 ====================
X = data.drop(['stabf'], axis=1)
y = data['stabf']

# 删除与标签高度相关的'stab'列（如果存在）
if 'stab' in X.columns:
    X = X.drop(['stab'], axis=1)
    print("已删除'stab'列")

print(f"\n特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"标签分布:\n{y.value_counts()}")

# 标准化
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== (1) 划分数据集（75%训练，25%测试）====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ==================== (2) 网格搜索 ====================
param_grid = {
    'n_estimators': [50, 100, 300],
    'max_depth': [3, 5, 7, 9]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

print("\n正在进行网格搜索...")
grid_search.fit(X_train, y_train)

print(f"\n{'='*60}")
print(f"网格搜索结果")
print(f"{'='*60}")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_scores = best_rf.predict_proba(X_test)[:, 1]

# ==================== (3) 混淆矩阵和评估指标 ====================
cm = confusion_matrix(y_test, y_pred)
print(f"\n{'='*60}")
print(f"混淆矩阵")
print(f"{'='*60}")
print(cm)

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("混淆矩阵")
plt.savefig("2_Confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== (4) 计算查准率、查全率、F1 ====================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*60}")
print(f"模型评估指标")
print(f"{'='*60}")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"F1分数 (F1-Score):  {f1:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==================== 绘制P-R曲线 ====================
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', lw=2, label='随机森林')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('P-R曲线')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("2_PR_Curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 绘制ROC曲线 ====================
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', lw=1, label='随机分类器')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('ROC曲线')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("2_ROC_Curve.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print(f"ROC AUC: {roc_auc:.4f}")

# ==================== 特征重要性 ====================
feature_importance = best_rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importance})
importance_df = importance_df.sort_values('重要性', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'], importance_df['重要性'], color='steelblue', edgecolor='black')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('随机森林特征重要性')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2_Feature_Importance.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n特征重要性排名:")
for i, row in importance_df.iterrows():
    print(f"  {row['特征']}: {row['重要性']:.4f}")

print(f"\n分析完成！")