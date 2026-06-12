# 手写字母识别 - SVM分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 导入数据 ====================
df = pd.read_csv('letterdata.csv')
print("数据预览:")
print(df.head(10))
print(f"\n数据集大小: {df.shape}")
print(f"特征数: {df.shape[1] - 1}")
print(f"类别数: {df['letter'].nunique()}")
print(f"\n各类别样本分布:\n{df['letter'].value_counts().sort_index()}")

# ==================== 数据预处理 ====================
# 检查缺失值
print(f"\n缺失值统计: {df.isnull().sum().sum()}")

# 分离特征和标签
X = df.drop('letter', axis=1)
y = df['letter']

# 标签编码（将字母A-Z转换为数字0-25）
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n编码后的类别: {le.classes_}")
print(f"各类别对应编码: {dict(zip(le.classes_, range(len(le.classes_))))}")

# 特征标准化
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# ==================== 划分数据集（25%测试集）====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ==================== 训练SVM模型 ====================
# 使用RBF核，针对多分类问题
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# ==================== 模型评估 ====================
acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"模型评估结果")
print(f"{'='*60}")
print(f"测试集准确率: {acc:.4f} ({acc*100:.2f}%)")

# 分类报告
print(f"\n分类报告:")
print(classification_report(y_test, y_pred,
                            target_names=le.classes_,
                            digits=4))

# ==================== 输出支持向量信息 ====================
print(f"{'='*60}")
print(f"支持向量信息")
print(f"{'='*60}")
print(f"支持向量个数: {svm_model.support_vectors_.shape[0]}")
print(f"支持向量占训练集比例: {svm_model.support_vectors_.shape[0]/X_train.shape[0]*100:.2f}%")

# ==================== 各类别准确率分析 ====================
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

# 各类别准确率
class_acc = {}
for i, letter in enumerate(le.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc[letter] = (y_pred[mask] == i).mean()

# 可视化各类别准确率
plt.figure(figsize=(14, 6))
letters = list(class_acc.keys())
accs = list(class_acc.values())
colors = ['green' if a >= acc else 'red' for a in accs]

plt.bar(letters, accs, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=acc, color='blue', linestyle='--', linewidth=1.5,
            label=f'平均准确率: {acc:.4f}')
plt.xlabel('字母类别')
plt.ylabel('准确率')
plt.title('各字母类别识别准确率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("letter_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 混淆矩阵（简化版）====================
# 由于26×26矩阵太大，只显示部分易混淆的字母
plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            linewidths=0.5, linecolor='gray')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'混淆矩阵 (准确率: {acc:.4f})')
plt.tight_layout()
plt.savefig("letter_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ==================== 找出最容易混淆的字母对 ====================
print(f"\n{'='*60}")
print(f"最容易混淆的字母对 (Top 10)")
print(f"{'='*60}")
confusion_pairs = []
for i in range(26):
    for j in range(26):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((le.classes_[i], le.classes_[j], cm[i, j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for i, (true, pred, count) in enumerate(confusion_pairs[:10]):
    print(f"  {i+1}. 真实='{true}' → 预测='{pred}', 误判次数: {count}")

# ==================== 准确率最高的字母 ====================
print(f"\n{'='*60}")
print(f"识别准确率最高和最低的字母")
print(f"{'='*60}")
sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
print(f"\n准确率最高 (Top 5):")
for letter, acc_val in sorted_acc[:5]:
    print(f"  '{letter}': {acc_val:.4f} ({acc_val*100:.2f}%)")

print(f"\n准确率最低 (Bottom 5):")
for letter, acc_val in sorted_acc[-5:]:
    print(f"  '{letter}': {acc_val:.4f} ({acc_val*100:.2f}%)")

print(f"\n分析完成！")