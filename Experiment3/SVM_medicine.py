from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入乳腺癌数据集
data = load_breast_cancer()
X = data['data']
y = data['target']
print(f"数据集形状: {X.shape}")
print(f"特征名称: {data['feature_names']}")
print(f"目标类别: {data['target_names']}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立支持向量机模型
clf1 = SVC(kernel='linear')  # 线性核函数
clf2 = SVC(kernel='rbf', C=10, gamma=0.0001)  # 采用高斯核函数
clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)

# ==================== 输出测试精度 ====================
# 线性核SVM
y_pred1 = clf1.predict(x_test)
acc1 = accuracy_score(y_test, y_pred1)

# 高斯核SVM
y_pred2 = clf2.predict(x_test)
acc2 = accuracy_score(y_test, y_pred2)

print("\n" + "=" * 50)
print("支持向量机模型测试精度")
print("=" * 50)
print(f"线性核函数 (Linear SVM): {acc1:.4f} ({acc1*100:.2f}%)")
print(f"高斯核函数 (RBF SVM):     {acc2:.4f} ({acc2*100:.2f}%)")

# 详细分类报告
print("\n" + "=" * 50)
print("线性核SVM 分类报告")
print("=" * 50)
print(classification_report(y_test, y_pred1, target_names=data['target_names']))

print("\n" + "=" * 50)
print("高斯核SVM 分类报告")
print("=" * 50)
print(classification_report(y_test, y_pred2, target_names=data['target_names']))

# 混淆矩阵可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=data['target_names'], yticklabels=data['target_names'])
axes[0].set_xlabel("预测标签")
axes[0].set_ylabel("真实标签")
axes[0].set_title(f"线性核SVM 混淆矩阵 (准确率: {acc1:.4f})")

cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=data['target_names'], yticklabels=data['target_names'])
axes[1].set_xlabel("预测标签")
axes[1].set_ylabel("真实标签")
axes[1].set_title(f"高斯核SVM 混淆矩阵 (准确率: {acc2:.4f})")

plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()