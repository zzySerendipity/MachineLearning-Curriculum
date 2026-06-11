# 生成一张sklearn决策树以及一张混淆矩阵图片
# 1）数据预览
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import ensemble, metrics, model_selection, tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读入数据
fr = open("glass-lenses.txt")
lenses = [inst.strip().split("\t") for inst in fr.readlines()]
lensesLabels = ["age", "prescript", "astigmatic", "tearRate", "type"]
lens = pd.DataFrame.from_records(lenses, columns=lensesLabels)
print("数据预览:")
print(lens)

# 2）数据预处理
dummy = pd.get_dummies(lens[["age", "prescript", "astigmatic", "tearRate"]])
lens = pd.concat([lens, dummy], axis=1)
lens.drop(["age", "prescript", "astigmatic", "tearRate"], inplace=True, axis=1)
print("\n哑变量处理后:")
print(lens.head())

# 3）拆分训练集和测试集（25%测试集）
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    lens.loc[:, "age_pre":"tearRate_reduced"],
    lens.type,
    test_size=0.25,
    random_state=1234,
)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4）构建分类决策树（ID3算法）
dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=1234)
dt.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(14, 8))
tree.plot_tree(dt, feature_names=X_train.columns,
               class_names=dt.classes_, filled=True, rounded=True,
               fontsize=10)
plt.title("ID3 决策树 (criterion='entropy')")
plt.tight_layout()
plt.savefig("decision_tree_sklearn.png", dpi=150, bbox_inches='tight')
plt.show()

# 5）输出预测准确率
y_train_pred = dt.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\n训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")

y_test_pred = dt.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

print("\n分类报告:")
print(classification_report(y_test, y_test_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
print("混淆矩阵:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dt.classes_, yticklabels=dt.classes_)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title(f"混淆矩阵 (测试准确率: {test_acc:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix_sklearn.png", dpi=150, bbox_inches='tight')
plt.show()