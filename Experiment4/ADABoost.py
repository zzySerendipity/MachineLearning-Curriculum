# 导入库
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, \
    f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 将数据转为数字,分割特征和标签
def loadDataSet(filename):
    dim = len(open(filename).readline().split('\t'))
    data = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        LineArr = []
        curline = line.strip().split('\t')
        for i in range(dim - 1):
            LineArr.append(float(curline[i]))
        data.append(LineArr)
        label.append(float(curline[-1]))
    return data, label


# 加载数据集
X_train,y_train = loadDataSet("horseColicTraining.txt")
X_test,y_test = loadDataSet("horseColicTest.txt")
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

# ==================== (1) 创建弱分类器决策树 ====================
weak_clf = DecisionTreeClassifier(max_depth=1)


# AdaBoost算法
def My_Adaboost(X_train, Y_train, X_test, Y_test, M=20, weak_clf=DecisionTreeClassifier(max_depth=1)):
    n_train, n_test = len(X_train), len(X_test)
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    for i in range(M):
        weak_clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f" % (i + 1, 1 - sum(miss) / n_train))

        err_m = np.dot(w, miss)
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        miss2 = [x if x == 1 else -1 for x in miss]
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1
    print("Accuracy of train is", sum(pred_train == y_train) / n_train)
    print("Accuracy of test is", sum(pred_test == y_test) / n_test)

    return pred_train, pred_test


# ==================== (2) 预测测试集标签 ====================
y_train_pred, y_test_pred = My_Adaboost(X_train, np.array(y_train), X_test, np.array(y_test), M=20, weak_clf=weak_clf)

# ==================== (3) 生成混淆矩阵 ====================
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix of Label is \n", cm)

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0(死亡)', '1(存活)'])
disp.plot()
plt.title("混淆矩阵")
plt.savefig("1_Confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# 计算死亡率
print("Rate of death is ", np.sum(y_test_pred == 0) / len(y_test))

# ==================== (4) 输出精确度 ====================
print("Precision of Label is", precision_score(y_test, y_test_pred))

# 召回率
print("Recall of Label is", recall_score(y_test, y_test_pred))

# F1度量
print("F1 of Label is", f1_score(y_test, y_test_pred))

# ==================== (5) 与sklearn的AdaBoostClassifier比较 ====================
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=20,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nsklearn AdaBoost 分类报告:")
print(classification_report(y_test, y_pred, target_names=['死亡(0)', '存活(1)']))


# ==================== 绘制P-R曲线 ====================
def PR_curve(y, pred):
    pos = np.sum(y == 1)
    pred_sort = np.sort(pred)[::-1]
    index = np.argsort(pred)[::-1]
    y_sort = y[index]

    Pre = []
    Rec = []
    for i, item in enumerate(pred_sort):
        if i == 0:
            Pre.append(1.0)  # 修正：起始点precision应为1
            Rec.append(0.0)
        else:
            Pre.append(np.sum((y_sort[:i] == 1)) / i)
            Rec.append(np.sum((y_sort[:i] == 1)) / pos)

    # 添加终点
    Rec.append(1.0)
    Pre.append(0.0)

    plt.figure(figsize=(8, 6))
    plt.plot(Rec, Pre, 'b-', lw=2, label='手动AdaBoost')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('P-R曲线')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("1_PRCurve.png", dpi=150, bbox_inches='tight')
    plt.show()


# 绘制P-R曲线（手动实现）
PR_curve(np.array(y_test), y_test_pred)

# ==================== 使用sklearn绘制P-R曲线对比 ====================
from sklearn.metrics import precision_recall_curve

# 手动AdaBoost的P-R曲线
precision_manual, recall_manual, _ = precision_recall_curve(y_test, y_test_pred, pos_label=1)

# sklearn AdaBoost的P-R曲线
precision_sk, recall_sk, _ = precision_recall_curve(y_test, y_pred, pos_label=1)

plt.figure(figsize=(8, 6))
plt.plot(recall_manual, precision_manual, 'b-', lw=2, label='手动AdaBoost')
plt.plot(recall_sk, precision_sk, 'r--', lw=2, label='sklearn AdaBoost')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('P-R曲线对比')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("1_PRCurve_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n分析完成！")