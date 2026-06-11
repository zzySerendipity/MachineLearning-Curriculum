#  标准 BP
#  Wine,数据预处理
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读数据
wine = np.genfromtxt("wine_data-2.csv", delimiter=",", skip_header=1)  # 二分类任务
X = wine[:, 0:13]
y = wine[:, 13]

sc = StandardScaler()
X_st = sc.fit_transform(X)  # 对样本的各属性值进行标准化

x_train, x_test, y_train, y_test = train_test_split(X_st, y, random_state=42)  # 固定随机种子，保证每次划分一致

label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数的导数
def d_sigmoid(x):
    return x * (1 - x)


# 训练函数（加入随机种子保证可比性）
def train(x, y, outputs_dim=1, lr=0.05, max_iter=1000):
    hiden_dim = 50  # 隐层神经元个数

    # 固定随机种子，保证不同学习率下初始权重相同
    np.random.seed(42)

    # 定义权重
    w1 = np.random.random((x.shape[1], hiden_dim)) * 2 - 1  # （13，50）
    b1 = np.zeros((1, hiden_dim))  # （1，50）
    w2 = np.random.random((hiden_dim, outputs_dim)) * 2 - 1  # （50，1）
    b2 = np.zeros((outputs_dim, 1))  # 1 X 1

    losslist = []  # 损失列表

    for ite in range(max_iter):
        loss_per_ite = []
        for m in range(x.shape[0]):  # 遍历样本
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])

            ##前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)  # 隐含层的输出  1 X 50

            u2 = np.dot(out1, w2) + b2  # (1,50) X (50,1) = (1,1)
            out2 = sigmoid(u2)  # 输出(激活)层的输出,(1,1)

            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss)

            ##反向传播（标准BP）
            d_out2 = -(yi - out2)  # (1,1)
            d_u2 = d_out2 * d_sigmoid(out2)  # gj

            d_w2 = np.dot(np.transpose(out1), d_u2)  # delta(whj)
            d_b2 = d_u2  # delta(thlrj)

            d_out1 = d_u2 * w2.T  # 修改：保证维度匹配
            d_u1 = d_out1 * d_sigmoid(out1)  # eh

            d_w1 = np.dot(np.transpose(xi), d_u1)  # delta(vih)
            d_b1 = d_u1  # delta(rh)

            ##更新权重
            w1 = w1 - lr * d_w1
            w2 = w2 - lr * d_w2
            b1 = b1 - lr * d_b1
            b2 = b2 - lr * d_b2

        losslist.append(np.mean(loss_per_ite))

    return w1, w2, b1, b2, losslist


# 定义不同的学习率
learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05]

# 存储结果
results = {}

# 训练并记录
for lr in learning_rates:
    print(f"正在训练学习率 lr = {lr:.2f}...")
    w1, w2, b1, b2, losslist = train(x_train, label_train, outputs_dim=1, lr=lr, max_iter=1000)
    results[lr] = {
        'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2,
        'losslist': losslist
    }

    # 测试
    test_label_list = []
    for m in range(x_test.shape[0]):
        xi = x_test[m, :].reshape(1, -1)
        u1 = np.dot(xi, w1) + b1
        out1 = sigmoid(u1)
        u2 = np.dot(out1, w2) + b2
        out2 = sigmoid(u2)
        if out2 >= 0.5:
            test_label_list.append(1)
        else:
            test_label_list.append(0)

    re = sum(1 for i in range(len(y_test)) if test_label_list[i] == y_test[i])
    acc = re / len(y_test)
    results[lr]['acc'] = acc
    print(f"  测试精度 acc = {acc:.4f}")

# 绘制不同学习率的损失函数曲线
plt.figure(figsize=(12, 6))

colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, lr in enumerate(learning_rates):
    plt.plot([j + 1 for j in range(1000)], results[lr]['losslist'],
             color=colors[i], label=f'lr = {lr:.2f} (acc={results[lr]["acc"]:.3f})')

plt.legend()
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.title("不同学习率下的损失函数曲线对比")
plt.grid(True, alpha=0.3)
plt.show()

# 输出精度对比表
print("\n" + "=" * 50)
print("不同学习率测试精度对比")
print("=" * 50)
print(f"{'学习率':<10} {'测试精度':<15}")
print("-" * 25)
for lr in learning_rates:
    print(f"{lr:<10.2f} {results[lr]['acc']:<15.4f}")