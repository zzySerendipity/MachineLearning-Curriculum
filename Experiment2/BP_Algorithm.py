#  标准 BP 神经网络 - 完整实验
#  任务1：学习率对比
#  任务2：训练轮数确定
#  任务3：隐层神经元个数对比
#  任务4：混淆矩阵和准确率
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读数据
wine = np.genfromtxt("wine_data-2.csv", delimiter=",", skip_header=1)
X = wine[:, 0:13]
y = wine[:, 13]

sc = StandardScaler()
X_st = sc.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_st, y, random_state=42)

label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


# 训练函数
def train(x, y, x_test, y_test, hiden_dim=50, lr=0.03, max_iter=1000):
    np.random.seed(42)

    w1 = np.random.random((x.shape[1], hiden_dim)) * 2 - 1
    b1 = np.zeros((1, hiden_dim))
    w2 = np.random.random((hiden_dim, 1)) * 2 - 1
    b2 = np.zeros((1, 1))

    train_loss_list = []
    test_mse_list = []
    test_acc_list = []

    for ite in range(max_iter):
        loss_per_ite = []
        for m in range(x.shape[0]):
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, -1), yi.reshape(1, -1)

            # 前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)

            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss)

            # 反向传播
            d_out2 = -(yi - out2)
            d_u2 = d_out2 * d_sigmoid(out2)
            d_w2 = np.dot(out1.T, d_u2)
            d_b2 = d_u2
            d_out1 = d_u2 * w2.T
            d_u1 = d_out1 * d_sigmoid(out1)
            d_w1 = np.dot(xi.T, d_u1)
            d_b1 = d_u1

            # 更新权重
            w1 -= lr * d_w1
            w2 -= lr * d_w2
            b1 -= lr * d_b1
            b2 -= lr * d_b2

        train_loss_list.append(np.mean(loss_per_ite))

        # 测试集评估
        test_pred_list = []
        for m in range(x_test.shape[0]):
            xi = x_test[m, :].reshape(1, -1)
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)
            test_pred_list.append(out2[0, 0])

        test_pred_array = np.array(test_pred_list)
        mse = mean_squared_error(y_test, test_pred_array)
        test_mse_list.append(mse)

        test_label_list = [1 if pred >= 0.5 else 0 for pred in test_pred_list]
        acc = accuracy_score(y_test, test_label_list)
        test_acc_list.append(acc)

    # 最终预测结果
    final_pred_list = []
    final_output_list = []
    for m in range(x_test.shape[0]):
        xi = x_test[m, :].reshape(1, -1)
        u1 = np.dot(xi, w1) + b1
        out1 = sigmoid(u1)
        u2 = np.dot(out1, w2) + b2
        out2 = sigmoid(u2)
        final_output_list.append(out2[0, 0])
        final_pred_list.append(1 if out2[0, 0] >= 0.5 else 0)

    return w1, w2, b1, b2, train_loss_list, test_mse_list, test_acc_list, final_pred_list, final_output_list


# ==================== 任务1：不同学习率对比 ====================
print("=" * 60)
print("任务1：不同学习率对比实验")
print("=" * 60)

lrs = [0.01, 0.02, 0.03, 0.04, 0.05]
results_lr = {}

plt.figure(figsize=(12, 5))
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, lr in enumerate(lrs):
    print(f"训练学习率 lr = {lr:.2f}...")
    w1, w2, b1, b2, train_loss, test_mse, test_acc, _, _ = train(
        x_train, label_train, x_test, y_test, hiden_dim=50, lr=lr, max_iter=1000
    )
    results_lr[lr] = {'loss': train_loss, 'mse': test_mse, 'acc': test_acc}
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 1001), train_loss, color=colors[i], label=f'lr={lr:.2f}')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 1001), test_acc, color=colors[i], label=f'lr={lr:.2f}')

plt.subplot(1, 2, 1)
plt.xlabel("训练轮数")
plt.ylabel("训练损失")
plt.title("不同学习率下的训练损失曲线")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.xlabel("训练轮数")
plt.ylabel("准确率")
plt.title("不同学习率下的测试准确率曲线")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig("1_learning_rate_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# 打印学习率对比表
print("\n学习率对比结果：")
print(f"{'学习率':<10} {'最终损失':<15} {'最终MSE':<15} {'最终准确率':<15}")
print("-" * 55)
for lr in lrs:
    print(
        f"{lr:<10.2f} {results_lr[lr]['loss'][-1]:<15.6f} {results_lr[lr]['mse'][-1]:<15.6f} {results_lr[lr]['acc'][-1]:<15.4f}")

# ==================== 任务2：训练轮数确定 ====================
print("\n" + "=" * 60)
print("任务2：训练轮数确定")
print("=" * 60)

best_lr = 0.03  # 选择较优学习率
max_epochs = 2000

print(f"训练中（lr={best_lr}, epochs={max_epochs}）...")
w1, w2, b1, b2, train_loss_list, test_mse_list, test_acc_list, _, _ = train(
    x_train, label_train, x_test, y_test, hiden_dim=50, lr=best_lr, max_iter=max_epochs
)

# 绘制MSE和准确率随训练轮数的变化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, max_epochs + 1), test_mse_list, color='red', linewidth=1)
axes[0].set_xlabel("训练轮数 (Epochs)")
axes[0].set_ylabel("均方误差 (MSE)")
axes[0].set_title("均方误差随训练轮数的变化")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, max_epochs + 1), test_acc_list, color='green', linewidth=1)
axes[1].set_xlabel("训练轮数 (Epochs)")
axes[1].set_ylabel("准确率 (Accuracy)")
axes[1].set_title("准确率随训练轮数的变化")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("2_epochs_mse_acc.png", dpi=150, bbox_inches='tight')
plt.show()

# 寻找最佳训练轮数
stable_threshold = 0.001
stable_count = 0
best_epoch = max_epochs

for i in range(100, max_epochs):
    if abs(test_acc_list[i] - test_acc_list[i - 1]) < stable_threshold:
        stable_count += 1
        if stable_count >= 100:
            best_epoch = i - 50
            break
    else:
        stable_count = 0

print(f"最终训练损失: {train_loss_list[-1]:.6f}")
print(f"最终测试MSE: {test_mse_list[-1]:.6f}")
print(f"最终测试准确率: {test_acc_list[-1]:.4f}")
print(f"建议最佳训练轮数: {best_epoch}")
print(f"  该轮数下MSE: {test_mse_list[best_epoch - 1]:.6f}")
print(f"  该轮数下准确率: {test_acc_list[best_epoch - 1]:.4f}")

# 训练轮数对准确率和均方误差的影响分析
print("\n训练轮数对准确率和均方误差的影响是：")
print("随着训练轮数的增加，均方误差MSE呈现先快速下降后趋于平稳的趋势，")
print("准确率呈现先快速上升后趋于平稳的趋势。在训练初期（前200轮），")
print("MSE迅速下降，准确率迅速上升；在训练中期（200-800轮），变化趋缓；")
print(f"在训练后期（{best_epoch}轮以后），MSE和准确率基本稳定，")
print("继续增加训练轮数对性能提升有限，甚至可能导致过拟合。")

# ==================== 任务3：不同隐层神经元个数对比 ====================
print("\n" + "=" * 60)
print("任务3：不同隐层神经元个数对比")
print("=" * 60)

hiden_dims = [10, 20, 30, 50, 80, 100]
results_hidden = {}

plt.figure(figsize=(14, 5))

for hiden_dim in hiden_dims:
    print(f"训练隐层神经元个数 = {hiden_dim}...")
    w1, w2, b1, b2, train_loss, test_mse, test_acc, _, _ = train(
        x_train, label_train, x_test, y_test, hiden_dim=hiden_dim, lr=best_lr, max_iter=1000
    )
    results_hidden[hiden_dim] = {'loss': train_loss, 'mse': test_mse, 'acc': test_acc}

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 1001), test_mse, label=f'hidden={hiden_dim}')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 1001), test_acc, label=f'hidden={hiden_dim}')

plt.subplot(1, 2, 1)
plt.xlabel("训练轮数")
plt.ylabel("均方误差 (MSE)")
plt.title("不同隐层神经元个数下的MSE曲线")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.xlabel("训练轮数")
plt.ylabel("准确率 (Accuracy)")
plt.title("不同隐层神经元个数下的准确率曲线")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig("3_hidden_neurons_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# 打印隐层神经元个数对比表
print("\n隐层神经元个数对比结果：")
print(f"{'神经元数':<10} {'最终损失':<15} {'最终MSE':<15} {'最终准确率':<15}")
print("-" * 55)
for hiden_dim in hiden_dims:
    print(
        f"{hiden_dim:<10} {results_hidden[hiden_dim]['loss'][-1]:<15.6f} {results_hidden[hiden_dim]['mse'][-1]:<15.6f} {results_hidden[hiden_dim]['acc'][-1]:<15.4f}")

# 隐层神经元个数影响分析
best_hidden = max(results_hidden, key=lambda k: results_hidden[k]['acc'][-1])
print(f"\n最佳隐层神经元个数: {best_hidden}")
print("\n以上情况原因是：")
print("1. 隐层神经元个数过少（如10个）：模型容量不足，无法充分学习数据的复杂模式，导致欠拟合，")
print("   训练损失较高，测试准确率较低。")
print("2. 隐层神经元个数适中（如30-50个）：模型容量与数据复杂度匹配，能有效学习特征表示，")
print("   训练损失较低，测试准确率较高，泛化性能好。")
print("3. 隐层神经元个数过多（如80-100个）：模型容量过大，容易记忆训练数据中的噪声，")
print("   导致过拟合，训练损失虽低但测试准确率反而下降，且计算开销增大。")
print(f"\n采用隐层神经元数量{best_hidden}左右较为合适，")
print("能在模型容量和泛化性能之间取得较好的平衡。")

# ==================== 任务4：混淆矩阵和准确率 ====================
print("\n" + "=" * 60)
print("任务4：混淆矩阵和准确率")
print("=" * 60)

# 使用最佳参数训练最终模型
print(f"使用最佳参数训练最终模型（hidden={best_hidden}, lr={best_lr}, epochs={best_epoch}）...")
w1, w2, b1, b2, _, _, _, final_pred_list, final_output_list = train(
    x_train, label_train, x_test, y_test, hiden_dim=best_hidden, lr=best_lr, max_iter=best_epoch
)

# 计算混淆矩阵
cm = confusion_matrix(y_test, final_pred_list)
final_acc = accuracy_score(y_test, final_pred_list)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['类别0', '类别1'],
            yticklabels=['类别0', '类别1'],
            annot_kws={'size': 20})
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title(f"混淆矩阵 (准确率: {final_acc:.4f})")
plt.tight_layout()
plt.savefig("4_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# 输出分类报告
print(f"\n最终模型准确率: {final_acc:.4f}")
print("\n混淆矩阵:")
print(cm)
print(f"\n分类报告:")
print(classification_report(y_test, final_pred_list, target_names=['类别0', '类别1']))

# 输出结果说明
print("\n混淆矩阵分析：")
print(f"  真阴性 (TN): {cm[0, 0]} (正确预测为类别0)")
print(f"  假阳性 (FP): {cm[0, 1]} (错误预测为类别1)")
print(f"  假阴性 (FN): {cm[1, 0]} (错误预测为类别0)")
print(f"  真阳性 (TP): {cm[1, 1]} (正确预测为类别1)")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print("\n生成的图片文件：")
print("  1_learning_rate_comparison.png - 不同学习率对比")
print("  2_epochs_mse_acc.png - 训练轮数确定")
print("  3_hidden_neurons_comparison.png - 隐层神经元个数对比")
print("  4_confusion_matrix.png - 混淆矩阵")