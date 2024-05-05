import matplotlib.pyplot as plt

# 数据集
datasets = ['MR', 'Subj', 'TREC', 'Reuters-Singel', 'Reuters-Full', 'Reuters-Multi']

# 每个数据集对应的四个模型的准确率
accuracies = {
    'MR': [76.9, 75.4,76.6,  78.1],
    'Subj': [ 88.9, 88.8,89.4, 90.8],
    'TREC': [ 86.8, 86.0, 83.8,88.0],
    'Reuters-Singel': [ 96.6, 96.0,96.6, 97.5],
    'Reuters-Full': [ 84.0, 83.5,85.0, 87.5],
    'Reuters-Multi': [ 27.1, 30.5, 60.9,68.4]
}

# 创建一个新的图表
plt.figure(figsize=(12, 8))

# 绘制折线图
index = range(len(datasets))
model1_accs = [accuracies[dataset][0] for dataset in datasets]
model2_accs = [accuracies[dataset][1] for dataset in datasets]
model3_accs = [accuracies[dataset][2] for dataset in datasets]
model4_accs = [accuracies[dataset][3] for dataset in datasets]

plt.plot(index, model1_accs, marker='o', linestyle='-', label='KimCNN', color='#2ca02c')
plt.plot(index, model2_accs, marker='s', linestyle='-', label='C-LSTM', color='#ff7f0e')
plt.plot(index, model3_accs, marker='^', linestyle='-', label='Capsule-A', color='#1f77b4')
plt.plot(index, model4_accs, marker='D', linestyle='-', label='MBConv-CapsNet', color='#d62728')

# 添加标题和标签
plt.title('Comparison of Model Accuracy Across Datasets', fontsize=12)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(index, datasets, fontsize=12)
plt.legend()

# 显示准确率数值
for i, acc in enumerate(model1_accs):
    plt.text(i, acc + 0.5, str(acc), ha='center', va='bottom', fontsize=8)

for i, acc in enumerate(model2_accs):
    plt.text(i, acc + 0.5, str(acc), ha='center', va='bottom', fontsize=8)

for i, acc in enumerate(model3_accs):
    plt.text(i, acc + 0.5, str(acc), ha='center', va='bottom', fontsize=8)

for i, acc in enumerate(model4_accs):
    plt.text(i, acc + 0.5, str(acc), ha='center', va='bottom', fontsize=8)

# 显示图表
plt.tight_layout()
plt.savefig('pic.png')
plt.show()
