#  基本图像分类  对服装图像进行分类

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # 关闭OneDNN优化

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 查看当前TensorFlow版本 
print(tf.__version__)

# 1. 导入Fashion MNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 浏览数据
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)

# 2. 数据预处理

# 数据查看
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 2.1 数据归一化  将像素值缩放到0到1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集的前25张图像
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 3. 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将28x28的图像展平为784维的向量
    tf.keras.layers.Dense(128, activation='relu'),  # 隐藏层，128个神经元，使用ReLU激活函数
    tf.keras.layers.Dense(10) # 输出层，10个神经元，对应10个类别
])

# 4. 编译模型
model.compile(
    optimizer='adam',  # 优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数
    metrics=['accuracy']  # 评估指标
)

# 5. 训练模型
model.fit(train_images, train_labels, epochs=20)  # 训练模型，训练5个周期

# 6. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # 在测试集上评估模型
print('\nTest accuracy:', test_acc)  # 打印测试集准确率

# 7. 进行预测
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()  # 添加Softmax层，将输出转换为概率分布
])
predictions = probability_model.predict(test_images)  # 对测试集进行预测

# print(predictions[0])  # 打印第一个测试样本的预测结果

# print(f"置信度最高的是：{np.argmax(predictions[0])}, 标签是：{test_labels[0]}")  # 打印置信度最高的类别和真实标签

# 8. 可视化预测结果
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 画出第i张图像的预测结果
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# 9. 使用训练好的模型
# 9.1 使用模型进行预测
img = test_images[0]  # 测试图像
img = (np.expand_dims(img,0))  # 扩展维度
predictions_single = probability_model.predict(img)  # 预测
print(predictions_single)  # 打印预测结果

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))  # 打印预测类别

# 10. 保存模型
model.save('fashion_mnist_model.h5')  # 保存模型