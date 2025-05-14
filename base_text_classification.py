import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


print(tf.__version__)


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset_dir = 'aclImdb_v1/aclImdb'
if not os.path.exists(dataset_dir):
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')
else:
    print(f"Dataset already downloaded at {dataset_dir}")

print(os.listdir(dataset_dir))

# 训练数据集目录
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# # 数据样本
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#   print(f.read())

# 要删除的目录， 如果目录存在，则删除
remove_dir = os.path.join(train_dir, 'unsup')
if os.path.exists(remove_dir):
    shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

# 使用工具text_dataset_from_directory将训练数据集生成带标签的数据
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', # 训练集
    seed=seed)
# 标签就是train目录下的目录neg和pos
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', # 验证集
    seed=seed)

# 测试数据集
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/test', 
    batch_size=batch_size)


def custom_standardization(input_data):
    """
    自定义标准化函数：
    1. 将输入文本转换为小写。
    2. 移除 HTML 标签（例如 <br />）。
    3. 去除所有标点符号。

    参数:
    input_data (tf.Tensor): 输入的文本数据。

    返回:
    tf.Tensor: 标准化后的文本数据。
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    """
    将输入文本向量化，并返回向量化后的文本及其标签。

    参数:
        text (tf.Tensor 或 str): 要向量化的输入文本。文本会被扩展为额外的维度。
        label (Any): 与输入文本关联的标签。

    返回:
        tuple: 包含向量化后的文本（作为张量）和对应标签的元组。
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# 在数据集中检索一批数据
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# 整数和词的对应关系
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# 查看 raw_train_ds 的数据格式
# for text, label in raw_train_ds.take(1):
#     print("Raw text:", text.numpy()[0])
#     print("Raw label:", label.numpy()[0])

# # 查看 train_ds 的数据格式
# for vectorized_text, label in train_ds.take(1):
#     print("Vectorized text:", vectorized_text.numpy()[0])
#     print("Label:", label.numpy()[0])


# 配置数据集以提高性能
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# 构建模型
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

# 损失函数和优化器
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0)])  # 将 metrics 包装为列表
# 训练模型
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# 评估模型
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# 训练过程可视化
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# 导出模型
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# 保存模型
# export_model.save('text_classification_model.h5')

# 使用新数据进行推断
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

# 将输入转换为 TensorFlow 张量
examples_tensor = tf.convert_to_tensor(examples)
predict_result = export_model.predict(examples_tensor)
print(predict_result)
# 预测类别
predicted_labels = tf.where(predict_result > 0.5, 1, 0)
print(predicted_labels.numpy().flatten())