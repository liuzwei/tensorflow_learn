import os
import re
import string
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers


# 数据集目录
dataset_dir = 'stack_overflow_datasets'
# 查看一下数据集下的目录文件信息
print(os.listdir(dataset_dir))

batch_size = 16
seed = 42

# 使用工具text_dataset_from_directory将训练数据集生成带标签的数据
train_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', # 训练集
    seed=seed)

# 循环打印所有的标签
for i, class_name in enumerate(train_ds.class_names):
    print(f"Label {i} corresponds to {class_name}")

# 验证数据集
val_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', # 验证集
    seed=seed)
# 测试数据集
test_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'test'),
    batch_size=batch_size)

# 数据标准化
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


max_features = 20000

# 将文本数据转换为整数序列的文本向量化层
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,  # 词汇表大小
    output_mode='int',  # 输出模式为整数
    output_sequence_length=300,  # 输出序列长度
    standardize=None,  # 不进行标准化
    split='whitespace'  # 按空格分割文本
)

# 获取训练数据的文本
train_texts = train_ds.map(lambda x, y: x)
# 适配文本向量化层
vectorize_layer.adapt(train_texts)

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
# text_batch, label_batch = next(iter(train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# 文本向量化
# print("Vectorized review:", vectorize_text(first_review, first_label)[0].numpy())

# 将训练集、验证集和测试集转换为向量化的格式
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

# 配置数据集以提高性能
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 构建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=max_features+1, output_dim=64, mask_zero=True),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.0005)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(4, activation='softmax')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features+1, output_dim=64, mask_zero=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()

# 损失函数和优化器
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# 训练模型
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# 评估模型
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
