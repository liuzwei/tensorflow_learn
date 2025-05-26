import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras
print(keras.__version__)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


# 下载IMDB数据集
train_data, validation_data, test_data = tfds.load(
    "imdb_reviews",
    split=["train[:60%]", "train[60%:]", "test"],
    as_supervised=True,
)

# 获取前10个样本，并打印数据
# train_examples, train_labels = next(iter(train_data.batch(10)))
# print("Training examples: ", train_examples.numpy())
# print("Training labels: ", train_labels.numpy())

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

# 用函数式API构建模型
inputs = tf.keras.Input(shape=(), dtype=tf.string)
x = hub_layer(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)
# 评估模型
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))