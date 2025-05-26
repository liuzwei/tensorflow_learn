import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
import string


def custom_standardization(input_data):
    # 这里需要和你训练模型时用的custom_standardization函数保持一致
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


def base_text_classification():
    # 1.加载模型
    model = tf.keras.models.load_model('model/text_classification_model.h5')
    # 2.加载数据
    examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
    ]
    # 将输入转换为 TensorFlow 张量
    examples_tensor = tf.convert_to_tensor(examples)
    predict_result = model.predict(examples_tensor)
    print(predict_result)


# 加载模型，预测图片
def base_image_classification(img_path):
    # 读取图片
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # 归一化

    # 数据查看
    plt.figure()
    plt.imshow(img, cmap='gray')  # 指定灰度模式
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # 1. 加载模型
    model = tf.keras.models.load_model('model/fashion_mnist_model.h5')

    # 2. 预处理图片
    img = np.expand_dims(img, axis=0)  # 扩展维度

    # 3. 预测
    predictions = model.predict(img)

    # 4. 返回预测结果
    return predictions
    

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main():
    # 图片分类
    # predictions = base_image_classification("images/trouser2.jpg")

    # # 打印预测结果
    # print(predictions)
    # # 预测类别
    # predicted_label = np.argmax(predictions[0])
    # print(f"Predicted label: {class_names[predicted_label]}")

    # -----------------------------------------
    pass

if __name__ == "__main__":
    main()
