import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU

# 数据预处理函数
def preprocess(image_path, label):
    image_path = tf.cast(image_path, tf.string)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label

# 数据生成器函数
def data_generator(image_paths, labels, batch_size):
    path_labels = list(zip(image_paths, labels))
    while True:
        random.shuffle(path_labels)
        for i in range(0, len(path_labels), batch_size):
            batch_paths_labels = path_labels[i:i + batch_size]
            batch_images = []
            batch_labels = []
            for image_path, label in batch_paths_labels:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, [224, 224])
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.cast(image, tf.float32) / 255.0
                batch_images.append(image)
                batch_labels.append(label)
            yield tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_labels)

# 获取图片路径和标签
def get_image_paths_and_labels(data_dir, class_names):
    image_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith(('JPEG', 'jpg', 'png', 'jpeg')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)
    return image_paths, labels

# 加载数据
def load_data(data_dir):
    train_data_dir = os.path.join(data_dir, 'train')
    class_names = sorted(os.listdir(train_data_dir))
    train_image_paths, train_labels = get_image_paths_and_labels(train_data_dir, class_names)
    val_image_paths, val_labels = load_val_data(data_dir)
    return train_image_paths, train_labels, val_image_paths, val_labels, class_names

def load_val_data(data_dir):
    val_data_dir = os.path.join(data_dir, 'val')
    annotations_file = os.path.join(val_data_dir, 'val_annotations.txt')

    with open(annotations_file, 'r') as f:
        annotations = f.readlines()

    val_image_paths = []
    val_labels = []
    class_name_to_label = {}

    for line in annotations:
        parts = line.strip().split('\t')
        image_name = parts[0]
        class_name = parts[1]

        if class_name not in class_name_to_label:
            class_name_to_label[class_name] = len(class_name_to_label)

        label = class_name_to_label[class_name]
        image_path = os.path.join(val_data_dir, 'images', image_name)
        val_image_paths.append(image_path)
        val_labels.append(label)

    return val_image_paths, val_labels

# 设置数据目录路径
data_dir = './tiny-imagenet-200'

# 加载数据
train_image_paths, train_labels, val_image_paths, val_labels, class_names = load_data(data_dir)

# 打印数据集大小
print(f"Train dataset size: {len(train_image_paths)}")
print(f"Validation dataset size: {len(val_image_paths)}")

# 设置批次大小
batch_size = 32

# 创建数据集
train_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(train_image_paths, train_labels, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
)

val_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(val_image_paths, val_labels, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
)

# 获取并打印一些数据
for images, labels in train_gen.take(1):
    print(images.shape, labels.numpy())
for images, labels in val_gen.take(1):
    print(images.shape, labels.numpy())

# 构建模型
class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class FeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.block1 = ConvBlock(filters=64, kernel_size=3)
        self.block2 = ConvBlock(filters=64, kernel_size=3)
        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()

    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.pool(x)
        x = self.block2(x, training=training)
        x = self.pool(x)
        x = self.flatten(x)
        return x

class FewShotModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FewShotModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.dense(x)
        return x

# 定义和编译模型
num_classes = 200
few_shot_model = FewShotModel(num_classes)
few_shot_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = few_shot_model.fit(train_gen, validation_data=val_gen, epochs=20, steps_per_epoch=len(train_image_paths)//batch_size, validation_steps=len(val_image_paths)//batch_size)

# 保存模型
few_shot_model.save('few_shot_model_self_trained', save_format='tf')

# 評估模型並繪製訓練曲線
loss, accuracy = few_shot_model.evaluate(val_gen, steps=len(val_image_paths)//batch_size)
print(f'Validation accuracy: {accuracy * 100:.2f}%')

# 绘制训练和验证的损失和准确率曲线
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.show()

plot_training_history(history)
