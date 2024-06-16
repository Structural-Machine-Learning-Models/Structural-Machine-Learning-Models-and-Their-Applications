from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Softmax
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import os

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class WindowAttention(Layer):
    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = Dense(dim * 3)
        self.proj = Dense(dim)
        self.softmax = Softmax(axis=-1)
        self.scale = dim ** -0.5

    def call(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # 3, B, num_heads, N, C//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.softmax(attn)

        x = attn @ v
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        x = self.proj(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, num_heads, window_size, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

    def build(self, input_shape):
        _, height, width, channels = input_shape
        self.channels = channels
        self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=channels)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        if self.shift_size > 0:
            inputs = tf.roll(inputs, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.window_size, self.window_size, 1],
            strides=[1, self.window_size, self.window_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches_shape = tf.shape(patches)
        print(f'Patches shape: {patches_shape}')

        patches = tf.reshape(patches, [-1, self.window_size * self.window_size, channels])
        attn_output = self.mha(patches, patches)
        attn_output = tf.reshape(attn_output, [-1, height, width, channels])

        if self.shift_size > 0:
            attn_output = tf.roll(attn_output, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        return attn_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'shift_size': self.shift_size,
        })
        return config

class ShiftedWindowAttention(Layer):
    def __init__(self, dim, num_heads, window_size, shift_size=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = Dense(dim * 3)
        self.proj = Dense(dim)
        self.softmax = Softmax(axis=-1)
        self.scale = dim ** -0.5

    def build(self, input_shape):
        H, W = input_shape[1], input_shape[2]
        if min(H, W) < self.window_size:
            self.shift_size = 0
            self.window_size = min(H, W)

        self.attn_mask = self.create_mask(H, W)
        super().build(input_shape)

    def create_mask(self, H, W):
        if self.shift_size > 0:
            img_mask = tf.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (
                slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, float('-inf'), float(0))
        else:
            attn_mask = None
        return attn_mask

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
        windows = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        windows = tf.reshape(windows, [-1, window_size, window_size, C])
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H, W, -1])
        return x

    def call(self, x):
        H, W = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, self.dim])

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, self.dim])

        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        return x

class SwinTransformerLayer(Layer):
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = ShiftedWindowAttention(dim, num_heads, window_size, shift_size)

        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            Dense(dim * mlp_ratio, activation='gelu'),
            Dense(dim)
        ])

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x

class PatchEmbedding(Layer):
    def __init__(self, patch_size=4, embed_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)

    def call(self, x):
        B, H, W, C = x.shape
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        x = tf.reshape(x, [B, -1, self.embed_dim])
        return self.proj(x)

class SwinTransformer(Model):
    def __init__(self, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedding(patch_size=4, embed_dim=embed_dim)
        self.pos_drop = tf.keras.layers.Dropout(0.1)

        self.layers = []
        for i in range(len(depths)):
            for j in range(depths[i]):
                self.layers.append(
                    SwinTransformerLayer(
                        dim=embed_dim * (2 ** i),
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio
                    )
                )
            if (i < len(depths) - 1):
                self.layers.append(PatchMerging(embed_dim=embed_dim * (2 ** i)))

    def call(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        return x

class PatchMerging(Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.reduction = Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H // 2, W // 2, 4 * C])
        x = self.reduction(x)
        return x


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


def load_data(data_dir):
    train_data_dir = os.path.join(data_dir, 'train')
    class_names = sorted(os.listdir(train_data_dir))
    train_image_paths, train_labels = get_image_paths_and_labels(train_data_dir, class_names)
    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    val_ds, val_class_names = load_val_data(data_dir)
    return train_ds, val_ds, class_names


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

    val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
    val_class_names = list(class_name_to_label.keys())

    return val_ds, val_class_names


def preprocess(image_path, label):
    image_path = tf.cast(image_path, tf.string)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def prepare_datasets(train_ds, val_ds, batch_size=16):
    train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds

# Example usage:
data_dir = './tiny-imagenet-200'
train_ds, val_ds, class_names = load_data(data_dir)
train_ds, val_ds = prepare_datasets(train_ds, val_ds, batch_size=8)
class SwinTransformerBlock(layers.Layer):
    def __init__(self, num_heads, window_size, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

    def build(self, input_shape):
        _, height, width, channels = input_shape
        self.channels = channels
        self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=channels)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        if self.shift_size > 0:
            inputs = tf.roll(inputs, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.window_size, self.window_size, 1],
            strides=[1, self.window_size, self.window_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches_shape = tf.shape(patches)
        print(f'Patches shape: {patches_shape}')

        patches = tf.reshape(patches, [-1, self.window_size * self.window_size, channels])
        attn_output = self.mha(patches, patches)
        attn_output = tf.reshape(attn_output, [-1, height, width, channels])

        if self.shift_size > 0:
            attn_output = tf.roll(attn_output, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        return attn_output

    def compute_output_shape(self, input_shape):
        return input_shape

def swin_transformer(input_shape=(224, 224, 3), num_classes=200):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(96, kernel_size=4, strides=4)(input)
    x = layers.LayerNormalization()(x)

    x = SwinTransformerBlock(num_heads=3, window_size=7)(x)
    x = layers.LayerNormalization()(x)

    x = SwinTransformerBlock(num_heads=6, window_size=7, shift_size=3)(x)
    x = layers.LayerNormalization()(x)

    x = SwinTransformerBlock(num_heads=12, window_size=7)(x)
    x = layers.LayerNormalization()(x)

    x = SwinTransformerBlock(num_heads=24, window_size=7, shift_size=3)(x)
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model

# Model creation
model = swin_transformer(input_shape=(224,224,3),num_classes=200)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
try:
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
except Exception as e:
    print(f"Error during training: {e}")

# Saving the model
model.save('swin_transformer_model.h5')

# Evaluating the model
loss, accuracy = model.evaluate(val_ds)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Plotting the training process
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()