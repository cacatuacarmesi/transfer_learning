"""CLI wrapper for TFLite Transfer Learning model using ResNet50."""

import os
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 4  # Cambia este valor si tu dataset tiene otro número de clases


class TransferLearningModelResNet(tf.Module):
    """TF Transfer Learning model class using ResNet50."""

    def __init__(self, learning_rate=0.001):
        self.num_classes = NUM_CLASSES

        # Base model: ResNet50 sin la capa final (include_top=False)
        base_model = tf.keras.applications.ResNet50(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet')
        base_model.trainable = False
        self.base = base_model

        # Cálculo del tamaño del bottleneck
        self.num_features = np.prod(base_model.output_shape[1:])

        # Pesos y bias entrenables
        self.ws = tf.Variable(tf.zeros((self.num_features, self.num_classes)), name='ws', trainable=True)
        self.bs = tf.Variable(tf.zeros((1, self.num_classes)), name='bs', trainable=True)

        # Funciones auxiliares
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
    ])
    def load(self, feature):
        """Extrae características (bottleneck) desde la imagen."""
        x = tf.keras.applications.resnet.preprocess_input(tf.multiply(feature, 255))
        bottleneck = tf.reshape(self.base(x, training=False), (-1, self.num_features))
        return {'bottleneck': bottleneck}

    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.float32),  # bottleneck
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),  # one-hot labels
    ])
    def train(self, bottleneck, label):
        """Entrena el clasificador sobre el bottleneck."""
        with tf.GradientTape() as tape:
            logits = tf.matmul(bottleneck, self.ws) + self.bs
            prediction = tf.nn.softmax(logits)
            loss = self.loss_fn(label, prediction)
        gradients = tape.gradient(loss, [self.ws, self.bs])
        self.optimizer.apply_gradients(zip(gradients, [self.ws, self.bs]))
        return {'loss': loss}

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
    ])
    def infer(self, feature):
        """Inferencia directa sobre una imagen."""
        x = tf.keras.applications.resnet.preprocess_input(tf.multiply(feature, 255))
        bottleneck = tf.reshape(self.base(x, training=False), (-1, self.num_features))
        logits = tf.matmul(bottleneck, self.ws) + self.bs
        return {'output': tf.nn.softmax(logits)}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        """Guarda pesos en un archivo de checkpoint."""
        tensor_names = [self.ws.name, self.bs.name]
        tensors_to_save = [self.ws.read_value(), self.bs.read_value()]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save')
        return {'checkpoint_path': checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        """Restaura pesos desde archivo de checkpoint."""
        restored_tensors = {}
        restored_ws = tf.raw_ops.Restore(
            file_pattern=checkpoint_path,
            tensor_name=self.ws.name,
            dt=tf.float32)
        self.ws.assign(restored_ws)
        restored_tensors['ws'] = restored_ws

        restored_bs = tf.raw_ops.Restore(
            file_pattern=checkpoint_path,
            tensor_name=self.bs.name,
            dt=tf.float32)
        self.bs.assign(restored_bs)
        restored_tensors['bs'] = restored_bs

        return restored_tensors

    @tf.function(input_signature=[])
    def initialize_weights(self):
        """Inicializa los pesos del clasificador de forma aleatoria."""
        self.ws.assign(tf.random.uniform((self.num_features, self.num_classes)))
        self.bs.assign(tf.random.uniform((1, self.num_classes)))
        return {'ws': self.ws, 'bs': self.bs}


def convert_and_save(saved_model_dir='saved_model_resnet'):
    """Guarda y convierte el modelo a TFLite."""
    model = TransferLearningModelResNet()

    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            'load': model.load.get_concrete_function(),
            'train': model.train.get_concrete_function(),
            'infer': model.infer.get_concrete_function(),
            'save': model.save.get_concrete_function(),
            'restore': model.restore.get_concrete_function(),
            'initialize': model.initialize_weights.get_concrete_function(),
        })

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    model_file_path = os.path.join('model_resnet.tflite')
    with open(model_file_path, 'wb') as model_file:
        model_file.write(tflite_model)


if __name__ == '__main__':
    convert_and_save()
