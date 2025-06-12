import tensorflow as tf
import numpy as np
import generate_training_model

IMG_SIZE = 224
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset_resized"  # cambia esto a tu ruta

# 1. Cargar imágenes
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# 2. Crear el modelo
model = generate_training_model.TransferLearningModel()
model.initialize_weights()

# 3. Entrenar
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for images, labels in dataset:
        bottleneck = model.load(images)['bottleneck']
        result = model.train(bottleneck, labels)
        print("  Loss:", float(result['loss']))

# 4. Guardar checkpoint
model.save(tf.constant("checkpoint.ckpt"))

# 5. Exportar como SavedModel
saved_model_dir = "saved_model"
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

# 6. Convertir a TFLite congelando variables
def freeze_and_convert(saved_model_dir):
    imported = tf.saved_model.load(saved_model_dir)
    infer = imported.infer.get_concrete_function()
    frozen_func = tf.graph_util.convert_variables_to_constants_v2(infer)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_model = converter.convert()
    with open("model_frozen.tflite", "wb") as f:
        f.write(tflite_model)
    print("\n✅ Modelo TFLite congelado exportado correctamente.")

freeze_and_convert(saved_model_dir)
