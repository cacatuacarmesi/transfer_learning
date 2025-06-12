import tensorflow as tf
import generate_training_model

IMG_SIZE = 224
NUM_CLASSES = 4
BATCH_SIZE = 32
DATASET_PATH = "dataset_resized"  # <-- tu ruta
CHECKPOINT_PATH = "checkpoint.ckpt"  # <-- nombre correcto del checkpoint
SAVED_MODEL_DIR = "saved_model"
TFLITE_PATH = "model.tflite"

# 1. Dataset (solo para inicializar entradas del modelo si es necesario)
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# 2. Crear y preparar el modelo
model = generate_training_model.TransferLearningModel()
model.initialize_weights()

# 3. Restaurar pesos desde checkpoint
model.restore(tf.constant(CHECKPOINT_PATH))
print("✅ Checkpoint restaurado correctamente.")

# 4. Exportar como SavedModel (con firmas necesarias)
tf.saved_model.save(
    model,
    SAVED_MODEL_DIR,
    signatures={
        'load': model.load.get_concrete_function(),
        'train': model.train.get_concrete_function(),
        'infer': model.infer.get_concrete_function(),
        'save': model.save.get_concrete_function(),
        'restore': model.restore.get_concrete_function(),
        'initialize': model.initialize_weights.get_concrete_function(),
    })
print("✅ Modelo guardado como SavedModel.")

# 5. Convertir a TFLite CON firmas
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # importante para usar operaciones no nativas
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"✅ Modelo TFLite exportado correctamente: {TFLITE_PATH}")