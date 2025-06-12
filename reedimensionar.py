import os
from PIL import Image

# Ruta al dataset original (sin redimensionar)
input_dir = 'animals'

# Ruta de salida para el dataset redimensionado
output_dir = 'dataset_resized'

# Tamaño deseado
target_size = (224, 224)

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Recorrer todas las carpetas e imágenes
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    if not os.path.isdir(class_input_path):
        continue

    os.makedirs(class_output_path, exist_ok=True)

    for filename in os.listdir(class_input_path):
        input_file = os.path.join(class_input_path, filename)
        output_file = os.path.join(class_output_path, filename)

        try:
            with Image.open(input_file) as img:
                img = img.convert('RGB')  # Asegura el formato RGB
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(output_file)
                print(f"Redimensionada: {output_file}")
        except Exception as e:
            print(f"Error con {input_file}: {e}")
