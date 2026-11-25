import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
  img_height = 300
  img_width = 200
  batch_size = 32
  epochs = 50
  training_dir = Path(__file__).resolve().parent
  data_dir = training_dir.parent / 'dataset_kaggle'
  run_dir = training_dir / 'runs' / 'mobilenet'
  run_dir.mkdir(parents=True, exist_ok=True)
  class_mapping_file_path = run_dir / 'class_mapping.txt'

  train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
  )

  train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
  )

  validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
  )

  class_indices = train_generator.class_indices
  with open(class_mapping_file_path, 'w') as f:
    for class_name, class_index in class_indices.items():
      f.write(f"{class_index}: {class_name}\n")

  base_model = tf.keras.applications.MobileNetV2(
      input_shape=(img_height, img_width, 3),
      include_top=False,
      weights='imagenet'
  )
  base_model.trainable = False

  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(train_generator.num_classes, activation='softmax')
  ])


  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
  )

  model.save(run_dir / 'cnn_newModel_v1.h5')

  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  plt.show()
