import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 128
batch = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode="categorical",
    subset="validation"
)

print(train.class_indices)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=5)
model.save("crop_model.h5")
