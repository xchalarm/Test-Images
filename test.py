# Improved Siamese Network with Transfer Learning and Corrected Labels
# Colab Version

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNet
import pandas as pd

BATCH_SIZE = 8
MAX_EPOCH = 100
IMAGE_SIZE = (224, 224)
TRAIN_IM_PAIR = 765
VALIDATE_IM_PAIR = 77

# Pretrained Encoder Model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
encoder = Model(inputs=base_model.input, outputs=x)

# Freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# Siamese Network structure
input_1 = Input(shape=(224,224,3))
input_2 = Input(shape=(224,224,3))

encoded_1 = encoder(input_1)
encoded_2 = encoder(input_2)

merged = tf.keras.layers.concatenate([encoded_1, encoded_2])
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

siamese_model = Model(inputs=[input_1, input_2], outputs=output)

siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# ImageDataGenerator with Data Augmentation
def myGenerator(path, batch_size=BATCH_SIZE, subset='training'):
    dataframe = pd.read_csv('shuffled_new.csv')
    dataframe['Winner'] = dataframe['Winner'].apply(lambda x: x-1)  # Change labels from (1,2) to (0,1)

    datagen = ImageDataGenerator(rescale=1./255,
                                 rotation_range=15,
                                 horizontal_flip=True,
                                 brightness_range=[0.8,1.2],
                                 validation_split=0.1)

    input_generator_1 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=path,
                                              x_col='Image 1',
                                              y_col='Winner',
                                              class_mode='raw',
                                              target_size=(224,224),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=42,
                                              subset=subset)

    input_generator_2 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=path,
                                              x_col='Image 2',
                                              y_col='Winner',
                                              class_mode='raw',
                                              target_size=(224,224),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=42,
                                              subset=subset)

    while True:
        in_batch_1 = next(input_generator_1)
        in_batch_2 = next(input_generator_2)
        yield (in_batch_1[0], in_batch_2[0]), in_batch_1[1]

# Callbacks for better training control
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Training with Validation
history = siamese_model.fit(myGenerator('dataset/merge', subset='training'),
                            steps_per_epoch=TRAIN_IM_PAIR//BATCH_SIZE,
                            epochs=MAX_EPOCH,
                            validation_data=myGenerator('dataset/merge', subset='validation'),
                            validation_steps=VALIDATE_IM_PAIR//BATCH_SIZE,
                            callbacks=[checkpoint, early_stop])

# Save final model
siamese_model.save('siamese_final.keras')
