import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers ,callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# ابعاد ورودی‌های مدل
input_shape = (256, 256, 3)

resnet_50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
# فریز کردن تمام لایه‌های وزنی مربوط به مدل اصلی VGG16
for layer in resnet_50.layers:
    layer.trainable = False

# # تعداد لایه‌هایی که می‌خواهید فریز شوند
# freeze_layers = len(resnet_50.layers) - 1
# for layer in resnet_50.layers[:freeze_layers]:
#     layer.trainable = False

# تعریف مدل جدید برای طبقه‌بندی باینری
model = Sequential([
    resnet_50,  # لایه‌های اولیه از مدل ResNet34
    layers.GlobalAveragePooling2D(),  # استفاده از Global Average Pooling برای تبدیل خروجی‌های مدل به بردار یک بعدی
    layers.Dense(256, activation='relu',
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
    ),
    layers.BatchNormalization(),
    layers.Normalization(axis=None),
    # layers.Dropout(0.4), # لایه‌ی پنهان با 256 گره و فعال‌ساز relu
    layers.Dense(4, activation='softmax'),
])


# کامپایل مدل
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # تنظیم learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# نمایش ساختار مدل
model.summary()

# تولید کننده داده برای داده‌های آموزشی و ارزیابی
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)

# تولید داده‌های آموزشی، ارزیابی و تست
train_generator = train_datagen.flow_from_directory(
    'E:/dataSets/faze2vgg/train',
    target_size=(256, 256),
    batch_size=20,  # تغییر batch size
    class_mode='sparse',
)

val_generator = val_datagen.flow_from_directory(
    'E:/dataSets/faze2vgg/val',
    target_size=(256, 256),
    batch_size=20,  # تغییر batch size
    class_mode='sparse',
)

test_generator = test_datagen.flow_from_directory(
    'E:/dataSets/faze2vgg/test',
    target_size=(256, 256),
    batch_size=20,  # تغییر batch size
    class_mode='sparse',
)



# تعیین وزن‌های کلاس‌ها برای موازنه داده‌ها
class_weights = {0: 10, 1: 10, 2:3.33, 3:2}  # بر اساس نسبت تعداد داده‌ها

callback = callbacks.EarlyStopping(monitor='val_loss', min_delta= 1e-3, patience=5, verbose=1, restore_best_weights = True)

model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    # verbose=1,
    # validation_steps = 35,
    callbacks=[callback],
    steps_per_epoch = 35,
    class_weight=class_weights  # استفاده از وزن کلاس‌ها
)

# # ارزیابی مدل
# train_loss, train_accuracy = model.evaluate(train_generator)
# print(f'Train accuracy: {train_accuracy}')

# ارزیابی مدل
train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Train accuracy: {train_accuracy}')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

test_loss, test_accuracy = model.evaluate(test_generator, steps = test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_accuracy}')


gaussian=0
periodic=0
salt=0
normal=0

# انجام پیش‌بینی بر روی دیتاست تست
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# تبدیل احتمالات به لیبل‌ها
predicted_labels = np.argmax(predictions, axis=1)



for i in range(len(predicted_labels)):
    if(predicted_labels[i] == 0):
        gaussian+=1
    elif(predicted_labels[i] == 1):
        periodic+=1
    elif(predicted_labels[i]==2):
        salt+=1
    else:
        normal+=1

print("gaussian: ", gaussian)
print("periodic: ", periodic)
print("salt: ", salt)
print("normal: ", normal)
# چاپ لیبل‌های پیش‌بینی شده
print("Predicted Labels:", predicted_labels)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# انجام پیش‌بینی بر روی داده‌های تست
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)

# محاسبه ماتریس درهم‌ریختگی
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# محاسبه دقت کلی
accuracy = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", accuracy)

# محاسبه دقت هر کلاس
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("Class-wise Accuracy:", class_accuracy)

model.save('my_model10.h5')