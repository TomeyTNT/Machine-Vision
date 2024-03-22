from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from google.colab import drive
# drive.mount('/content/drive')

# مشخص کردن اندازه ورودی‌ها
input_shape = (150, 150, 3)

# لود مدل VGG16 با عدم لود لایه‌های Fully-Connected (Dense) بالا
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# فریز کردن تمام لایه‌های وزنی مربوط به مدل اصلی VGG16
for layer in vgg_base.layers:
    layer.trainable = False

# اضافه کردن لایه‌های جدید به مدل
model = Sequential()
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# تولید داده‌های آموزشی، ارزیابی و تست
train_generator = train_datagen.flow_from_directory(
    'E:/dataSets/faze1vgg/train',
    target_size=(150, 150),
    batch_size=20,  # تغییر batch size
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'E:/dataSets/faze1vgg/val',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'E:/dataSets/faze1vgg/test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# تعیین وزن‌های کلاس‌ها برای موازنه داده‌ها
class_weights = {0: 1.66, 1: 2.5}  # بر اساس نسبت تعداد داده‌ها

# کامپایل مدل
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# نمایش ساختار مدل
model.summary()

model.fit(
    train_generator,
    steps_per_epoch = 35,
    epochs=30,
    validation_data=val_generator,
    validation_steps = 50,
    # verbose=1,
    class_weight=class_weights  # استفاده از وزن کلاس‌ها
)

# ارزیابی مدل
train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Train accuracy: {train_accuracy}')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# اجرای پیش‌بینی بر روی تصاویر تست
predictions = model.predict(test_generator)

# تبدیل احتمالات به لیبل‌های نهایی
predicted_labels = (predictions > 0.5).astype(int)

print("Predicted Labels:", predicted_labels)

defec = 0
okok=0

for i in range(len(predicted_labels)):
  if(predicted_labels[i] == 0):
      defec+=1
  elif(predicted_labels[i] == 1):
      okok+=1

print("Defected: ", defec)
print("OK: ", okok)