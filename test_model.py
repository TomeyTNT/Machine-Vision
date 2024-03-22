import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_binaryModel.h5')

# Show the model architecture
new_model.summary()

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


train_loss, train_accuracy = new_model.evaluate(train_generator)
print(f'Train accuracy: {train_accuracy}')

test_loss, test_accuracy = new_model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# اجرای پیش‌بینی بر روی تصاویر تست
predictions = new_model.predict(test_generator)

# تبدیل احتمالات به لیبل‌های نهایی
predicted_labels = (predictions > 0.5).astype(int)

print("Predicted Labels:", predicted_labels)


# ارزیابی ماژول Noise Classifier
true_labels = test_generator.labels

# محاسبه ماتریس درهم‌ریختگی
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# گزارش کامل ارزیابی
report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(report)

# محاسبه احتمالات پیش‌بینی برای داده‌های تست
predictions = new_model.predict(test_generator)

# تبدیل احتمالات به احتمال کلاس 1
positive_class_probabilities = predictions.ravel()

# محاسبه fpr و tpr برای منحنی ROC
fpr, tpr, thresholds = roc_curve(test_generator.classes, positive_class_probabilities)

# محاسبه مساحت زیر منحنی ROC
roc_auc = auc(fpr, tpr)

# رسم منحنی ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()