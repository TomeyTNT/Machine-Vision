import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import torch
from skimage.metrics import structural_similarity as ssim
import lpips

# Initialize the LPIPS model
loss_fn_alex = lpips.LPIPS(net='vgg')

# Define a function to calculate LPIPS score for a single image pair
def calculate_lpips(image1, image2):
    # Convert images to torch tensors
    image1_tensor = torch.tensor(image1.transpose(2, 0, 1)).unsqueeze(0).float()
    image2_tensor = torch.tensor(image2.transpose(2, 0, 1)).unsqueeze(0).float()

    # Normalize images to [0, 1]
    image1_tensor /= 255.0
    image2_tensor /= 255.0

    # Calculate LPIPS distance
    lpips_score = loss_fn_alex(image1_tensor, image2_tensor).item()
    return lpips_score

# ارزیابی ماژول Denoiser
psnr_values = []
ssim_values = []
lpips_values = []

# تابع حذف نویز دوره‌ای با استفاده از فیلتر گوسی در فضای فرکانس
def remove_periodic_noise_color(image):
    # اعمال تبدیل فوریه برای هر کانال رنگی
    channels = cv2.split(image)
    channels_filtered = []
    for i in range(len(channels)):
        f_transform = np.fft.fft2(channels[i])
        f_shift = np.fft.fftshift(f_transform)

        # ایجاد فیلتر گوسی در فضای فرکانس
        rows, cols = channels[i].shape
        crow, ccol = rows // 2, cols // 2
        gauss_kernel = cv2.getGaussianKernel(rows, 50)
        gauss_kernel_2d = np.outer(gauss_kernel, gauss_kernel.transpose())
        gauss_kernel_2d = gauss_kernel_2d / gauss_kernel_2d.max()
        f_shift = f_shift * gauss_kernel_2d

        # بازگرداندن تبدیل فوریه به فضای زمان
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # نرمالایز کردن مقادیر پیکسل
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)

        # اضافه کردن کانال رنگی فیلتر شده به لیست
        channels_filtered.append(img_back)

    # ترکیب کانال‌های جدید برای بازگرداندن به فضای رنگی BGR
    filtered_image = cv2.merge(channels_filtered)

    return filtered_image



# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_frameworkModelreznet50.h5')

# Show the model architecture
new_model.summary()

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
    'E:/dataSets/faze2vgg/train',
    target_size=(256, 256),
    batch_size=20,  # تغییر batch size
    class_mode='sparse',
)

val_generator = val_datagen.flow_from_directory(
    'E:/dataSets/faze2vgg/val',
    target_size=(256, 256),
    batch_size=20,
    class_mode='sparse',
    # shuffle=True,
    # seed =42
)

test_generator = test_datagen.flow_from_directory(
    'E:/dataSets/faze2vgg/test',
    target_size=(256, 256),
    batch_size=20,
    class_mode='sparse',
)

# انجام پیش‌بینی بر روی دیتاست تست
predictions = new_model.predict(test_generator, steps=len(test_generator), verbose=1)

# تبدیل احتمالات به لیبل‌ها
predicted_labels = np.argmax(predictions, axis=1)


# مسیر ذخیره‌سازی نتایج پیش‌بینی
output_directory = 'E:/dataSets/faze2output - Copy (12)'
for i in range(len(predicted_labels)):
    # مسیر فایل تصویر
    img_path = os.path.join(test_generator.directory, test_generator.filenames[i])
    img = image.load_img(img_path, target_size=(150, 150))
     # نام فایل
    file_name = os.path.basename(img_path)

    # # مسیر فایل تصویر
    img_path = os.path.join(test_generator.directory, test_generator.filenames[i])
    img = image.load_img(img_path, target_size=(256, 256))
     # نام فایل
    file_name = os.path.basename(img_path)


    if(predicted_labels[i] == 0):
        class_output_directory = os.path.join(output_directory, 'gaussian')
        output_path = os.path.join(class_output_directory, file_name)
        images = cv2.imread(img_path)
        filtered_image = cv2.GaussianBlur(images, (7, 7), 0)
        cv2.imwrite(output_path, filtered_image)
        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')  
         # ابعاد مورد نظر (مثال: 256x256)
        desired_shape = (256, 256)

        # تغییر ابعاد تصاویر
        images_resized = cv2.resize(images, desired_shape)
        filtered_image_resized = cv2.resize(filtered_image, desired_shape)

        # Calculate PSNR
        psnr = cv2.PSNR(images, filtered_image)
        psnr_values.append(psnr)

        # Calculate SSIM
        win_size = min(images_resized.shape[0], images_resized.shape[1])
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        ssim_score, _ = ssim(images_resized, filtered_image_resized, full=True, win_size=win_size, channel_axis=-1)
        ssim_values.append(ssim_score)

        # Calculate LPIPS
        lpips_score = calculate_lpips(images_resized, filtered_image_resized)
        lpips_values.append(lpips_score)

    elif(predicted_labels[i] == 1):
        class_output_directory = os.path.join(output_directory, 'periodic')
        output_path = os.path.join(class_output_directory, file_name)
        images = cv2.imread(img_path)
        # حذف نویز دوره‌ای با استفاده از فیلتر میانه
        filtered_image = remove_periodic_noise_color(images)
        cv2.imwrite(output_path, filtered_image)
        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')  

         # ابعاد مورد نظر (مثال: 256x256)
        desired_shape = (256, 256)

        # تغییر ابعاد تصاویر
        images_resized = cv2.resize(images, desired_shape)
        filtered_image_resized = cv2.resize(filtered_image, desired_shape)
        # Calculate PSNR
        psnr = cv2.PSNR(images, filtered_image)
        psnr_values.append(psnr)

        # # Calculate SSIM
        win_size = min(images_resized.shape[0], images_resized.shape[1])
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        ssim_score, _ = ssim(images_resized, filtered_image_resized, full=True, win_size=win_size, channel_axis=-1)
        ssim_values.append(ssim_score)

        # # Calculate LPIPS
        lpips_score = calculate_lpips(images_resized, filtered_image_resized)
        lpips_values.append(lpips_score)

        
    elif(predicted_labels[i] == 2):
        class_output_directory = os.path.join(output_directory, 'salt&peper')
        output_path = os.path.join(class_output_directory, file_name)
        images = cv2.imread(img_path)
        # حذف نویز دوره‌ای با استفاده از فیلتر میانه
        filtered_image = cv2.medianBlur(images, 7)
        cv2.imwrite(output_path, filtered_image)
        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')  

        # ابعاد مورد نظر (مثال: 256x256)
        desired_shape = (256, 256)

        # تغییر ابعاد تصاویر
        images_resized = cv2.resize(images, desired_shape)
        filtered_image_resized = cv2.resize(filtered_image, desired_shape)


        # Calculate PSNR
        psnr = cv2.PSNR(images, filtered_image)
        psnr_values.append(psnr)

        # # Calculate SSIM
        win_size = min(images_resized.shape[0], images_resized.shape[1])
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        ssim_score, _ = ssim(images_resized, filtered_image_resized, full=True, win_size=win_size, channel_axis=-1)
        ssim_values.append(ssim_score)

        # Calculate LPIPS
        # images_resized = images_resized.astype(np.float32)
        # filtered_image_resized = filtered_image_resized.astype(np.float32)
        lpips_score = calculate_lpips(images_resized, filtered_image_resized)
        lpips_values.append(lpips_score)

    elif(predicted_labels[i] == 3):
        class_output_directory = os.path.join(output_directory, 'Without_noise')
        output_path = os.path.join(class_output_directory, file_name)

        # ذخیره تصویر در مسیر مربوطه
        img.save(output_path)

        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')


# Calculate average PSNR, SSIM, and LPIPS
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_lpips = np.mean(lpips_values)

print(f'Average PSNR: {avg_psnr}')
print(f'Average SSIM: {avg_ssim}')
print(f'Average LPIPS: {avg_lpips}')

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
