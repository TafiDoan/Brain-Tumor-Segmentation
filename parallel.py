import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, binary_opening, binary_erosion, binary_closing
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage import io
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage import io
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.io import imshow


def parallel(ARR):
    # read JPEG image
    im = ARR
    I = im
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, (512, 512))
    im = im[2:-3, 3:-4]
    input_image = im.copy()
    org_img = im.copy()

    I1 = im.copy()
    # im = im.astype(float)

    # Display the input image
    # plt.imshow(im, cmap='gray')
    # plt.title('Input Image')
    # plt.show()

    ########################################
    max_val = np.max(im)
    z = max_val / 255
    im = im / z

    # Crop image to get rid of light box surrounding the image
    # im = im[2:-3, 3:-4]
    # Threshold to create a binary image
    binaryImage = im > 10
    # Get rid of small specks of noise
    # Tính toán các vùng liên thông
    labeled_image, num_features = ndimage.label(binaryImage)
    # Tính toán diện tích của từng vùng
    sizes = ndimage.sum(binaryImage, labeled_image, range(1, num_features + 1))
    # Lọc ra các vùng có diện tích lớn hơn hoặc bằng ngưỡng
    binaryImage_cleaned = np.isin(labeled_image, np.where(sizes >= 10))

    # Seal off the bottom of the head - make the last row white.
    binaryImage[-1, :] = True

    # Fill the image
    binaryImage = binary_fill_holes(binaryImage)

    # Erode away 15 layers of pixels.
    se = disk(15)
    binaryImage = binary_erosion(binaryImage, se)
    # Mask the gray image
    finalImage = im.copy()  # Initialize.
    finalImage[~binaryImage] = 0

    # Display the input image

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(binaryImage, cmap='gray')
    # plt.title('Binary Image')


    # plt.subplot(1, 2, 2)
    # plt.imshow(finalImage, cmap='gray')
    # plt.title('SkullStripping Image')
    # plt.show()

    @cuda.jit
    def update_membership(img, Unew, centroid, fuzziness, k):
        i, j = cuda.grid(2)
        if i < img.shape[0] and j < img.shape[1]:
            for uII in range(k):
                tmp = 0.0
                for uJJ in range(k):
                    disUp = abs(img[i, j] - centroid[uII])
                    disDn = abs(img[i, j] - centroid[uJJ])
                    tmp += (disUp / disDn) ** (2 / (fuzziness - 1))
                Unew[i, j, uII] = 1 / tmp

    def fuzzyCMeans_gpu(img, k):
        row, col = img.shape
        fuzziness = 2  # fuzzification parameter
        epsilon = 0.001  # stopping condition
        max_iter = 100  # number of maximum iteration

        Uold = np.random.rand(row, col, k)
        dep_sum = np.sum(Uold, axis=2, keepdims=True)
        Uold /= dep_sum

        centroid = np.zeros(k)
        for i in range(k):
            centroid[i] = np.sum(Uold[:, :, i] * img) / np.sum(Uold[:, :, i])

        obj_func_old = 0
        for i in range(k):
            obj_func_old += np.sum((Uold[:, :, i] * img - centroid[i]) ** 2)

        for iter in range(max_iter):
            Unew = np.zeros_like(Uold)

            # CUDA kernel invocation
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(row / threadsperblock[0])
            blockspergrid_y = math.ceil(col / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            update_membership[blockspergrid, threadsperblock](img, Unew, centroid, fuzziness, k)
            cuda.synchronize()

            obj_func_new = 0
            for i in range(k):
                obj_func_new += np.sum((Unew[:, :, i] * img - centroid[i]) ** 2)

            if np.max(np.abs(Unew - Uold)) < epsilon or abs(obj_func_new - obj_func_old) < epsilon:
                break
            else:
                Uold = Unew ** fuzziness
                for i in range(k):
                    centroid[i] = np.sum(Uold[:, :, i] * img) / np.sum(Uold[:, :, i])
                obj_func_old = obj_func_new

        return Unew, centroid, obj_func_new

    # Usage
    img = np.array(finalImage, dtype=float)  # Assuming finalImage is defined elsewhere as an image matrix
    k = 6
    Unew, centroid, obj_func_new = fuzzyCMeans_gpu(img, k)


    # import matplotlib.pyplot as plt

    # for i in range(k):
    #     plt.subplot(1, k, i + 1)
    #     plt.imshow(Unew[:, :, i], cmap='gray')
    #     plt.title('Cluster {}'.format(i+1))
    # plt.show()

    # Calculate the number of black pixels in each image
    black_pixel_counts = []
    for i in range(k):
        black_pixel_count = np.sum(Unew[:, :, i] < 0.01)  # Assuming black pixels have intensity close to 0
        black_pixel_counts.append(black_pixel_count)

    # Select the image with the most black pixels
    index_of_max_black_pixels = np.argmax(black_pixel_counts)
    image_with_most_black_pixels = Unew[:, :, index_of_max_black_pixels]

    # # Visualize the selected image
    # plt.imshow(image_with_most_black_pixels, cmap='gray')
    # plt.title('Image with Most Black Pixels')
    # plt.show()



    # Assuming 'finalImage' is defined elsewhere as an image matrix
    finalImage = image_with_most_black_pixels.copy()  # Example random image

    # Convert image to uint8 and normalize it
    img = (finalImage / np.max(finalImage) * 255).astype(np.uint8)

    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Perform connected component analysis (CCA)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Find the label corresponding to the largest white area
    largest_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a mask for the largest white area
    largest_area_mask = np.where(labels == largest_area_label, 255, 0).astype(np.uint8)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(img, img, mask=largest_area_mask)

    # Display the filtered image
    # plt.imshow(filtered_image, cmap='gray')
    # plt.title('Filtered Image')
    # plt.show()

# Assuming IMMM is already loaded and is a grayscale or RGB image
    IMMM = filtered_image
    if len(IMMM.shape) == 3:  # Check if the image is RGB
        IMMM = rgb2gray(IMMM)

    # Convert image to binary
    thresh = threshold_otsu(IMMM)
    im_bin = IMMM > thresh

    # Perform edge detection using Canny
    var1 = canny(im_bin)
    # Resize var1 to match the shape of rescale_intensity(org_img, out_range=(0, 1))
    # var1_resized = resize(var1, org_img.shape[:2])

    # Fuse the images
    # Đọc ảnh đen trắng
    gray_image = org_img

    # Tạo ảnh màu RGB từ ảnh đen trắng
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    image2 = var1
    # Hiển thị và lưu ảnh đã chuyển đổi
    # plt.imshow(rgb_image)
    # plt.title('Input Image')
    # plt.show()
    height, width = var1.shape

    # Lặp qua từng pixel của ảnh 1 và 2
    for y in range(height):
        for x in range(width):
            if ( 0 != image2[y, x]):
                # Chuyển màu tại vị trí (x, y) của ảnh 1 sang màu xanh
                rgb_image[y, x,] = [0, 255, 0]  # Màu xanh
    return rgb_image

