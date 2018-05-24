import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    features = hog(img, orientations = orient, pixels_per_cell= (pix_per_cell, pix_per_cell),
                   cells_per_block = (cell_per_block, cell_per_block), 
                   transform_sqrt = True, visualise = False, feature_vector = feature_vec)
    return features

def bin_spatial(img, size = (16, 16)):
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins = 32):
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    hist = np.hstack((ch1, ch2, ch3))
    return hist

def img_features(feature_image, hist_bins, orient, pix_per_cell, cell_per_block, spatial_size):
    features = []
    features.append(bin_spatial(feature_image, size=spatial_size))
    features.append(color_hist(feature_image, nbins=hist_bins))
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_LUV2RGB)
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
    features.append(get_hog_features(feature_image, orient, pix_per_cell, cell_per_block))
    return features

def extract_features(imgs, augmented_count = 2, color_space='RGB', spatial_size=(32, 32), 
                     hist_bins=32, orient=9, pix_per_cell=8, 
                     cell_per_block=2):
    features = []
    for img in imgs:
        image = cv2.imread(img)
        image = cv2.resize(image, (64, 64))
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: 
            feature_image = np.copy(image)
        file_features = img_features(feature_image, hist_bins, orient, 
                                     pix_per_cell, cell_per_block, spatial_size)
        features.append(np.concatenate(file_features))
        for i in range(augmented_count):
            file_features = random_augmentation(feature_image)
            file_features = img_features(feature_image, hist_bins, orient, 
                                         pix_per_cell, cell_per_block, spatial_size)
            features.append(np.concatenate(file_features))
    return features

color_space = 'HLS'
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
xy_window = (64, 64)
augmentation_count = 2