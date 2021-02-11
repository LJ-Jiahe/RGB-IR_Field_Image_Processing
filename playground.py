import matplotlib
from skimage import io
import matplotlib.pyplot as plt

matplotlib.use('MacOSX')

image_file = '/Users/lj/Library/Mobile Documents/com~apple~CloudDocs/Developer/RGB-IR_Field_Image_Processing/data/field_image/BRC/BRC_20191008_145306_Thermal.tif'
img = io.imread(image_file)
plt.imshow(img[:, :, 0])
plt.show()
