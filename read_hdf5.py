from fast_imagenet import ImageNet21kDatasetH5
import numpy as np
from skimage.io import imshow, show

tdata = ImageNet21kDatasetH5('./imagenet21k.hdf5', split='train', n_train=200000, n_val=200000)
vdata = ImageNet21kDatasetH5('./imagenet21k.hdf5', split='val', n_train=200000, n_val=200000)

image, target = tdata[0]
print(target)
imshow(np.array(image))
show()

image, target = vdata[0]
print(target)
imshow(np.array(image))
show()