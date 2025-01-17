""" A dataset parser that reads single tarfile based datasets
This parser can read datasets consisting if a single tarfile containing images.
I am planning to deprecated it in favour of ParerImageInTar.
Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import tarfile
from time import time

from PIL import Image
from skimage.io import imshow, show
from timm.utils.misc import natural_key

from timm.data.parsers.class_map import load_class_map
from timm.data.parsers.img_extensions import get_img_extensions
from timm.data.parsers.parser import Parser
from tqdm import tqdm


CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]


def get_label(filename: str, target: str):
    peaces = filename.split("_")[1:7]
    index = CATEGORIES_2021.index(target)+1
    relevant_peaces = peaces[:index]
    class_name = "_".join(relevant_peaces)
    return class_name



def extract_tarinfo(tarfile, class_to_idx=None, sort=True, subset: str = 'train', target: str = "full"):
    extensions = get_img_extensions(as_set=True)
    files = []
    labels = []
    for ti in tqdm(tarfile.getmembers(), desc=f"extracting files for subset {subset}"):
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        if subset is not None and not subset in dirname:
            continue
        label = os.path.basename(dirname) if target == "full" \
            else get_label(os.path.basename(dirname), target)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in extensions:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class iNatParserImageTar(Parser):
    """ Single tarfile dataset where classes are mapped to folders within tar
    NOTE: This class is being deprecated in favour of the more capable ParserImageInTar that can
    operate on folders of tars or tars in tars.
    """
    def __init__(self, root, tf=None, class_map='', subset=None, target="full"):
        super().__init__()

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        #assert os.path.isfile(root)
        self.root = root

        if tf is None:
            with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
                self.samples, self.class_to_idx = extract_tarinfo(tf, class_to_idx, subset=subset, target=target)
        else:
            self.samples, self.class_to_idx = extract_tarinfo(tf, class_to_idx, subset=subset, target=target)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        self.subset = subset

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        fileobj = self.tarfile.extractfile(tarinfo)
        return fileobj, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename

"""



This code is for appending hf5 files

h5f.create_dataset('X_train', data=orig_data, compression="gzip", chunks=True, maxshape=(None,)) 


with h5py.File('.\PreprocessedData.h5', 'a') as hf:
    hf["X_train"].resize((hf["X_train"].shape[0] + X_train_data.shape[0]), axis = 0)
    hf["X_train"][-X_train_data.shape[0]:] = X_train_data

    hf["X_test"].resize((hf["X_test"].shape[0] + X_test_data.shape[0]), axis = 0)
    hf["X_test"][-X_test_data.shape[0]:] = X_test_data

    hf["Y_train"].resize((hf["Y_train"].shape[0] + Y_train_data.shape[0]), axis = 0)
    hf["Y_train"][-Y_train_data.shape[0]:] = Y_train_data

    hf["Y_test"].resize((hf["Y_test"].shape[0] + Y_test_data.shape[0]), axis = 0)
    hf["Y_test"][-Y_test_data.shape[0]:] = Y_test_data
"""

if __name__ == '__main__':
    import numpy as np
    data = "../../Downloads/val.tar"
    #print(os.path.exists(data))
    #with tarfile.open(data) as tf:
    #    extract_tarinfo(tf, class_to_idx=None, sort=True, subset=None, target="family")
    for targer in CATEGORIES_2021:
        with tarfile.open(data) as tf:  # cannot keep this open across processes, reopen later
            start = time()
            train = iNatParserImageTar(data, tf=tf, subset="val", target=targer)
            ttime = time()
            print("Target:", targer, "Num classes:", len(train.class_to_idx))
            #print("Total Time:\t", ftime-start)
            #print("Train Samples:\t", len(train.samples), "\ttook", ttime-start, "seconds")
            #print("Val Samples:\t", len(train.samples), "\ttook", ftime-ttime, "seconds")
            #print(len(train.class_to_idx))
            #for i in range(10):
            #    atime = time()
            #    img, target = train[i]
            #    etime = time()
            #    img = Image.open(img).convert('RGB')
            #    img = np.array(img)
            #    print("Requesting took", etime-atime, "seconds")
            #    imshow(img)
            #    show()

            #    img, target = val[i]
            #    img = np.array(Image.open(img).convert('RGB'))
            #    imshow(img)
            #    show()