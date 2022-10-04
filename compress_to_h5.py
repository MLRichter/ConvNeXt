import io
import pickle
from pathlib import Path

import h5py
import numpy as np
import h5py
import os

import multiprocessing

from PIL import Image
from skimage.io import imshow, show
from tqdm import tqdm
from joblib import Parallel, delayed


def process(f):
    try:
        with Path(f).open("rb") as fp:
            try:
                bytesIO = fp.read()
            except:
                print('Failed to process:', f)
                bytesIO = None
    except:
        bytesIO = None
    return bytesIO


def check(f):
    try:
        with Path(f).open("rb") as fp:
            pass
        return f
    except:
        return None

def func(x):
    return x**2



def verify_filelist():
    batch_size = 75000
    stop_early = 3
    num_cpus = multiprocessing.cpu_count()
    prl = Parallel(n_jobs=num_cpus)


    ## Training Data
    prefix = r'C:\Users\matsl\Downloads\imagenet21k_resized\imagenet21k_train'
    l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))
    files = []
    for file in tqdm(l, 'obtaining files'):
        files.extend(list(map(lambda x : os.path.join(file, x), os.listdir(file))))

    print(len(files))
    l = files
    result = prl(delayed(check)(f) for f in tqdm(l))
    train_files = [r for r in result if r is not None]
    print(f"{len(train_files)} of {len(l)} are valid ({len(l)-len(train_files)} corrupted)")

    with open("train_filelist.pkl", "wb") as fp:
        pickle.dump(train_files, fp)

    ## Validation Data
    prefix = r'C:\Users\matsl\Downloads\imagenet21k_resized\imagenet21k_val'
    l = list(map(lambda x: os.path.join(prefix, x), os.listdir(prefix)))
    files = []
    for file in tqdm(l, 'obtaining files'):
        files.extend(list(map(lambda x: os.path.join(file, x), os.listdir(file))))

    print(len(files))
    l = files
    result = prl(delayed(check)(f) for f in tqdm(l))
    val_files = [r for r in result if r is not None]
    print(f"{len(val_files)} of {len(l)} are valid ({len(l) - len(val_files)} corrupted)")

    with open("val_filelist.pkl", "wb") as fp:
        pickle.dump(val_files, fp)






def main():
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

    batch_size = 75000
    stop_early = 3
    num_cpus = multiprocessing.cpu_count()



    ## Training Data
    prefix = r'C:\Users\matsl\Downloads\imagenet21k_resized\imagenet21k_train'


    cls_map = []
    pool = multiprocessing.Pool(num_cpus)
    with open("train_filelist.pkl", mode="rb") as fp:
        files = pickle.load(fp)
    with open("train_filelist.pkl", mode="rb") as fp:
        valfiles = pickle.load(fp)
    print(len(files))
    num_train = len(files)
    l = files

    raw_img = process(l[0])
    raw = io.BytesIO(raw_img)
    img = np.array(Image.open(raw).convert('RGB'))
    imshow(img)
    show()

    with h5py.File('imagenet21k.hdf5', 'w') as hf:

        def extract(l, cls_map):
            i = 0
            steps = len(l) // batch_size
            with tqdm(desc="extracting images", total=steps) as pbar:
                while i < len(l):
                    if i > batch_size*3:
                        break
                    current_batch = l[i:i + batch_size]
                    mapped = pool.map(process, current_batch)
                    filtered = [m for m in mapped if m is not None]
                    current_res = np.array(filtered)
                    cls_names = [Path(c).parent.name for c in current_batch]
                    cls_idx = []
                    for m, name in zip(mapped,cls_names):
                        if m is None:
                            continue
                        if name not in cls_map:
                            cls_map.append(name)
                        cls_idx.append([cls_map.index(name)])

                    assert len(current_res) == len(cls_idx)

                    filenames = [str(Path(f).relative_to(prefix)) for f in l]
                    if i == 0:
                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        hf.create_dataset('encoded_images', data=np.void(current_res), maxshape=(len(l)+len(valfiles),), dtype=dt)
                        hf.create_dataset('targets', data=np.asarray(cls_idx), maxshape=(len(l)+len(valfiles), 1))
                        #hf.create_dataset('filenames', data=np.asarray(filenames), chunks=True, maxshape=(None,))
                    else:
                        hf["encoded_images"][i:i + current_res.shape[0]] = current_res
                        hf["targets"][i:i + current_res.shape[0]] = np.asarray(cls_idx)


                    i += len(current_res)
                    pbar.update()
            return i
        i = extract(l, cls_map)
    train_samples = i
    num_classes = len(cls_map)
    with open("class_map_train.txt", "w") as fp:
        fp.write(f"train: {train_samples}\nclasses: {num_classes}\n")
        fp.write("\n\n\n"
                 "===========================================Class List==========================================="
                 "\n")
        for cls in cls_map:
            fp.write(f"{cls}\n")

    prefix_val = r'C:\Users\matsl\Downloads\imagenet21k_resized\imagenet21k_val'


    # Validation Data

    l = valfiles
    raw_img = process(l[0])
    raw = io.BytesIO(raw_img)
    img = np.array(Image.open(raw).convert('RGB'))
    imshow(img)
    show()

    with h5py.File('imagenet21k.hdf5', 'a') as hf:

        def extract(l, cls_map):
            i = 0
            steps = len(l) // batch_size
            with tqdm(desc="extracting images", total=steps) as pbar:
                while i < len(l):
                    if i > batch_size*3:
                        break
                    current_batch = l[i:i + batch_size]
                    mapped = pool.map(process, current_batch)
                    filtered = [m for m in mapped if m is not None]
                    current_res = np.array(filtered)
                    cls_names = [Path(c).parent.name for c in current_batch]
                    cls_idx = []
                    for m, name in zip(mapped, cls_names):
                        if m is None:
                            continue
                        if name not in cls_map:
                            print("Extending Class map during validation", name)
                            cls_map.append(name)
                        cls_idx.append([cls_map.index(name)])

                    filenames = [str(Path(f).relative_to(prefix_val)) for f in l]
                    assert len(current_res) == len(cls_idx)

                    hf["encoded_images"][i+train_samples:i+train_samples+current_res.shape[0]] = current_res
                    hf["targets"][i+train_samples:i+train_samples+current_res.shape[0]] = np.asarray(cls_idx)
                    i += len(current_res)
                    pbar.update()
            return i
        i = extract(l, cls_map)
    val_samples = i
    num_classes = len(cls_map)
    with open("class_map_val.txt", "w") as fp:
        fp.write(f"val: {val_samples}\nclasses: {num_classes}\n")
        fp.write("\n\n\n"
                 "===========================================Class List==========================================="
                 "\n")
        for cls in cls_map:
            fp.write(f"{cls}\n")


if __name__ == '__main__':
    #verify_filelist()
    main()