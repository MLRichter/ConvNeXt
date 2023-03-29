# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import pickle
import tarfile
from os.path import isfile

from torchvision.datasets import ImageFolder, CIFAR100, CIFAR10, SVHN, Omniglot
from torchvision import transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from fast_imagenet import ImageNetDatasetH5
from inaturalist_tarbal_parser import iNatParserImageTar
from nested_tarbal_parser import ParserImageInTar
from tarbal_parser import ParserImageTar
from merge_datasets import obtain_class_mapping, MergeDataset
from copy import deepcopy

imnet21k_cache = {}
global multi_dataset_class_mapping
multi_dataset_class_mapping = None

greyscale_datasets = [
    "OMNIGLOT", "MNIST"
]

def build_dataset(is_train, args):
    transform = build_transform(is_train, args, to_rgb=args.data_set in greyscale_datasets)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    # this is specifically for multi-dataset training
    if ";" in args.data_set:
        datasets = args.data_set.split(";")
        datapaths = args.data_path.split(";")
        total_classes = 0
        dataset_obj = []
        for ds, dp in zip(datasets, datapaths):
            copy_args = deepcopy(args)
            copy_args.data_set = ds
            copy_args.data_path = dp
            dataset, nb_classes = build_dataset(is_train=is_train, args=copy_args)
            total_classes += nb_classes
            dataset_obj.append(dataset)
        global multi_dataset_class_mapping
        if multi_dataset_class_mapping is None:
            multi_dataset_class_mapping = obtain_class_mapping(dataset_obj)
        dataset = MergeDataset(dataset_obj, class_mapping=multi_dataset_class_mapping)
        nb_classes = total_classes
        with open("class_mapping.json", "w") as fp:
            ds_to_idx = {ds: idx for idx, ds in enumerate(datasets)}
            json_out = {"Dataser2Index": ds_to_idx, "Class2Index": multi_dataset_class_mapping}
            json.dump(json_out, fp)
    elif args.data_set == 'CIFAR':
        dataset = CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == "OMNIGLOT":
        dataset = Omniglot(args.data_path, background=is_train, transform=transform, download=True)
        nb_classes = 963
    elif args.data_set == 'SVHN':
        dataset = SVHN(args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        if isfile(args.data_path):
            print("Detected file instead of folder, assuming hdf5")
            dataset = ImageNetDatasetH5(args.data_path, split='train' if is_train else 'val', transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "IMNET21K":
        from dataset import ImageDataset
        print("Pretraining on ImageNet21K")

        if "train" not in imnet21k_cache:
            with tarfile.open(args.data_path) as tf:  # cannot keep this open across processes, reopen later
                train = ParserImageTar(args.data_path, tf=tf, subset="train")
                val = ParserImageTar(args.data_path, tf=tf, subset="val")
                imnet21k_cache["train"] = train
                imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                                reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        nb_classes = 10450

    elif "INATMINI" in args.data_set:
        dataset, target = args.data_set.split("_")
        from dataset import ImageDataset
        print("Pretraining on ImageNet21K")

        if "train" not in imnet21k_cache:
            train = iNatParserImageTar(os.path.join(args.data_path, "train_mini.tar"), subset="train_mini", target=target)
            val = iNatParserImageTar(os.path.join(args.data_path, "val.tar"), subset="val", target=target)
            imnet21k_cache["train"] = train
            imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                               reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        nb_classes = len(imnet21k_cache["val"].class_to_idx)
    elif "INATFULL" in args.data_set:
        dataset, target = args.data_set.split("_")
        from dataset import ImageDataset
        print("Pretraining on ImageNet21K")

        if "train" not in imnet21k_cache:
            train = iNatParserImageTar(os.path.join(args.data_path, "train.tar"), subset="train", target=target)
            val = iNatParserImageTar(os.path.join(args.data_path, "val.tar"), subset="val", target=target)
            imnet21k_cache["train"] = train
            imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                               reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        nb_classes = len(imnet21k_cache["val"].class_to_idx)
    elif args.data_set == "FOOD101":
        from dataset import ImageDataset
        print("Pretraining on FOOD101")

        if "train" not in imnet21k_cache:
            with tarfile.open(args.data_path) as tf:  # cannot keep this open across processes, reopen later
                train = ParserImageTar(args.data_path, tf=tf, subset="train")
                val = ParserImageTar(args.data_path, tf=tf, subset="test")
                imnet21k_cache["train"] = train
                imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                                reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        nb_classes = 101
    elif args.data_set == "FALLIMNET21K":
        from dataset import ImageDataset
        print("Pretraining on ImageNet21K")
        train = ParserImageInTar(args.data_path)
        val = train
        imnet21k_cache["train"] = train
        imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                                reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                                transform=transform)
        nb_classes = len(imnet21k_cache["train"].class_to_idx)
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def _inject_to_rgb_after_to_tensor(transform_pipeline: list):
    for i, t in enumerate(transform_pipeline):
        if t.__class__.__name__ == "ToTensor":
            transform_pipeline.insert(i+1, transforms.Lambda(lambda x: x.repeat((3, 1, 1))))
            break


def build_transform(is_train, args, to_rgb=False):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if to_rgb:
            transform.transforms.insert(0, transforms.Lambda(lambda x: x.convert('RGB')))
        #    _inject_to_rgb_after_to_tensor(transform.transforms)

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if to_rgb:
        t.insert(0, transforms.Lambda(lambda x: x.convert('RGB')))

        #t.append(transforms.Lambda(lambda x: x.repeat((3, 1, 1))))
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == "__main__":
    from pathlib import Path
    Path("../tmp").mkdir(parents=True, exist_ok=True)
    class FakeArgs:

        def __init__(self):
            self.data_path = "../tmp;../tmp;../tmp;../tmp"
            self.data_set = "OMNIGLOT;CIFAR;CIFAR10;SVHN"
            self.input_size = 224
            self.crop_pct = None
            self.imagenet_default_mean_and_std = "IMNET"
            self.color_jitter = 0.2
            self.aa = 'rand-m9-mstd0.5-inc1'
            self.train_interpolation = 'bicubic'
            self.reprob = 0.25
            self.remode = 'pixel'
            self.recount = 1
    ds, classes = build_dataset(is_train=True, args=FakeArgs())
    val_ds, val_classes = build_dataset(is_train=False, args=FakeArgs())

    print("Samples", len(ds), "Classes", classes, "CumSum", ds.cummulative_sizes)
    print("Val Samples", len(val_ds), "Val Classes", val_classes, "CumSum", val_ds.cummulative_sizes)
    print(multi_dataset_class_mapping)
