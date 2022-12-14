import pickle
import tarfile
import os
from glob import glob
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

from joblib import Parallel, delayed
from timm.utils import natural_key
from tqdm import tqdm
from PIL import Image
import logging
import numpy as np

from nested_tarbal_parser import CACHE_FILENAME_SUFFIX
from nested_tarbal_parser import ParserImageInTar

_logger = logging.getLogger(__name__)

#logging.basicConfig(level=logging.DEBUG)

import tarfile


def create_tar_file(src_path, tar_path):
    # Create a new tar file if it doesn't exist
    tar = tarfile.open(tar_path, "w")


    # Add the folder at the provided path to the tar file
    for file_name in os.listdir(src_path):
        tar.add(os.path.join(src_path, file_name), arcname=file_name)

    # Close the tar file
    tar.close()
    rmtree(str(src_path))


def create_parser(src_tar: Path) -> ParserImageInTar:
    return ParserImageInTar(root=str(src_tar))


def save_datapoint(image: Image.Image, root: Path, cls_name: str, filename: str, size: int):
    path = root / cls_name / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    resized_img = image.resize((size, size))
    with path.open("wb") as fp:
        #print("saving", str(path))
        resized_img.save(fp)


def obtain_image(img):
    try:
        img = Image.open(img).convert('RGB')
    except Exception as e:
        img = None
    return img


def process_datapoints(parser: ParserImageInTar, cls: int, tgt: Path, size: int):
    if not (Path(tgt) / parser.class_idx_to_name[cls]).exists():
        if (Path(tgt) / parser.class_idx_to_name[cls]).with_suffix(".tar").exists():
            print(
                str((Path(tgt) / parser.class_idx_to_name[cls]).with_suffix(".tar")),
                "is exists skipping..."
                  )
            return
        else:
            print("Exists but is corrupted, rewriting...")
    condidates = np.where(parser.targets == cls)[0]
    print("Found", len(condidates), "for class", cls)
    for i, candidate in enumerate(condidates):
        #print("processing datapoint", i, "for class", cls)
        tar_img, cls = parser[candidate]
        img = obtain_image(tar_img)
        if img is None:
            print("Failed to read", parser.filename(candidate))
            continue
        else:
            pass
            #print("Processing to read", parser.filename(candidate))
        cls_name = parser.class_idx_to_name[cls]
        filename = parser.filename(candidate)
        save_datapoint(image=img, root=Path(tgt), cls_name=cls_name, filename=filename, size=size)
    create_tar_file(src_path=Path(tgt) / cls_name, tar_path=(Path(tgt) / cls_name).with_suffix(".tar"))


def create_jobs(parser: ParserImageInTar, n_classes: int, tgt: Path, size: int
                ) -> List[Tuple[ParserImageInTar, int, Path, int]]:
    jobs = []
    for i, idx in enumerate(tqdm(list(parser.class_idx_to_name.keys()), "Creating Jobs")):
        if idx >= n_classes:
            break
        jobs.append((parser, idx, tgt, size))
    return jobs


def main(src: str, tgt: str, njobs: int, size: int, n_classes: int):
    parser = create_parser(src_tar=Path(src))
    jobs = create_jobs(parser, n_classes=n_classes, tgt=tgt, size=size)
    pool = Parallel(n_jobs=njobs, verbose=10000)
    pool(delayed(process_datapoints)(*job) for job in jobs)


if "__main__" == __name__:
    src: str = "../../Downloads/fall11_whole.tar"
    tgt: str = "../../Downloads/ImageNet21K_Fall"
    jobs = 1
    main(src, tgt, jobs, size=224, n_classes=10000000000000000000)