import tarfile
import os
import tqdm

def extract(src, tgt):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Open the .tar.gz file in read mode
    with tarfile.open(src, "r:gz") as gz_file:
        # Open a new .tar file in write mode
        with tarfile.open(tgt, "w") as tar_file:
            # Iterate over each file in the .tar.gz archive
            for member in tqdm.tqdm(gz_file.getmembers(), "extracting files"):
                # Extract the file from the .tar.gz archive to the temporary directory
                if len(member.name.split(".")) == 1:
                    continue
                gz_file.extract(member, tmp_dir)

                # Construct the relative path to the file in the new .tar archive
                rel_path = os.path.relpath(os.path.join(tmp_dir, member.name), tmp_dir)

                # Add the extracted file to the new .tar file using the relative path
                tar_file.add(os.path.join(tmp_dir, member.name), arcname=rel_path)



if __name__ == '__main__':
    src = "../../Downloads/train_mini.tar.gz"
    tgt = "../../Downloads/train_mini.tar"
    extract(src, tgt)
