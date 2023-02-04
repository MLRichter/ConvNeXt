# ConvNeXt Cluster Benchmark Training

### Fetching ImageNet21K-P
ImageNet21K-P can be obtained from here:

https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md

## Running Multi-Node Training
By default, I recommend training with submitit. However, submitit does not 
make use of tcp-based storage, since you cannot set MASTER_ADDRESS trivially, like you
could in a SLURM-bash-script.
I recommend going the SLURM-script route if you run on multiple nodes and you cannot guarantee that file-locks 
are enforced over multiple nodes.
In this case there is a non-zero-chance of a deadlock during process group initialization, leading to a deadlock.


### Submitit (SLURM)

#### ImageNet1K (.h5-file)
This training uses 64 GPUs, runs for 3 days or until 300 epochs. If this runs has been started before it auto-resumes from 
the last completed epoch.

This setup is made for V100 GPUs with 16GB of memory.
```
python run_with_submitit.py --nodes 16 --ngpus 4 --timeout 72 --job_dir ~/save/my/results/here/please/ --model convnext_large --drop_path 0.5  --batch_size 16 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path ~/scratch/ilsvrc2012.hdf5 --enable_wandb False --use_amp False --num_workers 6 --copy True --epochs 300
```

#### ImageNet21K-P
This training uses 64 GPUs, runs for 3 days or until 300 epochs. If this runs has been started before it auto-resumes from 
the last completed epoch.

This setup is made for A100 GPUs with 40GB of memory.
```
python run_with_submitit.py --nodes 16 --ngpus 2 --timeout 72 --job_dir ~/save/my/results/here/please/ --data_set IMNET21K --epochs 90 --warmup_epochs 5 --model convnext_large --drop_path 0.1 --batch_size 32 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path ~/scratch/imagenet21k_resized.tar --enable_wandb True --use_amp False --num_workers 6 --copy True 
```
### Running as script

In case of a slurm is not available or submitit is not an option it may be desireable to run
the training directly from the main script:

```
python main.py --world size 64 --timeout 72 --job_dir ~/save/my/results/here/please/ --data_set IMNET21K --epochs 90 --warmup_epochs 5 --model convnext_large --drop_path 0.1 --batch_size 32 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path ~/scratch/imagenet21k_resized.tar --enable_wandb True --use_amp False --num_workers 6 --copy True --tcp true
```

In this case it has be ensured that MASTER_ADDR and MASTER_PORT are set in the environment of every single process.
Also, the environment variables `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are required.
`RANK` is the total index of the GPU, `LOCAL_RANK` refers to the rank of the GPU in the respective node.
For example, in a two-node training with 4 GPUs per node, the 1st GPU in the first node has rank 0 and local_rank 0.
The 4th GPU in the 2nd node has a rank of 7 and a local rank of 3.
Please note that the example command above has the tcp-flag set, to indicate that communication of the process group is 
handled via tcp and is not filebased.

### Using WANDB for simple monitoring
Ensure that wandb is in offline mode if you do not have access to the internet.
Otherwise, you can use it to monitor performance on the master node as well as GPU-util, which can come in handy.

## Troubleshooting

### OOM-Errors
The commands shown above are designed to fit into very specific setups regarding GPU memory.
If the setup you are using has insufficient GPU-memory you may half the `--batch_size` and double `--update_freq`, repeat
this step until the forward pass does no longer raises errors.

### Deadlock During Initializing Process Group
If the process group cannot be initialized due to a deadlock occuring the likely cause the file-based communication, which is enabled by default.
To circumvent this use tcp-basesd communication. In this case it has be ensured that `MASTER_ADDR` and `MASTER_PORT` are set in the environment of every single process.
Also, the environment variables `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are required.

### Mixed Precision
In case you are interested in mixed precision performance you can enable mixed-precision training using `--use_amp true`, however
in this case I recommend using `convnext_tiny` instead of `convnext_large`, since large models tend to produce irreversable
rounding errors more often, which results in individual processes dying, deadlocking the entire process group.



