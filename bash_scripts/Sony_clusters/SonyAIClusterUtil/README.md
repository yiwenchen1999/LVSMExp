# Slurm User Guide
## Quick Tutorial

The following tutorials are available to help you get started with the Slurm cluster once you are onboarded.

* [Building and Running a Singularity Container in a Slurm Job](./tutorial/singularity.md)
* [Launching and Connecting to Jupyter Server](./tutorial/jupyter.md)
* [PyTorch Multi-GPU, Multi-node Distributed Training on Slurm and Singularity](./tutorial/pytorch-multi-gpu-multi-node/README.md)

## Cluster overview
The **Crusoe Slurm cluster** builds on top of 20 compute nodes with 8x Nvidia H100 GPUs each, and 4 compute nodes with 10x L40S GPUs each, and 2 login nodes. It enables sharing of compute resources across users and teams via the Slurm scheduler.

The Crusoe compute nodes and resources are grouped into so-called partitions, set up to cover the compute needs for each work team. To run, jobs needs to be submitted to one or more of these partitions, with a Slurm controller node scheduling these according to a priority system. To submit jobs you need to first `ssh` to the login node `mfml1` or `mfmsc`.

Currently, there are two login nodes: `mfml1` and `mfmsc`. The `mfml1` node was added in October 2025 as a more powerful login node with 128 logical CPUs and 512GB of memory. The `mfmsc` node is the original login node with 32 logical CPUs and 128GB of memory, which also serves as the Slurm controller and database node. We plan to deprecate the `mfmsc` login node in the future and recommend using the `mfml1` login node instead.

## Slurm user setup
Before you start using Slurm, we recommend you to get access to a bunch of scripts that build on top of Slurm to make certain tasks easier (see [Slurm tools](#5-crusoe-slurm-tools)). To do so, just

   - add the following to your `~/.bash_profile`, `~/.bashrc` or an equivalent file:

     ```bash
     SLURM_USER_TOOL_ROOT=/usr/local/share/slurm-user/
     if [ -d $SLURM_USER_TOOL_ROOT ]; then
       export PATH="${SLURM_USER_TOOL_ROOT}:$PATH"
       source ${SLURM_USER_TOOL_ROOT}/slurm-crusoe.bash_profile
     fi
     ```

## Accounts
Users and working groups need to be registered in Slurm separately, before they can submit jobs. This has to be done when you get your account.

Slurm usernames match Linux usernames. Every user is assigned one or more accounts that correspond to working teams across the MFM group, e.g. `3d`, `mt`, `ct`, `dgm`, `sfxfm`.
**Remember**: If you are member of more than one group then you have to take care to use the right account for the partition of your choice when you submit jobs.

- **Main accounts**
  - `mfm` – members of the MFM parent group

- **Team accounts**
  - `3d`, `ct`, `mt`, `dgm`, `sfxfm`, `ds` – MFM working group accounts

- **User accounts**

You can check which team accounts you belong to and which partitions you have access to by running:
```
$ sua
Account|Partitions
account1|partition1,partition2
account2|partition3
```

## Partitions
Some partitions provide guaranteed compute resources to certain working groups for their primary use. All other users can use these reserved resources when not in use, via a `sharedp` partition for H100 nodes and a `sharedp_l40s` partition for L40S nodes.

Below is a table with current partitions and overall resources assigned to them (see also `sip` output for the current partition status). Note that a compute node can be part of more than one partition.

| Partition | CPUs | GPUs | Limits | Nodes | Use |
|:-:|:-:|:-:|:-:|---|:-:|
| **3d** | 2816 | 26x H100 |  | mfmc[1-20] | 3D team members |
| **ct** | 2816 | 26x H100 |  | mfmc[1-20] | CT team members |
| **dgm** | 2816 | 24x H100 |  | mfmc[1-20] | DGM team members |
| **ds** | 2816 | 12x H100 |  | mfmc[1-20] | DS team members |
| **mt** | 2816 | 26x H100 |  | mfmc[1-20] | MT team members |
| **sfxfm** | 2816 | 26x H100 |  | mfmc[1-20] | SFXFM team members |
| **3d_l40s** | 320 | 7x L40S |  | mfmcl[1-4] | 3D team members |
| **ct_l40s** | 320 | 7x L40S |  | mfmcl[1-4] | CT team members |
| **dgm_l40s** | 320 | 7x L40S |  | mfmcl[1-4] | DGM team members |
| **ds_l40s** | 320 | 5x L40S |  | mfmcl[1-4] | DS team members |
| **mt_l40s** | 320 | 7x L40S |  | mfmcl[1-4] | MT team members |
| **sfxfm_l40s** | 320 | 7x L40S |  | mfmcl[1-4] | SFXFM team members |
| **sharedp** | 2816 | 128x H100 | 24h,REQUEUE | mfmc[1-20] | All users |
| **sharedp_l40s** | 320 | 40x L40S | 72h,REQUEUE | mfmcl[1-4] | All users |
| ~~**cpu**~~ | ~~16~~ | - |  | ~~mfmsc~~ | ~~CPU-only jobs<br>All users~~ |
| ~~**cpuc**~~ | ~~240~~~~ | - | ~~16CPU,128G/node~~ | ~~mfmc[1-20]~~ | ~~CPU-only jobs on GPU nodes<br>All users~~ |
| **reserved** | 2816 | 128x H100 |  | mfmc[1-20] | Advanced reservation only<br>All users |
| **maintenance** | ALL | ALL | - | mfmc[1-20] | Admin user only |
| **maintenance_l40s** | ALL | ALL | - | mfmcl[1-4] | Admin user only |

If no partition is specified for a job, **the default partition is set dynamically to the team account of the current user**. As team partitions require GPU allocation, **`--gres=gpu:1` is allocated by default as well.**

### Team partitions (`3d`, `ct`, `dgm`, `mt`, `sfxfm`, `ds` and their `_l40s` counterparts)
These are partitions that guarantee resources, namely GPUs, for each team. The resources are available at any time to the team account members. Jobs using such resources run by non-team members may be preempted at any time, if a team member launches a job on this partition.

All team partitions have GPUs assigned. To avoid using GPU compute slots for CPU-only jobs, jobs require the number of gpus to be allocated to be specified explicitly, e.g.

```bash
srun --partition=sfxfm --gres=gpu:1 python my_job.py
```

### CPU partitions (`cpu`)

**NOTE**: The `cpuc` partition has been disabled to improve GPU utilization. If you don't have access to other CPU resources, use the L40S GPU partitions instead, even if you don't need GPUs.

CPU partitions are dedicated to jobs that do not use GPUs. Compute nodes that do not have GPUs are included in the `cpu` partition. GPU compute nodes also contribute to a CPU partition, `cpuc`, but only with few of their CPUs. To run CPU jobs on nodes in any of these two partitions use the `--partition=cpu` option, e.g.

```bash
srun --partition=cpu python my_job.py
```

As opposed to the team partition no GPUs can be allocated on CPU-only partitions, so `--gres=gpu:X` will not work, unless X=0.

### Shared partitions (`sharedp` and `sharedp_l40s`)
Shared partitions are meant to be used by any user. At the current time, only `sharedp` is available, with `p` standing for `preemption`. The current setup assigns all team compute nodes to this shared partition, so the entire compute resources in the cluster are available via `sharedp`. However, as soon as a team user requires his/her team's resources, jobs using these resources on `sharedp` will be preempted immediately. In practical terms, this means **any job running on `sharedp` or `sharedp_l40s` can be killed at any time**. So, for long jobs to be able to run on the `sharedp` or `sharedp_l40s` partition, jobs need to be fully resumable, and this is left to the user to be implemented (see [sbatch-restart-template](sbatch-restart-template.script) for a sbatch template). To encourage job rotation, `sharedp` and `sharedp_l40s` also impose a max run time of 24h and 72h respectively, after which the job will be killed. To run a job on the `sharedp` partition:

```bash
srun --partition=sharedp --gres=gpu:1 python my_job.py
```

### Reserved partition (`reserved`)
This partition gathers the entire cluster resources spanning cross-team compute resources. The partition is protected via a reservation system, making it possible to gather any number of GPUs and nodes for one or more users, but for a limited amunt of time. A Slurm administrator is required to create the reservation id for you or your team first. Once you receive receive this id, you can run your jobs as

```bash
srun --reservation=my_reservation_id --partition=reserved --gres=gpu:64 python my_job.py
```

Large scale reservations need to be first discussed across teams, so its use is not automated.

### Maintenance partition (`maintenance`)
This partition is reserved for maintenance tasks. It is not available for user jobs. If you are an admin, and need to run on this partition, please refer to the documentation found in the admin repository.

## Storage Space and Usage

In this section, we will detail the various types of storage available in our cluster, their primary uses, and any limitations associated with each type. Below is a summary table, followed by detailed explanations.

| Storage Type | Location | Size | Purpose | Performance |
|---|---|---|---|---|
| Home | `/home/$USER` | 100GB~/user | Personal storage for small files like scripts, config files, etc. | Slow (20Gbps shared among all users and instances) |
| Shared Disk User | `/scratch2/$USER` | 100GB~/user | Temporary user data | Fast for large files (Up to 20GB/s read, 4GB/s write, 120k IOPS per 100TiB storage size) |
| Shared Disk Group | `/group2/TEAM` | 5TB~/team | Shared group storage mainly for datasets, checkpoints, etc. | Same as above |
| Slurm Job Scratch | `$SLURM_SCRATCH` only accesible in slurm job | 7TB/node | Temporary job-specific storage mainly for cacheing dataset | Very fast (Striped 8x NVMe SSDs, ~40Gbps each) |
| S3 Bucket | AWS S3 (`s3://BUCKET_NAME`) | Varies | Cloud storage | Variable |

### Home
Each user has a personal directory located at `/home/$USER` with a size of around 100Gb/user. This is a persistent storage space accessible from any compute node. The file system has a throughput of only 20Gbps shared among all users and all instances, making it relatively slow.

### Shared Disk

The [shared disk](https://docs.crusoecloud.com/storage/disks/overview#shared-disks) is a managed storage service provided by Crusoe, and mounted at `/music-shared-disk` across all compute nodes and login nodes. The storage size is currently around 125Tb in total (as of 9 Feb 2025), and is shared among all users and groups.

Although it is provided in a reasonable price compared to the BeeGFS storage which we managed ourselves, it is a quite new service as of Feb 2025, and there are often some issues with the service. Please be patient and contact the admins via the `#mfm-crusoe-users` channel on Slack if you encounter any issues.

There are two types of shared disk spaces:

#### Group space

The group space is located at `/group2/TEAM` and is shared storage dedicated to each team account. This space is intended for storing large shared files like datasets and model checkpoints.

**Note**: While there are currently no per-team quota limits (unlike the previous BeeGFS storage), the total storage is shared between all teams. Please be mindful of usage, as hitting the overall storage limit will impact jobs across all teams. Team-specific quotas may be implemented in the future.

#### User space

The user space is located at `/scratch2/$USER` and is a fast storage space for each user. It is quite fast and suitable for storing large temporary files. While the space is intended for temporary data, there is currently no automatic deletion process for old files. It is recommended to store temporary files like pip cache, Hugging Face checkpoints, and Singularity cache/images (by default, `SINGULARITY_CACHEDIR` is set to this space) here, while keeping smaller, critical files in your home directory, an S3 bucket, or a GitHub repository.

**Note**: Quota is not applied to the user space yet as well. Please be mindful of usage, as hitting the overall storage limit will impact jobs across all teams.

### Slurm Job Scratch
Each compute server holds fast NVMe storage under `/data`. Each Slurm job creates a directory for your user and job (`/data/${USER}/slurm-scratch-${SLURM_JOBID}-${SLURM_JOB_NAME}`) which is accessible as the `SLURM_SCRATCH` environment variable. This directory will be automatically deleted upon job finish or failure. You can run the following command inside your `sbatch` script to safely copy your datasets:

```
srun cp -rf --no-preserve=mode /group/mfm/data/MY_DATASET $SLURM_SCRATCH
```

### S3 Bucket
Every user on Crusoe is set up to use S3. The platform team configures AWS credentials for each user, allowing access to S3 buckets from any Crusoe instance. If your team has requested an S3 bucket accessible from your EC2 instance or elsewhere, it is available for use. However, some users onboarded before our Slurm integration might need to retrieve their credentials from `/lhome/USERNAME/.aws` on their onboarding instance.

### Quota

You can see the available usage and quota for "Home" with the `show_quota` command as follows.

```
$ show_quota
# --------------------------------------------------------------------
# NOTE: Quota updates every 10 minutes. Displayed info may be delayed.
# NOTATION: *Limit = single value (hard) or soft/hard format,
#           I* = inode used/limit, Checked = quota checked time
# --------------------------------------------------------------------

[Home]
| User            | ID   | Used   | Limit     | IUsed | ILimit | Checked              |
| --------------- | ---- | ------ | --------- | ----- | ------ | -------------------- |
| takuya.narihira | 1008 | 404.1M | 100G/100G | 4.0k  | 0/0    | 2024-07-11 03:00:01  |
```

We are considering to apply quota to the shared disk group and user spaces as well. 

You can ask admins to change the quota for your team and user space.

## Login nodes
The cluster provides only one login node `mfmsc` for now. All users are supposed to use this login node to access the cluster, edit files, queue jobs, etc.

Since this login node is shared among all users, and is not very powerful, please avoid running heavy tasks on the login node.

## 1. Slurm Basic usage
To submit a job via Slurm, you need to use `srun`, `sbatch` or `salloc` commands. `srun` is the most straightforward way to run a task. `srun` finishes when your command exits and it redirects stdout/stderr to your console, so it can be used the same way you would run your command without Slurm.

To submit a `command`, just run:

```
srun echo "hello"
```

`echo "hello"` will run on a compute node and its output will be redirected to the console of the server you ran `srun` from, just as if you ran it locally.

`srun` accepts many options that control what resources are allocated to your job. These options are specified as arguments to `srun`, `sbatch` and `salloc`: `srun --option1=value1 --option2=value2 command arg1 arg2 ...`. Check the official [Slurm Quick Reference](https://slurm.schedmd.com/pdfs/summary.pdf) for details about Slurm submit options and other commands.

- Submit partitions: `--partition=dgm,sharedp

- Set job name: `--job-name=this_is_my_job`

- Submit account (which identity should be used to submit the job): `--account=dgm`

- Choose node(s) where job should run: `--nodelist=mfmc1`

- Specify the number and/or type of GPUs allocated: `--gres=gpu:1`, `--gpus-per-node=1`, `--gres=gpu:h100:2` or `--gres=gpu:2,gpu_mem:24000`

- Specify the number of parallel tasks: `--ntasks=1`

- Specify the size of the scratch disk: `--gres=scratch:500000`

- Maximum duration of the job: `--time=DD-HH`, `--time=HH:MM:SS`, `--time=MM:SS`

**Note**: In GPU partitions, you no longer need to manually allocate CPUs or memory. These are automatically set based on the number of GPUs specified via the `--gres` option. CPU cores and system memory are evenly distributed across GPUs, with a small portion of memory reserved for system use.

**The job allocation defaults are `--nodes=1` and `--ntasks=1` in GPU partitions. If no partition is specificed, the job partition is set dynamically to the team account of the current user. As team partitions require GPU allocation, `--gres=gpu:1` is being allocated by default as well.**

Note that some partitions have some specificities that you need to account for at submission time

  - No GPUs can be allocated on `cpu` partition. Simply omitting `--gres=gpu:X` will do.

  - At least one GPU needs to be allocated in partitions providing GPUs, i.e. `3d`, `mt`, `ct`, `dgm`, `sfxfm`, `sharedp`. Always specifically add the number of requested GPUs with `--gres=gpu:X` with `X>0`.

## 2. Slurm batch scripts
Slurm submit arguments can be stored in the header of a `bash` script that is run via the `sbatch` command. Jobs submitted via `sbatch` run asynchronously (they exit as soon as the job is submitted) and have the advantage that they can be requeued upon preemption (only in certain partitions). `sbatch` scripts need stdout/stderr be redirected to output and error log files via the `--output` and `--error` arguments. Basically, the same options above for `srun` commands are available for `sbatch` jobs.

```
sbatch my_batch_script.sh
```

with `my_batch_script.sh`:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.log
#SBATCH --error=slurm-logs/error.log
#SBATCH --gres=gpu:h100:4
#SBATCH --account=dgm
#SBATCH --requeue

command arg1 arg2 …
```

Batch jobs allow two additional arguments, on top of `srun` arguments:

- Specify the stdout/stderr log files: `--output=slurm-logs/out.log` and `--error=slurm-logs/error.log` . Note that the parent directory of the log files, `slurm-logs/` in the example, needs to be created outside of the batch script itself. If the directory does not exist the sbatch job will fail without message in either log files.

- Allow the job to be requeued upon preemption or when its time limit has been reached: `--requeue`


Here are a couple of **use case command examples**:

  - #### Single-GPU training
```
#!/bin/bash
#SBATCH --partition=dgm,sharedp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log
#SBATCH --gres=gpu:1
#SBATCH --account=dgm
#SBATCH --requeue

command arg1 arg2 …
```


  - #### Multi-GPU (multi-process) training, single node
```
#!/bin/bash
#SBATCH --partition=dgm,sharedp
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log
#SBATCH --gres=gpu:2,gpu_mem:10000
#SBATCH --account=dgm
#SBATCH --requeue

command arg1 arg2 …
```

## 3. Slurm interactive Jobs
Interactive jobs are submitted via the `sbash` script in the provided Slurm tools above. They provide an interactive `bash` shell environment based on the allocation options above (see `srun`). Interactive jobs have a default maximum time of 24h in the Crusoe cluster. You should **release interactive resources** as soon as they are not used anymore. Just type `exit` to exit the allocated shell.

## 4. Slurm user commands
You can interact with your jobs in several ways:

- `squeue` – list submitted/running jobs
- `squeue –u username` – list a specific user's jobs
- `scancel jobid` – cancel a running/pending job
- `scontrol show job job_id` – show details about running/pending job
- `sinfo` : information about nodes and partitions

Jobs go through a series of states as they run on the Slurm cluster. These are the states you can see using the Slurm `squeue` command:

- PD – Pending
- R – Running
- S – Suspended
- CG – Completing
- CD – Completed
- PR – Job terminated due to preemption
- SI – Signaling (job has been signaled, e.g. before termination)


## 5. Crusoe Slurm tools

A number of tools that work on top of base Slurm commands are provided here to ease cluster usage. Make sure you have cloned the Crusoe admin repo and have set it up as described in [Setup Slurm on Crusoe](#slurm-user-setup).

## Commands:

#### `sua` (Slurm User Accounts)
Shows the team accounts and partitions available to the current user. This command helps users identify which team accounts they belong to and which partitions they have access to.

Example output:
```
$ sua
Account|Partitions
account1|partition1,partition2
account2|partition3
```

This command is particularly useful when you need to:
- Verify which team accounts you belong to
- Check which partitions you have access to
- Determine the correct account to use when submitting jobs to specific partitions

Remember that if you are a member of multiple teams, you need to specify the correct account when submitting jobs to team-specific partitions.

#### `sq` (Slurm Queue information)
Short for `squeue` with extended fields, including allocated cpus, memory and gpus per job. Shows all pending/running jobs in the cluster.

Options:
  - `--gpu` : show GPU jobs only
  - `--cpu` : show CPU jobs only
  - `--user username_pattern` : show jobs for this username
  - `--partition partition_pattern` : show jobs for this partition
  - `--nodename nodename_pattern` : show jobs for this nodename
  - `--job job_pattern` : show jobs with this jobid or jobname
  - `--sort user|partition|nodename|job` : sort output by different criteria

```
    JobId     JobName     Partition         User       State      Time       Nodes CPU GPU Mem     NodeList(Reason)
     208      train2        sfxfm       marc.ferrasfo    R        0:09           1   1   4 100M          mfmc1
     207      train1        sfxfm       marc.ferrasfo    R        0:16           1   1   2 100M          mfmc1
```

#### `squ` (Slurm Queue for current User)
Same as `sq` but for your own jobs.

#### `sc` (Slurm Cancel)
Short for `scancel`. It accepts a JobId or JobName.

#### `sin` (Slurm Info on compute Nodes)
Shows extended information about compute nodes. Note that, to save space, only GPU partitions are shown in the Partition field. Run `sip` to check all partitions.

  - `-c|--color` : show CPU,Mem and GPU usage with colors

```
   sin
Nodename        Partition              GPUType             Scratch  CPUUse               MemUse        GPUUse              State
mfmc1           sfxfm,cpuc             8,h100,79.6G           6.3T   4/176 (2%,  0)    0/850(0%)   3/8 (38%)[0-2]      mix
mfmc13          3d,cpuc                8,h100,79.6G           6.3T   0/176 (0%,  0)    0/850(0%)   0/8  (0%)           idle
mfmc14          ct,cpuc                8,h100,79.6G           6.3T   0/176 (0%,  0)    0/850(0%)   0/8  (0%)           idle
mfmc15          mt,cpuc                8,h100,79.6G           6.3T   0/176 (0%,  0)    0/850(0%)   0/8  (0%)           idle
mfmc16          ds,cpuc                8,h100,79.6G           6.3T   0/176 (0%,  0)    0/850(0%)   0/8  (0%)           idle
mfmc2           dgm,cpuc               8,h100,79.6G           6.3T   0/176 (0%,  0)    0/850(0%)   0/8  (0%)           idle
mfmsc           cpu                                                  2/32  (6%,  0)    0/113(0%)                       mix
```

#### `sip` (Slurm Info on Partitions)
Shows extended information about Slurm partitions.

  - `-c|--color` : show CPU and GPU usage with colors

```
   sip
 Partition    Avail         Limits          Nodes       CPUUse           GPUUse       NodeList
    3d         up                             1        0/176(0%)        0/8 (0%)      mfmc13
    cpu        up                             1        0/16 (0%)                      mfmsc
   cpuc        up       16CPU,125G/node       6        0/96 (0%)                      mfmc[1-2,13-16]
    ct         up                             1        0/176(0%)        0/8 (0%)      mfmc14
    dgm        up                             1        0/176(0%)        0/8 (0%)      mfmc2
    ds         up                             1        0/176(0%)        0/8 (0%)      mfmc16
    mt         up                             1        0/176(0%)        0/8 (0%)      mfmc15
 reserved      up                             6       0/1056(0%)        0/48(0%)      mfmc[1-2,13-16]
   sfxfm       up                             1        0/176(0%)        0/8 (0%)      mfmc1
  sharedp      up         24h,REQUEUE         6       0/1056(0%)        0/48(0%)      mfmc[1-2,13-16]
```

#### `sig` (Slurm Info on GPUs)
Shows extended information about GPUs and GPU usage per node, per GPU type and overall.

  - `-c|--color` : show GPU usage with colors

```
    sig
   Nodename            GPUType          GPUMem          GPUUse         GPUAvail
    mfmc1               h100             79.6G    3/8 (38%)[0-2]          5
    mfmc13              h100             79.6G    0/8  (0%)               8
    mfmc14              h100             79.6G    0/8  (0%)               8
    mfmc15              h100             79.6G    0/8  (0%)               8
    mfmc16              h100             79.6G    0/8  (0%)               8
    mfmc2               h100             79.6G    0/8  (0%)               8
    mfmsc

 GPUType           GPUMem             GPUUse      GPUAvail
   h100             79.6G            3/48(6%)       45

 GPUUse(Overall)   GPUAvail
    3/48 (6%)        45
```

#### `siu` (Slurm Info on Users)
Shows resources being used on a per-user basis.

  - `-s|--sort user|job|cpu|gpu|mem` : sort output by different criteria
  - `-r|--show-real-names` : Show user's real names

```
    siu
     Username        Jobs    CPU    GPU    Memory            Partition          Nodenames
  marc.ferrasfont      3      3      3      300M         cpu,sfxfm,sharedp      mfmc[1]
```

#### `slog` (Slurm Log)
Shows a job's stdout/stderr log file from its JOB_ID or JOB_NAME, e.g. taken from the squeue command. It also outputs the name of the found logfile so you can locate it manually as well. Some notes abut `slog`:

   - By default, `slog` will search for log files in `/home/${USER}/slurm_logs`. Slurm stdout/stderr log files should be formatted with `sbatch` options:

      ```bash
         --output=/home/${USER}/slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out
         --error=/home/${USER}/slurm_logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err
      ```

   - Use `-sld my_slurm_log_directory` to use an alternative location, or change its defualt in the Python code

- `slog JOB_ID` shows the corresponding log file (`cat logfile`).

- `slog -f JOB_ID` shows the last lines of the log file and is updated as it is written (`tail -f logfile`).

- `slog -l JOB_ID` shows the last part of the log and allows you to move up/down the content (`less logfile`).

- `slog -e JOB_ID` shows the stderr log file of the JOB. `-f` and `-l` options can be used as well.

#### `sbash` (Slurm Bash)
Allocates resources for an interactive Slurm session and runs a Bash shell on a specific compute node. Interactive jobs are limited to 12h running time. Upon session start, a banner appears as a reminder of the resources having been allocated, and not available to other users. **Make sure you exit the interactive job as soon as the resources are not used anymore by typing `exit`**.

```
marc.ferrasfont@mfmsc:[~]$ sbash --partition=sfxfm --gpus=1 mfmc1

**************************************** INTERACTIVE SLURM JOB! ****************************************

     The allocated resources are exclusively granted to you until you manually exit this session.
                     Run 'exit' as soon as you do not need the resources anymore.

                                    Allocated nodes: mfmc14
                                Allocated resources: nodes=1,cpu=1,gpu=1,mem=100M

********************************************************************************************************

marc.ferrasfont@mfmc14 [~] Allocated nodes=1,cpu=1,gpu=1,mem=100M | Time 0:01 | Run 'exit' to release GPUs!
  ll
total 36
drwxrwxr-x 10 marc.ferrasfont marc.ferrasfont   263 Jun  5 11:55 ./
drwxrwxr-x  6 marc.ferrasfont marc.ferrasfont   186 Jun  5 12:54 ../
-rw-rw-r--  1 marc.ferrasfont marc.ferrasfont 17209 Jun  4 08:06 README.md

marc.ferrasfont@mfmc14 [~] Allocated nodes=1,cpu=1,gpu=1,mem=100M | Time 1:46 | Run 'exit' to release GPUs!
  exit
logout
marc.ferrasfont@mfmsc:[~]$

```

#### `sjob` (Slurm Job info)
Short for `scontrol show job JOB_ID`. Shows information about a submitted or running job. If your job is in pending **PD** state, you can check `Reason` here to know why.

```
 sjob sleep
JobId=205 JobName=sleep
   UserId=marc.ferrasfont(1006) GroupId=marc.ferrasfont(1006) MCS_label=N/A
   Priority=1946 Nice=0 Account=sfxfm QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0
   RunTime=00:00:10 TimeLimit=UNLIMITED TimeMin=N/A
   SubmitTime=2024-05-01T12:28:41 EligibleTime=2024-05-01T12:28:41
   AccrueTime=Unknown
   StartTime=2024-05-01T12:28:41 EndTime=Unknown Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2024-05-01T12:28:41 Scheduler=Main
   Partition=sfxfm AllocNode:Sid=localhost:252271
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=mfmc1
   BatchHost=mfmc1
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=1,mem=100M,node=1,billing=1,gres/gpu=1,gres/gpu:h100=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryCPU=100M MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=sleep
   WorkDir=/mnt/auto/shared/user/marc.ferrasfont/project_mfm_crusoe_admin
   Power=
   TresPerJob=gres:gpu:1
```

#### `sres` (Slurm Resources)
Shows all of the resources in the cluster (also aggregate) including nodes, cores, cpus, gpus, scratch disk size.

```
    sres
   Nodename      Cores   CPUs   CPUMem[Gb]   GPUs    GPUType:GPUMem[Gb]   Scratch[Tb]
    mfmc1         88      176       850        8         h100:79.6            6.3
    mfmc10        88      176       850        8         h100:79.6            6.3
    mfmc13        88      176       850        8         h100:79.6            6.3
    mfmc14        88      176       850        8         h100:79.6            6.3
    mfmc15        88      176       850        8         h100:79.6            6.3
    mfmc16        88      176       850        8         h100:79.6            6.3
    mfmc2         88      176       850        8         h100:79.6            6.3
    mfmc4         88      176       850        8         h100:79.6            6.3
    mfmc5         88      176       850        8         h100:79.6            6.3
    mfmc6         88      176       850        8         h100:79.6            6.3
    mfmsc          8      16        84         -

   Nodes    Cores    CPUs    CPUMem[Gb]   GPUs   GPUMem[Gb]    Scratch[Tb]
    11       888     1776       8588       80        796           63
```
