# Explorer Cluster SSH Tunnel

SSH from your laptop directly to a Northeastern Explorer compute node for interactive work, port forwarding, and remote development.

## How It Works

```
Local machine  ssh explorer_tunnel
  └─> login node (explorer)
        └─> squeue finds a running job named tunnel → compute node hostname
              └─> nc node:2222 → user-space sshd on the compute node
```

Your local `~/.ssh/config` must include something like:

```
Host explorer
  HostName login.explorer.northeastern.edu
  User chen.yiwe

Host explorer_tunnel
  User chen.yiwe
  ProxyCommand ssh explorer "nc $(squeue --me --name=tunnel --states=R -h -O NodeList) 2222"
  StrictHostKeyChecking no
```

**Important:** the SLURM job must be named `tunnel`, and `sshd` must listen on port `2222` on the compute node.

## Files

| File | Description |
|------|-------------|
| `tunnel.sh` | Main entry point; run all commands from here |
| `config.env` | Default SLURM / SSH settings; override via environment variables |
| `start_sshd.sh` | Starts user-space sshd inside the SLURM allocation |
| `sbatch_tunnel.sh` | Template for manual `sbatch` on Explorer |

Scripts are automatically rsync'd to `~/cluster-tunnel/` on Explorer.

## Prerequisites

- You can `ssh explorer` to the Explorer login node from your laptop
- `~/.ssh/config` defines both `explorer` and `explorer_tunnel`
- Your local public key (default: `~/.ssh/id_rsa.pub`) is in `~/.ssh/authorized_keys` on Explorer

## Quick Start

From the project root:

```bash
# 1. One-time setup (host key, script sync, public key check)
bash bash_scripts/cluster-tunnel/tunnel.sh setup

# 2. Submit a background tunnel job
bash bash_scripts/cluster-tunnel/tunnel.sh submit

# 3. Check whether the tunnel is ready
bash bash_scripts/cluster-tunnel/tunnel.sh status

# 4. Connect (waits until the job is Running and port 2222 is reachable)
bash bash_scripts/cluster-tunnel/tunnel.sh connect
```

Once connected, you can also use:

```bash
ssh explorer_tunnel
```

## Commands

| Command | Description |
|---------|-------------|
| `setup` | One-time remote prep: public key, host key, script sync |
| `sync` | Push scripts to `~/cluster-tunnel/` only |
| `submit` | Submit a background tunnel job via `sbatch` |
| `interactive` | Request a node with `srun` and run sshd in the foreground (holds a terminal) |
| `status` | Show SLURM job status and probe port 2222 |
| `connect` | Wait for the tunnel, then `ssh explorer_tunnel` |
| `stop` | Cancel all SLURM jobs named `tunnel` |
| `help` | Print usage |

Aliases: `interactive_tunnel` and `srun` are equivalent to `interactive`.

## Usage Modes

### Mode A: Background job (recommended)

Good for long-running access; connect from your laptop at any time.

```bash
bash bash_scripts/cluster-tunnel/tunnel.sh submit
bash bash_scripts/cluster-tunnel/tunnel.sh connect
```

Job logs are written on Explorer as `~/cluster-tunnel/tunnel-<jobid>.out` and `.err`.

### Mode B: Interactive

Good for debugging or short sessions. Requires two terminals:

```bash
# Terminal 1: keep this running
bash bash_scripts/cluster-tunnel/tunnel.sh interactive

# Terminal 2: connect
bash bash_scripts/cluster-tunnel/tunnel.sh connect
# or
ssh explorer_tunnel
```

## Custom SLURM Resources

Defaults live in `config.env`:

| Variable | Default |
|----------|---------|
| `SBATCH_PARTITION` | `jiang` |
| `SBATCH_GRES` | `gpu:a6000:1` |
| `SBATCH_MEM` | `32G` |
| `SBATCH_TIME` | `8:00:00` |
| `SBATCH_NTASKS` | `8` |
| `SBATCH_NODES` | `1` |

Override for a single run:

```bash
SBATCH_TIME=12:00:00 SBATCH_MEM=64G \
  bash bash_scripts/cluster-tunnel/tunnel.sh submit
```

For permanent changes, edit `config.env`.

## Custom SSH Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_LOGIN_HOST` | `explorer` | Local SSH config host for the login node |
| `SSH_TUNNEL_HOST` | `explorer_tunnel` | Local SSH config host for the tunnel |
| `TUNNEL_JOB_NAME` | `tunnel` | SLURM job name; must match ProxyCommand |
| `TUNNEL_PORT` | `2222` | sshd port on the compute node |
| `CONNECT_WAIT_SECS` | `600` | Max wait time for `connect` |
| `CONNECT_POLL_SECS` | `5` | Poll interval for `connect` |

If you change `TUNNEL_PORT` or `TUNNEL_JOB_NAME`, update the `explorer_tunnel` entry in `~/.ssh/config` as well.

## Manual Use on Explorer

After scripts are synced, you can also run on the login node:

```bash
cd ~/cluster-tunnel
sbatch sbatch_tunnel.sh
squeue --me --name=tunnel
```

## Troubleshooting

### `Ncat: You must specify a host to connect to`

No SLURM job named `tunnel` is in the Running (`R`) state. Run `status`, or start the tunnel with `submit` / `interactive`.

### Job name is `bash` instead of `tunnel`

Only jobs named `tunnel` are picked up by `explorer_tunnel`. Use this script to submit, or pass `--job-name=tunnel` explicitly.

### `Port 2222 on <node> is not reachable`

The allocation exists but sshd is not up yet. Wait a few seconds and retry `status`. If it persists, check `tunnel-<jobid>.err`.

### `Permission denied (publickey)`

Re-run `setup`, or append your local `~/.ssh/id_rsa.pub` to `~/.ssh/authorized_keys` on Explorer.

### `ERROR: Unable to locate a modulefile for 'gcc/10.1.0'`

Your Explorer `~/.bashrc` loads a gcc module that no longer exists. This does not block the tunnel, but you can run `module avail gcc` on Explorer and update `.bashrc`.

### Port 2222 already in use

Use a different port (update both `config.env` and `~/.ssh/config`):

```bash
TUNNEL_PORT=2223 bash bash_scripts/cluster-tunnel/tunnel.sh submit
```

## Stopping the Tunnel

```bash
bash bash_scripts/cluster-tunnel/tunnel.sh stop
```

Or on Explorer:

```bash
scancel --name=tunnel --user=$USER
```
