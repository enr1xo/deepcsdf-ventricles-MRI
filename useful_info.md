




# Full Guide: Slurm, PyTorch Lightning, DataLoader, and Concurrent Training

## 1️⃣ Slurm Resource Parameters

Example Slurm directives:

```bash
#SBATCH --cpus-per-gpu=6    # CPU cores allocated per GPU
#SBATCH --mem=16G            # RAM allocated per job
#SBATCH --gres=gpu:1         # GPU allocation
```

## 2️⃣ PyTorch DataLoader (`num_workers`)

- `num_workers=N` spawns N subprocesses per training process for parallel data loading.
- Each worker consumes roughly:
  - 1 CPU core
  - 50–200 MB RAM depending on dataset size
- **Rule of thumb:** `num_workers <= cpus-per-gpu` for best utilization.
- `num_workers=0`:
  - Loads data inside the main process
  - Avoids conflicts when running multiple processes concurrently
  - Slightly slower but safe for parallel experiments

---

## 3️⃣ GPU Memory Considerations

- Each training process uses GPU memory depending on model size and batch size.
- Multiple jobs on the same GPU share memory. Too many jobs → OOM.
- Example:
  - A100 GPU with 45 GB
  - Each job ~1.1 GB
  - Maximum safe concurrent jobs ≈ 40

---

## 4️⃣ Python Multiprocessing (`spawn`)

- Use `multiprocessing.set_start_method('spawn')` for CUDA safety.
- Each spawned process re-imports the script.
- If `num_workers>0`, each child spawns its own DataLoader workers → potential conflicts and crashes.
- **Solution:** use `num_workers=0` for manual concurrent processes in one Python session.

### Why `num_workers=0` is recommended when using multiprocessing with `spawn`

#### Multiple separate Python scripts
- Each script (`python train.py`) runs in its **own independent OS process**.
- Each script imports PyTorch, Lightning, and your modules fresh.
- DataLoader workers are independent per script.
- OS schedules them independently; `num_workers > 0` works fine.
- No interference or conflicts between scripts.

#### Multiprocessing within a single Python script (`spawn`)
- The parent process spawns child processes, which **re-import all modules**.
- Each child tries to start its own DataLoader workers if `num_workers > 0`.
- This creates a **process tree**:

parent
├─ child 1
│ ├─ DataLoader worker 1
│ ├─ DataLoader worker 2
├─ child 2
│ ├─ DataLoader worker 1
│ ├─ DataLoader worker 2



- Quickly multiplies the number of processes, leading to:
  - Worker crashes
  - Resource exhaustion
  - `DataLoader worker exited unexpectedly` errors

#### Why `num_workers=0` fixes it
- Each child loads data **sequentially in its own process**.
- Avoids spawning additional subprocesses for workers.
- Slightly slower batch loading, but safe and stable.
- All data still loads correctly; no interference between training processes.

---

## 5️⃣ Concurrency Strategies

**Option A: Python multiprocessing**

- Launch multiple `train()` processes from one script.
- Must use `num_workers=0`.
- Pros: flexible, fine-grained control.
- Cons: slower DataLoader.

**Option B: Slurm (`sbatch`)**

- Each job is isolated with dedicated CPUs and memory.
- Can safely use `num_workers = cpus-per-gpu`.
- Slurm queues jobs if resources unavailable.
- Pros: fully utilizes CPU, safer for high `num_workers`.
- Cons: less interactive, slightly more overhead.

---

## 6️⃣ Recommended Practice for Maximum Speed + Stability

- Large GPUs (A100, H100):
  - Slurm jobs: request CPUs matching intended `num_workers` per job.
  - Manual multiprocessing: set `num_workers=0`.
  - Memory: slightly higher than expected usage per job.
- Prefer Slurm for multiple jobs to safely allow more `num_workers` per job.
- Manual multiprocessing with `num_workers=0` is safe for many concurrent jobs.

---

## 7️⃣ Quick Rules of Thumb

| Parameter                  | Recommendation                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| `--cpus-per-gpu`           | ≥ `num_workers` (if running single job per GPU via Slurm)                     |
| `--mem`                    | Slightly more than expected job usage; Slurm queues if not enough              |
| `num_workers`              | 0 for concurrent Python processes; =cpus-per-gpu for dedicated Slurm jobs     |
| GPU memory per job         | Ensure `MAX_CONCURRENT_JOBS * per_job_GPU_mem <= total_GPU_mem`              |
| Python multiprocessing     | Use `spawn` to avoid CUDA conflicts                                           |
| DataLoader performance     | Slightly slower with `num_workers=0`, but safe for concurrent jobs            |
| Slurm queuing              | Prevents OOM; jobs wait until resources are free                               |