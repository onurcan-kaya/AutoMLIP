# AutoMLIP

Automated training, finetuning, and distillation of machine learning interatomic potentials (MLIPs) with active learning. Supports MACE and NequIP backends. Uses Quantum ESPRESSO for DFT labelling. Runs on PBS clusters or a local workstation.

**Status**: Under active development. Bug reports welcome.

---

## What it does

AutoMLIP builds interatomic potentials from scratch or by finetuning foundation models, using as little human intervention as possible. You describe your chemical system, point it at a DFT setup, and it handles:

- Random structure generation with optional foundation-model pre-relaxation
- DFT labelling via Quantum ESPRESSO (local or PBS)
- Training a committee of MACE or NequIP models
- Active learning: MD sampling -> committee disagreement -> DFT labelling -> retraining
- Distillation: compressing a large trained teacher into a small, fast student model
- Checkpointing and resume across walltime kills

Everything is configured through a single Python `Config` object - no YAML files, no CLI flags.


## Install

```
pip install -e .
```

Backend-specific dependencies are not pulled in automatically:

```
pip install mace-torch    # for MACE backend
pip install nequip        # for NequIP backend
```

You also need `pw.x` (Quantum ESPRESSO) installed separately for DFT labelling. Python >= 3.9. GPU strongly recommended for training.


## How it works

### Five modes

| Mode | What it does | Needs data? | Needs DFT? | Active learning? |
|---|---|---|---|---|
| `finetune` | Finetune a foundation model on your data | Yes | No | No |
| `train` | Train from scratch on your data | Yes | No | No |
| `finetune_al` | Finetune with active learning | Optional seed | Yes | Yes |
| `train_al` | Train from scratch with active learning | Optional seed | Yes | Yes |
| `distill` | Compress a teacher into a small student | Seed structures | Optional | No |

### The active learning loop (`finetune_al` / `train_al`)

1. **Seed data**: load from file, or generate random structures (RSS) and label with DFT.
2. **Train committee**: train N models with different random seeds on the same data.
3. **MD sampling**: run Langevin MD with the lead model to explore configuration space.
4. **Committee disagreement**: predict with all members, rank snapshots by mean force standard deviation (QBC).
5. **Select and label**: send the most-disagreed snapshots to DFT.
6. **Extend and retrain**: add labelled structures to the training set, retrain the committee.
7. **Convergence**: stop when RMSE is below tolerance, QBC selects nothing new, or max iterations reached.

### RSS pre-relaxation

Random placement can produce unphysical configurations that waste DFT compute and cause SCF failures. Setting `rss.prerelax_model` runs a short LBFGS relaxation with a foundation model (e.g. MACE-MP) before DFT. Structures where relaxation fails or the energy is an outlier (beyond 3sigma) are discarded. This typically removes 10-30% of RSS structures and significantly reduces DFT failure rates.

### Distillation

A trained teacher committee (or single model) labels a large set of cheaply generated synthetic structures (rattle-relax family trees grown from seed structures, plus isolated dimers for short-range coverage). Structures where the committee disagrees above a force-std threshold are either sent to QE for a true DFT label or dropped. A small student model is then trained from scratch on the combined labels.

### Range-valued config

Many config fields accept either a single value (fixed) or a `(min, max)` tuple. When a tuple is given, the value is drawn uniformly at random each time it is used. This applies to: `system.density`, `system.n_atoms`, `system.composition` (per-element), `active.md_temp`, `active.md_steps`, and several distillation fields (`rattle_sigma_pos`, `rattle_sigma_cell`, `relax_steps`, `temperature`). A single value is used as is.

```python
density=2.0          # fixed: every structure gets 2.0 g/cm3
density=(1.5, 3.5)   # range: each structure draws a density uniformly from [1.5, 3.5]
```

### Checkpointing and resume

The AL modes save a checkpoint after every iteration: `checkpoint.json` (pipeline state), `training_data.extxyz`, and `validation_data.extxyz`. Set `start="resume"` to continue an interrupted run. Resume skips DFT jobs whose `pw.out` already parses, lets MACE pick up from its latest checkpoint, and skips synthetic generation if the distillation set is already written.

`start="resume"` on an empty `work_dir` warns and starts fresh. `start="new"` on top of an existing run warns and overwrites as it goes. Neither hard-stops.


## Recipes

### Recipe 1: Finetune MACE-MP on existing data (simplest case)

You already have a labelled `.extxyz` file. Finetune a MACE foundation model on it.

```python
from automlip import Config, Pipeline
from automlip.config import TrainerConfig

Pipeline(Config(
    mode="finetune",
    data_path="/path/to/data.extxyz",
    work_dir="./my_finetune",
    trainer=TrainerConfig(
        backend="mace",
        device="cuda",
        foundation_model="small",  # "small", "medium", "large", or a local path
        committee_size=3,
        finetune_strategy="naive",
        e0s="estimated",           # re-estimate E0s against the foundation model
        mace_epochs=500,
        mace_lr=0.01,
        mace_batch_size=10,
    ),
)).run()
```

The pipeline writes all models to `work_dir/models/`, one subdirectory per committee member. MACE handles its own train/validation split via `mace_valid_fraction` (default 0.1).


### Recipe 2: Train from scratch on existing data

Same as Recipe 1 but without a foundation model. Uses `E0s="average"` by default.

```python
Pipeline(Config(
    mode="train",
    data_path="/path/to/data.extxyz",
    work_dir="./my_training",
    trainer=TrainerConfig(
        backend="mace",
        device="cuda",
        foundation_model=None,     # train from scratch
        committee_size=3,
        mace_r_max=5.0,
        mace_num_channels=128,
        mace_max_L=1,
        mace_epochs=500,
        mace_lr=0.01,
        mace_batch_size=10,
    ),
)).run()
```


### Recipe 3: No data, finetune with active learning

You describe the system and let the pipeline generate initial structures via RSS, label with DFT, and iterate.

```python
from automlip import Config, Pipeline
from automlip.config import *

Pipeline(Config(
    mode="finetune_al",
    work_dir="./abn_finetune_al",
    system=SystemConfig(
        elements=["B", "N"],
        composition={"B": 0.5, "N": 0.5},
        n_atoms=100,
        density=2.1,
    ),
    rss=RSSConfig(
        n_structures=20,
        prerelax_model="mace_mp",   # pre-relax with MACE-MP-0
        prerelax_fmax=5.0,
        prerelax_steps=50,
    ),
    dft=DFTConfig(
        pseudopotentials={"B": "B.UPF", "N": "N.UPF"},
        pseudo_dir="/path/to/pseudo/",
        ecutwfc=80.0,
        kpoints=[1, 1, 1],
    ),
    trainer=TrainerConfig(
        backend="mace",
        device="cuda",
        foundation_model="small",
        committee_size=3,
    ),
    active=ActiveLearningConfig(
        max_iterations=30,
        md_temp=300.0,
        md_steps=5000,
    ),
    scheduler=SchedulerConfig(backend="local", cores=8),
)).run()
```


### Recipe 4: Train from scratch with active learning on PBS

The real production recipe for an HPC cluster. The driver runs on a GPU node and submits DFT jobs as PBS jobs.

```python
from automlip import Config, Pipeline
from automlip.config import *

cfg = Config(
    mode="train_al",
    start="new",                  # set to "resume" after a walltime kill
    work_dir="/scratch/my_run",
    seed=42,
    cleanup_dft_scratch=True,     # delete QE wavefunctions after each job
    system=SystemConfig(
        elements=["B", "N", "O"],
        composition={"B": 0.45, "N": 0.45, "O": 0.10},
        n_atoms=96,
        density=2.1,
    ),
    dft=DFTConfig(
        pseudopotentials={
            "B": "B.pbe-n-kjpaw_psl.UPF",
            "N": "N.pbe-n-kjpaw_psl.UPF",
            "O": "O.pbe-n-kjpaw_psl.UPF",
        },
        pseudo_dir="/path/to/pseudopotentials",
        ecutwfc=60.0,
        ecutrho=480.0,        # 8x ecutwfc for PAW
        kpoints=[2, 2, 2],
    ),
    trainer=TrainerConfig(
        backend="mace",
        device="cuda",
        committee_size=3,
        mace_r_max=5.0,
        mace_num_channels=128,
        mace_max_L=1,
        mace_epochs=400,
        mace_lr=0.01,
        mace_batch_size=10,
    ),
    active=ActiveLearningConfig(
        max_iterations=10,
        md_temp=1000.0,
        md_steps=2000,
        md_interval=50,
        qbc_select_fraction=0.1,
        max_new_per_iter=30,
        rmse_energy_tol=0.005,    # eV/atom
        rmse_force_tol=0.10,     # eV/Angstrom
    ),
    rss=RSSConfig(n_structures=30),
    scheduler=SchedulerConfig(
        backend="pbs",
        cores_per_dft_job=8,
        mpi_command="mpirun",
        dft_executable="pw.x",
        pbs_max_concurrent=4,
        walltime="06:00:00",
        queue="batch",
        account="my_allocation",
        modules=["module load quantum-espresso"],
        poll_interval=60.0,
    ),
)

if __name__ == "__main__":
    Pipeline(cfg).run()
```

Submit the driver inside a GPU job:

```bash
#!/bin/bash
#PBS -N automlip_driver
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=24:00:00
#PBS -q gpu_queue
#PBS -A my_allocation
#PBS -j oe

cd "$PBS_O_WORKDIR"
# module load conda && conda activate automlip
python my_run.py
```


### Recipe 5: Finetune with multihead replay

MACE multihead finetuning replays a pretraining dataset alongside your target data.

```python
trainer=TrainerConfig(
    backend="mace",
    device="cuda",
    foundation_model="/path/to/mace_foundation.model",
    finetune_strategy="multihead",
    pt_train_file="mp",              # replay on Materials Project data, or a local .xyz
    pseudolabel_replay=False,        # True: label replay set with the foundation model
    weight_pt=1.0,                   # replay-head loss weight
    e0s="estimated",
    mace_lr=1e-4,                    # lower LR recommended for multihead
    mace_epochs=500,
),
```


### Recipe 6: Finetune with stress training

Opt in to stress training when you want the model to predict stress tensors (for NPT MD, equation of state, etc.).

```python
dft=DFTConfig(
    tstress=True,                    # compute stress in QE (default True)
    # ... other DFT settings
),
trainer=TrainerConfig(
    # ... other trainer settings
    mace_compute_stress=True,        # tell MACE to train on stress
    mace_stress_key="stress",        # key in extxyz (default)
    mace_stress_weight=10.0,         # relative weight in the loss
),
```

The pipeline parses the QE stress tensor (Ry/bohr^3 -> eV/Ang^3 Voigt 6-vector, matching ASE convention) and writes it to the extxyz as `stress`.


### Recipe 7: Finetune with SWA tail

Stochastic weight averaging refines energies in the last phase of training.

```python
trainer=TrainerConfig(
    # ... other trainer settings
    mace_loss="huber",
    mace_energy_weight=1.0,
    mace_forces_weight=1000.0,
    mace_swa=True,
    mace_start_swa=300,              # switch to SWA after epoch 300
    mace_swa_lr=1e-3,
    mace_swa_energy_weight=1000.0,   # re-weight energy higher in SWA phase
    mace_swa_forces_weight=100.0,
),
```


### Recipe 8: Distillation (teacher -> student)

You have a trained teacher committee and seed structures. Generate a large synthetic dataset, label it with the teacher, and train a small fast student.

```python
from automlip import Config, Pipeline
from automlip.config import *

Pipeline(Config(
    mode="distill",
    work_dir="./distill_run",
    system=SystemConfig(
        elements=["B", "N"],
        composition={"B": 0.5, "N": 0.5},
    ),
    dft=DFTConfig(
        pseudopotentials={"B": "B.UPF", "N": "N.UPF"},
        pseudo_dir="/path/to/pseudo/",
        ecutwfc=80.0,
    ),
    trainer=TrainerConfig(
        backend="mace",
        device="cuda",
    ),
    distill=DistillConfig(
        teacher_model_paths=[
            "/path/to/teacher_0.model",
            "/path/to/teacher_1.model",
            "/path/to/teacher_2.model",
        ],
        seed_structures="/path/to/seeds.extxyz",
        n_synthetic=2000,
        rattle_sigma_pos=0.1,          # Angstrom displacement per atom
        rattle_sigma_cell=0.03,        # cell strain
        temperature=1000.0,            # for Boltzmann parent selection
        add_dimers=True,               # short-range two-body coverage
        force_std_threshold=0.2,       # eV/Ang; above this -> flagged
        route_flagged_to_qe=True,      # send disagreed structures to DFT
        # Student architecture
        student_num_channels=32,
        student_max_L=0,
        student_r_max=4.5,
        student_epochs=400,
        student_pair_repulsion=True,   # ZBL short-range repulsion
    ),
    scheduler=SchedulerConfig(backend="local", cores=8),
)).run()
```


### Recipe 9: NequIP backend

```python
trainer=TrainerConfig(
    backend="nequip",
    device="cuda",
    committee_size=3,
    foundation_model=None,           # or a path to pretrained .pth weights
    nequip_r_max=5.0,
    nequip_num_layers=4,
    nequip_l_max=2,
    nequip_num_features=32,
    nequip_max_epochs=1000,
    nequip_learning_rate=0.005,
    nequip_batch_size=5,
),
```

NequIP finetuning uses `initialize_from_state` in the YAML config to load pretrained weights. Set `foundation_model` to the `.pth` weights path.


### Recipe 10: Amorphous carbon with pre-relaxation (production example)

Full production config for amorphous carbon with RSS pre-relaxation, finetuning, huber loss, and SWA. See `examples/run_amorphous_carbon.py` for the complete script.

```python
cfg = Config(
    mode="finetune",
    data_path=None,                   # None triggers RSS generation
    system=SystemConfig(
        elements=["C"],
        composition={"C": 1.0},
        n_atoms=64,
        density=(1.0, 3.6),          # wide density range for amorphous
        min_distance=1.1,
    ),
    rss=RSSConfig(
        n_structures=400,
        prerelax_model="/path/to/mace_foundation.model",
        prerelax_fmax=0.1,
        prerelax_steps=100,
    ),
    trainer=TrainerConfig(
        foundation_model="/path/to/mace_foundation.model",
        committee_size=1,
        mace_loss="huber",
        mace_energy_weight=1.0,
        mace_forces_weight=1000.0,
        mace_swa=True,
        mace_start_swa=300,
        mace_swa_lr=1e-3,
        mace_swa_energy_weight=1000.0,
        mace_swa_forces_weight=100.0,
        mace_patience=50,
        mace_valid_fraction=0.4,
    ),
    # ... DFT and scheduler config
)
```


## Config reference

### Config (top-level)

| Field | Default | Description |
|---|---|---|
| `mode` | `"finetune"` | One of: `finetune`, `train`, `finetune_al`, `train_al`, `distill` |
| `start` | `"new"` | `"new"` or `"resume"`. Resume continues from `work_dir` checkpoints |
| `data_path` | `None` | Path to `.extxyz`, directory, or list of paths. `None` triggers RSS generation |
| `work_dir` | `"./automlip_run"` | Working directory for all outputs |
| `seed` | `None` | Master random seed. `None` draws a random seed at runtime |
| `cleanup_dft_scratch` | `False` | Delete QE scratch files (wavefunctions, tmp/) after each job |
| `energy_key` | `None` | Override auto-detected energy key in extxyz. Auto-detects: `energy`, `REF_energy`, `dft_energy`, `DFT_energy`, `free_energy`, `total_energy`, `potential_energy` |
| `force_key` | `None` | Override auto-detected force key. Auto-detects: `forces`, `REF_forces`, `dft_forces`, `DFT_forces`, `force` |


### SystemConfig

| Field | Default | Type | Description |
|---|---|---|---|
| `elements` | `[]` | `List[str]` | Element symbols, e.g. `["B", "N"]` |
| `composition` | `{}` | `Dict[str, float or (min, max)]` | Fractional composition per element. Renormalised to sum to 1 |
| `n_atoms` | `100` | `int or (min, max)` | Atoms per generated structure |
| `density` | `2.0` | `float or (min, max)` | Target mass density in g/cm^3 |
| `min_distance` | `None` | `float` | Minimum interatomic distance (Ang). Auto-calculated from covalent radii if `None` |


### DFTConfig

| Field | Default | Description |
|---|---|---|
| `pseudopotentials` | `{}` | Element -> UPF filename map, e.g. `{"B": "B.UPF"}` |
| `pseudo_dir` | `"./"` | Path to pseudopotential directory |
| `ecutwfc` | `80.0` | Wavefunction cutoff (Ry) |
| `ecutrho` | `None` | Charge density cutoff (Ry). Defaults to `8 x ecutwfc` |
| `kpoints` | `[1, 1, 1]` | k-point grid |
| `smearing` | `"mv"` | QE smearing type |
| `degauss` | `0.01` | Smearing width (Ry) |
| `mixing_beta` | `0.7` | SCF mixing parameter |
| `electron_maxstep` | `200` | Max SCF iterations |
| `conv_thr` | `1e-8` | SCF convergence threshold |
| `tstress` | `True` | Compute stress tensor in QE. Set `False` to save cost if you are not training on stress |


### TrainerConfig

**General:**

| Field | Default | Description |
|---|---|---|
| `backend` | `"mace"` | `"mace"` or `"nequip"` |
| `device` | `"auto"` | `"auto"` (detects GPU), `"cpu"`, `"cuda"` |
| `committee_size` | `3` | Number of models trained with different seeds |
| `seed` | `42` | Training random seed. Committee member `i` uses `seed + i` |
| `foundation_model` | `None` | Foundation model for finetuning. `None` = train from scratch. MACE: `"small"`, `"medium"`, `"large"`, or a local `.model` path. NequIP: `.pth` weights path |

**Finetuning (MACE only, used when `foundation_model` is set):**

| Field | Default | Description |
|---|---|---|
| `finetune_strategy` | `"naive"` | `"naive"` (continue training all params) or `"multihead"` (replay finetuning) |
| `e0s` | `"estimated"` | E0s method for finetuning. `"estimated"`, `"average"`, or a dict string like `"{5:-77.5}"` |
| `weight_decay` | `0.0` | Weight decay. 0 recommended for finetuning |
| `pt_train_file` | `None` | Replay dataset for multihead. Path to `.xyz`, or `"mp"` for Materials Project. Required when `finetune_strategy="multihead"` |
| `pseudolabel_replay` | `False` | Label the replay set with the foundation model |
| `weight_pt` | `1.0` | Replay-head loss weight |
| `weight_ft` | `1.0` | Target-head loss weight (kept in config but not emitted to MACE CLI) |

**MACE architecture (ignored during finetuning, foundation model sets these):**

| Field | Default | Description |
|---|---|---|
| `mace_model` | `"MACE"` | Model class, e.g. `"MACE"` or `"ScaleShiftMACE"` |
| `mace_r_max` | `5.0` | Cutoff radius (Ang) |
| `mace_num_channels` | `128` | Number of channels |
| `mace_max_L` | `1` | Maximum angular momentum |
| `mace_correlation` | `None` | Body-order correlation. `None` uses MACE default |
| `mace_num_interactions` | `None` | Number of interaction layers |
| `mace_hidden_irreps` | `None` | Hidden irreps string |
| `mace_mlp_irreps` | `None` | MLP irreps string |
| `mace_num_radial_basis` | `None` | Number of radial basis functions |
| `mace_distance_transform` | `None` | Distance transform type |

**MACE training:**

| Field | Default | Description |
|---|---|---|
| `mace_epochs` | `500` | Max training epochs |
| `mace_lr` | `0.01` | Learning rate |
| `mace_batch_size` | `10` | Training batch size |
| `mace_valid_fraction` | `0.1` | Validation fraction (used when no explicit val set is given) |
| `mace_default_dtype` | `"float64"` | `"float64"` or `"float32"` |
| `mace_scaling` | `"rms_forces_scaling"` | Finetuning scaling method |
| `mace_ema` | `True` | Exponential moving average (finetuning) |
| `mace_ema_decay` | `0.99` | EMA decay factor |
| `mace_amsgrad` | `True` | Use AMSGrad variant of Adam (finetuning) |

**MACE loss:**

| Field | Default | Description |
|---|---|---|
| `mace_loss` | `None` | Loss function, e.g. `"huber"`, `"weighted"`. `None` uses MACE default |
| `mace_energy_weight` | `None` | Energy loss weight |
| `mace_forces_weight` | `None` | Forces loss weight |
| `mace_stress_weight` | `None` | Stress loss weight |

**MACE stress:**

| Field | Default | Description |
|---|---|---|
| `mace_compute_stress` | `False` | Train on stress. Requires `dft.tstress=True` and stress in extxyz |
| `mace_stress_key` | `"stress"` | Key for stress in extxyz |

**MACE SWA (stochastic weight averaging):**

| Field | Default | Description |
|---|---|---|
| `mace_swa` | `False` | Enable SWA tail phase |
| `mace_start_swa` | `None` | Epoch to start SWA |
| `mace_swa_lr` | `None` | SWA learning rate |
| `mace_swa_energy_weight` | `None` | SWA energy loss weight |
| `mace_swa_forces_weight` | `None` | SWA forces loss weight |

**MACE optimiser and scheduler:**

| Field | Default | Description |
|---|---|---|
| `mace_optimizer` | `None` | Optimiser name |
| `mace_scheduler` | `None` | LR scheduler name |
| `mace_lr_scheduler_gamma` | `None` | Scheduler gamma |
| `mace_lr_factor` | `None` | LR factor |
| `mace_scheduler_patience` | `None` | Scheduler patience (epochs) |

**MACE training control:**

| Field | Default | Description |
|---|---|---|
| `mace_patience` | `None` | Early stopping patience (epochs) |
| `mace_eval_interval` | `None` | Evaluate every N epochs |
| `mace_clip_grad` | `None` | Gradient clipping value |
| `mace_keep_checkpoints` | `False` | Keep all checkpoints (vs only latest) |
| `mace_save_cpu` | `False` | Save model in CPU format |
| `mace_extra_args` | `{}` | Escape hatch: `{"flag": "value"}` emits `--flag=value`. `{"flag": None}` emits `--flag` |

**From-scratch E0s:**

| Field | Default | Description |
|---|---|---|
| `mace_e0s_scratch` | `"average"` | E0s method when training from scratch |

**NequIP-specific (ignored by MACE):**

| Field | Default | Description |
|---|---|---|
| `nequip_r_max` | `5.0` | Cutoff radius (Ang) |
| `nequip_num_layers` | `4` | Number of layers |
| `nequip_l_max` | `2` | Maximum angular momentum |
| `nequip_num_features` | `32` | Number of features |
| `nequip_max_epochs` | `1000` | Max training epochs |
| `nequip_learning_rate` | `0.005` | Learning rate |
| `nequip_batch_size` | `5` | Batch size |


### ActiveLearningConfig

| Field | Default | Type | Description |
|---|---|---|---|
| `max_iterations` | `30` | `int` | Maximum AL iterations |
| `md_temp` | `300.0` | `float or (min, max)` | MD temperature (K) |
| `md_steps` | `5000` | `int or (min, max)` | MD steps per iteration |
| `md_timestep` | `1.0` | `float` | Timestep (fs) |
| `md_interval` | `10` | `int` | Collect a snapshot every N steps |
| `qbc_select_fraction` | `0.1` | `float` | Top fraction by disagreement |
| `qbc_min_disagreement` | `0.0` | `float` | Floor for selection (eV/Ang). Structures below this are not selected. Set > 0 so `stop_on_no_new` can trigger when the committee agrees everywhere |
| `max_new_per_iter` | `50` | `int` | Max new structures per iteration |
| `rmse_energy_tol` | `0.005` | `float` | Energy RMSE convergence threshold (eV/atom) |
| `rmse_force_tol` | `0.1` | `float` | Force RMSE convergence threshold (eV/Ang) |
| `validation_fraction` | `0.1` | `float` | Fraction held out for validation |
| `stop_on_no_new` | `True` | `bool` | Stop if QBC selects zero structures |


### SchedulerConfig

| Field | Default | Description |
|---|---|---|
| `backend` | `"local"` | `"local"` (subprocesses) or `"pbs"` (one qsub per structure) |
| `cores` | `4` | Total cores available |
| `cores_per_dft_job` | `None` | Cores per QE job. Defaults to `cores`. On local, controls parallelism: `cores // cores_per_dft_job` jobs run concurrently |
| `mpi_command` | `"mpirun"` | MPI launcher. Can include flags, e.g. `"mpiexec.hydra -machinefile $PBS_NODEFILE -np $(wc -l < $PBS_NODEFILE)"` |
| `dft_executable` | `"pw.x"` | Quantum ESPRESSO executable |
| `walltime` | `"02:00:00"` | Walltime per job (HH:MM:SS). On local, enforced as a timeout |
| `queue` | `""` | PBS queue name. Empty string omits `#PBS -q` |
| `account` | `None` | PBS allocation/account. `None` omits `#PBS -A` |
| `modules` | `[]` | Shell lines run at the top of each PBS script (before the pw.x command), e.g. `["module load quantum-espresso"]` |
| `env_vars` | `{}` | Environment variables set in each job, e.g. `{"OMP_NUM_THREADS": "1"}` |
| `poll_interval` | `30.0` | Seconds between job-status checks |
| `pbs_max_concurrent` | `4` | Max PBS DFT jobs running simultaneously. Enforced by the driver, not a PBS array slot limit |


### RSSConfig

| Field | Default | Description |
|---|---|---|
| `n_structures` | `20` | Number of random structures to generate |
| `prerelax_model` | `None` | Foundation model for pre-relaxation. `"mace_mp"`, `"mace_mp_medium"`, `"mace_mp_large"`, or a local model path. `None` skips pre-relaxation |
| `prerelax_fmax` | `5.0` | Force convergence for pre-relaxation (eV/Ang). Intentionally loose |
| `prerelax_steps` | `50` | Max optimisation steps for pre-relaxation |


### DistillConfig

**Synthetic generation:**

| Field | Default | Type | Description |
|---|---|---|---|
| `teacher_model_paths` | `[]` | `List[str]` | Paths to trained teacher `.model` files |
| `seed_structures` | `None` | `str` | Path to extxyz of real seed structures |
| `n_synthetic` | `2000` | `int` | Total synthetic structures to generate |
| `rattle_sigma_pos` | `0.1` | `float or (min, max)` | Per-atom position displacement std (Ang) |
| `rattle_sigma_cell` | `0.03` | `float or (min, max)` | Cell strain std (dimensionless) |
| `relax_steps` | `20` | `int or (min, max)` | Robbins-Monro relaxation steps per child |
| `temperature` | `1000.0` | `float or (min, max)` | Temperature (K) for Boltzmann parent selection |
| `beta` | `0.5` | `float` | Explore (energy, Boltzmann) vs exploit (generation depth) balance. 1.0 = pure energy, 0.0 = pure generation |
| `max_force_stop` | `30.0` | `float` | Force ceiling (eV/Ang) for early-stop during relaxation |

**Dimers:**

| Field | Default | Description |
|---|---|---|
| `add_dimers` | `True` | Include isolated dimers for short-range coverage |
| `dimers_per_pair` | `50` | Number of dimer structures per element pair |
| `dimer_min_factor` | `0.6` | Min distance as fraction of sum of covalent radii |
| `dimer_max_factor` | `3.0` | Max distance as fraction of sum of covalent radii |
| `dimer_box` | `20.0` | Cubic box size (Ang) for isolated dimers |

**Committee disagreement gate:**

| Field | Default | Description |
|---|---|---|
| `force_std_threshold` | `0.2` | Force std threshold (eV/Ang). Structures above this are flagged |
| `route_flagged_to_qe` | `True` | Send flagged structures to QE for DFT labels. `False` drops them |

**Student model:**

| Field | Default | Description |
|---|---|---|
| `student_num_channels` | `32` | Student channel count (small by design) |
| `student_max_L` | `0` | Student max angular momentum (scalar-only for speed) |
| `student_r_max` | `4.5` | Student cutoff radius (Ang) |
| `student_epochs` | `400` | Student training epochs |
| `student_lr` | `0.01` | Student learning rate |
| `student_batch_size` | `10` | Student batch size |
| `student_pair_repulsion` | `True` | Add ZBL short-range pair repulsion |
| `student_loss` | `"weighted"` | Student loss function |
| `student_energy_weight` | `1.0` | Student energy loss weight |
| `student_forces_weight` | `100.0` | Student forces loss weight |

**Student SWA (off by default):**

| Field | Default | Description |
|---|---|---|
| `student_swa` | `False` | Enable SWA tail for the student |
| `student_start_swa` | `300` | Epoch to start SWA |
| `student_swa_lr` | `0.001` | SWA learning rate |
| `student_swa_energy_weight` | `1000.0` | SWA energy weight |
| `student_swa_forces_weight` | `100.0` | SWA forces weight |
| `validation_fraction` | `0.1` | Student validation fraction |


## Important defaults to know

These defaults are the most likely to need adjustment for your system:

| What | Default | Why you might change it |
|---|---|---|
| `trainer.device` | `"auto"` | Fails hard if no GPU is found. Set to `"cpu"` for testing (but training will be very slow) |
| `trainer.committee_size` | `3` | Set to 1 for `finetune`/`train` modes where you don't need QBC. More models = better disagreement estimates but slower |
| `dft.tstress` | `True` | Computes stress in QE even if you are not training on it. Set to `False` to save DFT compute time |
| `trainer.mace_compute_stress` | `False` | Even with `tstress=True`, MACE will not train on stress unless you set this to `True` |
| `active.md_temp` | `300.0` | Room temperature. Set higher (1000+) for amorphous/disordered systems to sample more diverse configurations |
| `active.md_interval` | `10` | Collects a snapshot every 10 MD steps. With `md_steps=5000` that is 500 snapshots per iteration, which may be more than needed. Increase for cheaper selection |
| `active.qbc_min_disagreement` | `0.0` | With the default of 0, QBC always selects at least one structure (even if the committee agrees). Set > 0 to let AL converge when the committee agrees everywhere |
| `rss.prerelax_model` | `None` | No pre-relaxation. Strongly recommended to set this for amorphous/disordered systems to avoid DFT failures on unphysical configurations |
| `scheduler.backend` | `"local"` | Fine for testing. Switch to `"pbs"` for production HPC runs |
| `scheduler.pbs_max_concurrent` | `4` | How many DFT jobs run at once. Tune to your allocation throughput |
| `distill.route_flagged_to_qe` | `True` | Sends disagreed structures to DFT, which requires a valid DFT + scheduler config even in distill mode |


## Data format

Extended XYZ (`.extxyz`). Each frame needs an energy in `info` and forces in `arrays`. The loader auto-detects common key names:

**Energy keys** (checked in order): `energy`, `Energy`, `REF_energy`, `dft_energy`, `DFT_energy`, `free_energy`, `total_energy`, `potential_energy`. Also checks the attached ASE calculator results dict for `energy` and `free_energy`.

**Force keys** (checked in order): `forces`, `Forces`, `REF_forces`, `dft_forces`, `DFT_forces`, `force`. Also checks the calculator results dict.

Override with `energy_key` and `force_key` in `Config` if your file uses non-standard names.

Frames without energy, forces, or periodic boundary conditions are silently skipped during loading.

The loader accepts a single file, a directory (reads all `.extxyz` and `.xyz` files), or a list of file paths.


## DFT failures

When a QE job fails, the labeller logs the specific reason and discards that structure. There are no retries. Common failure reasons:

- SCF convergence not achieved
- S matrix not positive definite
- Eigenvalues not converged
- Wrong charge
- CRASH file present (likely out of memory)
- Run did not finish (killed by walltime or scheduler)

The run continues with the structures that succeeded.


## Work directory layout

```
<work_dir>/
    automlip.log                    # full pipeline log
    checkpoint.json                 # AL state (iteration, model paths, history)
    training_data.extxyz            # current training set (AL modes)
    validation_data.extxyz          # current validation set (AL modes)
    final_training_set.extxyz       # final dataset after AL converges
    models/
        iter_000/                   # models from iteration 0
            train.extxyz
            committee_00/
                committee_00.model
            committee_01/
                committee_01.model
        iter_001/
            ...
    iter_000/                       # DFT labelling directories
        label_0000/
            pw.in
            pw.out
        label_0001/
            ...
    iter_001/
        ...
    # Distillation:
    synthetic_data.extxyz           # labelled synthetic set
    student/
        synthetic.extxyz
        student/
            student.model
    distill_qe/                     # DFT labels for flagged structures
        label_0000/
            ...
```


## Extending AutoMLIP

### Adding a new MLIP backend

All backends implement `BaseTrainer` in `automlip/trainers/__init__.py`. To add a new one (e.g. Allegro, FLARE, ACE):

**Step 1.** Create `automlip/trainers/mybackend.py`:

```python
from pathlib import Path
from typing import List
from ase import Atoms
from automlip.trainers import BaseTrainer, resolve_device

class MyBackendTrainer(BaseTrainer):
    def __init__(self, system_config, trainer_config):
        self.system = system_config
        self.config = trainer_config
        self.device = resolve_device(trainer_config)

    def train_committee(self, data: List[Atoms], model_dir: Path,
                        n_models: int = 3, restart: bool = False,
                        val_data: List[Atoms] = None) -> List[Path]:
        """Train N models with different seeds. Return list of model paths.

        Must handle:
        - Writing data to model_dir in whatever format your backend needs
        - Training n_models, each with seed = self.config.seed + i
        - restart=True: resume from checkpoint if one exists
        - val_data: if given, use as explicit validation set
        - Return a list of Paths to the saved model files
        """
        paths = []
        for i in range(n_models):
            seed = self.config.seed + i
            # ... train model ...
            paths.append(model_path)
        if not paths:
            raise RuntimeError("All models failed")
        return paths

    def get_calculator(self, model_path: Path):
        """Return an ASE Calculator for inference with this model.

        Used for MD sampling, committee prediction, and validation.
        Must work with atoms.calc = calc; atoms.get_potential_energy(), etc.
        """
        from mybackend import MyCalculator
        return MyCalculator(model_path, device=self.device)
```

The base class provides default implementations of `predict_committee` and `compute_validation_error` that use `get_calculator`, so you only need to implement `train_committee` and `get_calculator`. Override `predict_committee` if your backend has a more efficient batched inference path.

For distillation support, also implement `train_student`:

```python
    def train_student(self, data: List[Atoms], model_dir: Path,
                      distill_config, restart: bool = False) -> Path:
        """Train a small student model from scratch. Return path to model."""
        # ... train a small model on synthetic data ...
        return model_path
```

**Step 2.** Add your backend-specific config fields to `TrainerConfig` in `automlip/config.py`:

```python
@dataclass
class TrainerConfig:
    # ... existing fields ...

    # MyBackend-specific (ignored by MACE/NequIP).
    mybackend_cutoff: float = 5.0
    mybackend_num_layers: int = 3
    # ... etc ...
```

**Step 3.** Register in the factory in `automlip/trainers/__init__.py`:

```python
def make_trainer(system_config, trainer_config) -> BaseTrainer:
    backend = trainer_config.backend.lower()
    if backend == "mace":
        from automlip.trainers.mace import MACETrainer
        return MACETrainer(system_config, trainer_config)
    elif backend == "nequip":
        from automlip.trainers.nequip import NequIPTrainer
        return NequIPTrainer(system_config, trainer_config)
    elif backend == "mybackend":
        from automlip.trainers.mybackend import MyBackendTrainer
        return MyBackendTrainer(system_config, trainer_config)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
```

**Step 4.** Add optional dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
mybackend = ["mybackend-package"]
```

That is it. The rest of the pipeline (RSS, DFT labelling, active learning, selection, checkpointing) is backend-agnostic.

### Adding a new DFT labeller

The QE labeller lives in `automlip/labellers/qe.py`. To add support for VASP, CP2K, etc.:

**Step 1.** Create `automlip/labellers/vasp.py` (for example) with:

```python
def write_vasp_input(atoms, calc_dir, dft_config) -> Path:
    """Write input files. Return path to main input file."""
    ...

def parse_vasp_output(calc_dir, atoms) -> Optional[Atoms]:
    """Parse output. Return labelled Atoms or None on failure."""
    ...

def get_run_command(sched_config) -> str:
    """Return the shell command to run the DFT code."""
    ...
```

**Step 2.** Modify `automlip/labellers/batch.py` to select between QE and your new labeller based on a config field (or add a `dft.backend` field to `DFTConfig`).

The batch labeller, schedulers, and the rest of the pipeline do not know or care what DFT code runs inside each directory - they just submit a shell command and check whether parsing succeeded.


### Adding a new scheduler backend

The scheduler interface is in `automlip/schedulers/__init__.py`. To add Slurm:

**Step 1.** Create `automlip/schedulers/slurm.py`:

```python
from pathlib import Path
from typing import List
from automlip.config import SchedulerConfig
from automlip.schedulers import JobStatus

class SlurmScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config

    def run_batch(self, calc_dirs: List[Path], run_cmd: str,
                  config: SchedulerConfig) -> List[JobStatus]:
        """Submit jobs, poll until done, return statuses.

        Must return one JobStatus per calc_dir (same order).
        DONE means the job finished (caller checks pw.out for success).
        FAILED means the job could not be submitted or was killed.
        """
        ...
```

**Step 2.** Register in `automlip/schedulers/__init__.py`:

```python
def make_scheduler(config):
    backend = config.backend.lower()
    if backend == "local":
        ...
    elif backend == "pbs":
        ...
    elif backend == "slurm":
        from automlip.schedulers.slurm import SlurmScheduler
        return SlurmScheduler(config)
    else:
        raise ValueError(f"Unknown scheduler: {backend!r}")
```

**Step 3.** Add any Slurm-specific fields to `SchedulerConfig`.


### Adding a new sampler

Samplers generate candidate structures. The current ones are:

- `automlip/samplers/rss.py` - random structure generation
- `automlip/samplers/md.py` - Langevin MD sampling
- `automlip/samplers/augment.py` - rattle-relax trees and dimers for distillation

To add a new sampler (e.g. metadynamics, basin hopping, genetic algorithm):

**Step 1.** Create `automlip/samplers/mysampler.py`:

```python
def my_sampler(atoms, calculator, **kwargs) -> List[Atoms]:
    """Generate candidate structures. Return unlabelled Atoms."""
    ...
```

**Step 2.** Call it from `pipeline.py` wherever MD is currently called (in `_al_loop` for AL, or in `_build_initial_data` for seed generation).


### Adding a new selector

Selectors decide which candidates to send to DFT. The current one (`automlip/selectors/committee.py`) ranks by mean force standard deviation (QBC).

To add a new strategy (e.g. maximum entropy, D-optimality, or a hybrid score):

**Step 1.** Create `automlip/selectors/myselector.py`:

```python
def my_selector(predictions, **kwargs) -> Tuple[List[int], List[float]]:
    """Return (selected_indices, scores)."""
    ...
```

The input is a list of `Prediction` objects (one per candidate), each with `.energies` (shape `(n_models,)`) and `.forces` (shape `(n_models, n_atoms, 3)`).

**Step 2.** Swap the call in `pipeline.py`'s `_al_loop`.


## Committee strategies

MACE and NequIP build committees by training with different random seeds on the same data. Each model converges to a different local minimum, so the committee produces diverse predictions. The standard deviation of their force predictions is the disagreement metric (QBC score) used for active learning.

When finetuning, each committee member starts from the same foundation model weights but trains with a different seed.

