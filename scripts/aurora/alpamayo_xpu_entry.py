"""Aurora XPU launch wrapper for Alpamayo-R1 SFT (finetune.sft.train_hf).

Aurora's XPU collective (xccl over oneCCL) is launched via mpiexec/PMIx, not
torchrun's TCP rendezvous (which segfaults with oneCCL). HuggingFace Trainer /
accelerate / deepspeed only *reuse* an already-initialized process group, so we
pin this rank's tile and init the xccl group here BEFORE importing train_hf (any
XPU-touching import before set_device makes every rank grab tile 0 and segfault).

Usage (under mpiexec, one rank per tile):
    python scripts/aurora/alpamayo_xpu_entry.py --config-name sft_stage1 <hydra overrides...>
All argv after this script is forwarded verbatim to finetune.sft.train_hf's
hydra CLI. Mirrors VLA-JEPA's VLAJEPA_XPU_MPI bootstrap (see aurora-xpu-distributed).
"""
import os
import runpy
import socket
import sys

import torch
from mpi4py import MPI

_comm = MPI.COMM_WORLD
_rank, _world = _comm.Get_rank(), _comm.Get_size()
_local = int(os.environ.get("PALS_LOCAL_RANKID", os.environ.get("MPI_LOCALRANKID", _rank)))
_master = _comm.bcast(socket.gethostname() if _rank == 0 else None, root=0)
# Aurora rendezvous must go over the high-speed network (HSN) interface.
os.environ["MASTER_ADDR"] = f"{_master}.hsn.cm.aurora.alcf.anl.gov"
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["RANK"] = str(_rank)
os.environ["WORLD_SIZE"] = str(_world)
os.environ["LOCAL_RANK"] = str(_local)
os.environ["LOCAL_WORLD_SIZE"] = os.environ.get("PALS_LOCAL_SIZE", str(_world))

torch.xpu.set_device(_local)
if not torch.distributed.is_initialized():
    # Backend must be "xccl" (torch 2.10+xpu); "ccl" is not registered.
    torch.distributed.init_process_group(
        backend="xccl", rank=_rank, world_size=_world,
        device_id=torch.device(f"xpu:{_local}"),
    )

# Forward remaining argv to train_hf's hydra @main. runpy(alter_sys=True) sets
# argv[0] to the module file; we keep argv[1:] as the hydra overrides.
sys.argv = [sys.argv[0]] + sys.argv[1:]
runpy.run_module("finetune.sft.train_hf", run_name="__main__", alter_sys=True)
