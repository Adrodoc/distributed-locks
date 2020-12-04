# coding=utf8

from pathlib import Path
import re

pattern = re.compile(
    r"((?:\s*\{\s*\"name\": \"\w+/McsLock/.+(?:\s+\".+)+\s*\},?)+)," +
    r"((?:\s*\{\s*\"name\": \"\w+/McsLockAccumulate/.+(?:\s+\".+)+\s*\},?)+)," +
    r"((?:\s*\{\s*\"name\": \"\w+/McsLockMpi/.+(?:\s+\".+)+\s*\},?)+)," +
    r"((?:\s*\{\s*\"name\": \"\w+/McsLockTwoSided/.+(?:\s+\".+)+\s*\},?)+)")
substitution = "\\4,\\1,\\2,\\3"

renamings = {
    "CTktMcsAtomicLockOptimizedCounter": "C-TKT-MCS",
    "CTktMcsLock": "C-TKT-MCS",
    "CTktMcsLockLocalCounter": "C-TKT-MCS (local counter)",
    "CTktMcsLockMemCounter": "C-TKT-MCS (direct counter)",
    "CTktMcsLockOptimizedCounter": "C-TKT-MCS (inline counter)",
    "DMcsLock": "D-MCS",
    "RmaMcsLock": "RMA-MCS",
    "HMcsLockAtomic": "H-MCS",
    "McsLock": "MCS (non-atomic)",
    "McsLockAccumulate": "MCS (MPI_Iprobe)",
    "McsLockMpi": "MCS",
    "McsLockTwoSided": "MCS (P2P)",
    "MpiWinLock": "MPI",
    "RhLock": "RH",
    "RhLockAtomicRef": "RH (atomic_ref)",
    "ShflLockNoStealing": "SHFL",
}

dir_path = Path(
    'reports/c58ecbd-lrz-cm2_tiny-mcs-optimization-vortrag/json')
for path in dir_path.iterdir():
    with open(path, 'r+') as file:
        content = file.read()
        content = re.sub(pattern, substitution, content)
        for old, new in renamings.items():
            content = content.replace(f'/{old}/', f'/{new}/')
        file.seek(0)
        file.write(content)
        file.truncate()
