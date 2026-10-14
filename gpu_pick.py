#!/usr/bin/env python3
import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

CpuArch = Literal["x86_64", "aarch64"]


@dataclass(frozen=True)
class GpuSpec:
    memory_gb: float
    min_cuda: tuple[int, int]
    max_cuda: tuple[int, int] | None = None


GPU_SPEC_BY_TYPE: dict[str, GpuSpec] = {
    "p100": GpuSpec(memory_gb=16, min_cuda=(8, 0), max_cuda=(12, 9)),
    "l40s": GpuSpec(memory_gb=48, min_cuda=(11, 8)),
    "a100": GpuSpec(memory_gb=80, min_cuda=(11, 0)),
    "h100": GpuSpec(memory_gb=80, min_cuda=(11, 8)),
    "gh200": GpuSpec(memory_gb=96, min_cuda=(12, 0)),
    "h200": GpuSpec(memory_gb=141, min_cuda=(12, 0)),
    "b200": GpuSpec(memory_gb=180, min_cuda=(12, 8)),
}
CPU_ARCH_BY_PARTITION: dict[str, CpuArch] = {"mgh": "aarch64"}
STANDBY_GPU_QOS = "gstandby"
NOT_YET_ELIGIBLE_REASONS = frozenset(
    {"Dependency", "DependencyNeverSatisfied", "JobHeldUser", "JobHeldAdmin", "BeginTime"}
)
USABLE_NODE_STATES = frozenset({"idle", "mix", "alloc", "mix-", "alloc-", "idle-", "comp"})


@dataclass(frozen=True)
class Qos:
    name: str
    priority: int
    usage_factor: float
    max_wall_hours: float | None
    preempted_qos_names: frozenset[str]


@dataclass(frozen=True)
class Partition:
    name: str
    allowed_qos_names: frozenset[str]
    max_time_hours: float | None
    gpu_billing_weight: float
    cpu_billing_weight: float
    mem_billing_weight_per_gb: float


@dataclass(frozen=True)
class GpuNode:
    name: str
    partition_name: str
    gpu_type: str
    total_gpus: int
    used_gpus: int


@dataclass(frozen=True)
class PendingJob:
    partition_names: frozenset[str]
    qos_name: str


@dataclass(frozen=True)
class JobRequest:
    vram_gb: float
    hours: float
    cpus: int
    mem_gb: float
    requeue_ok: bool
    fast_turnaround: bool
    cpu_arch: CpuArch
    cuda: tuple[int, int]
    excluded_gpu_types: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    partition_name: str
    gpu_type: str
    qos: Qos
    submit_qos_name: str
    gpu_count: int
    max_wall_hours: float
    idle_gpus: int
    preemptable_gpus: int
    startable_now: bool
    pending_jobs_ahead: int
    billing_per_hour: float
    total_billing: float


def run_command(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def parse_slurm_duration_hours(duration: str) -> float | None:
    if duration in ("", "UNLIMITED", "Partition_Limit", "INFINITE"):
        return None
    days, _, clock = duration.rpartition("-")
    clock_parts = [int(part) for part in clock.split(":")]
    hours, minutes, seconds = [0] * (3 - len(clock_parts)) + clock_parts
    return (int(days) if days else 0) * 24 + hours + minutes / 60 + seconds / 3600


def parse_billing_weights(weights_text: str) -> tuple[float, float, float]:
    weight_by_tres = dict(item.split("=", 1) for item in weights_text.split(",") if "=" in item)
    mem_weight_text = weight_by_tres.get("Mem", "0")
    mem_weight_per_gb = float(mem_weight_text[:-1]) if mem_weight_text.endswith("G") else float(mem_weight_text) * 1024
    return float(weight_by_tres.get("GRES/gpu", "0")), float(weight_by_tres.get("CPU", "0")), mem_weight_per_gb


def load_user_qos_names() -> frozenset[str]:
    output = run_command(["sacctmgr", "-nP", "show", "assoc", "user=" + run_command(["whoami"]).strip(), "format=QOS"])
    return frozenset(name for line in output.split() for name in line.split(",") if name)


def load_qos_by_name() -> dict[str, Qos]:
    output = run_command(["sacctmgr", "-nP", "show", "qos", "format=Name,Priority,UsageFactor,MaxWall,Preempt"])
    rows = [line.split("|") for line in output.splitlines() if line]
    return {
        name: Qos(
            name=name,
            priority=int(priority or 0),
            usage_factor=float(usage_factor),
            max_wall_hours=parse_slurm_duration_hours(max_wall),
            preempted_qos_names=frozenset(filter(None, preempt.split(","))),
        )
        for name, priority, usage_factor, max_wall, preempt in rows
    }


def load_partitions() -> list[Partition]:
    output = run_command(["scontrol", "show", "partition", "-o"])
    field_maps = [dict(re.findall(r"(\w+)=(\S*)", line)) for line in output.splitlines() if line]
    return [
        Partition(
            name=fields["PartitionName"],
            allowed_qos_names=frozenset(fields.get("AllowQos", "").split(",")),
            max_time_hours=parse_slurm_duration_hours(fields.get("MaxTime", "")),
            gpu_billing_weight=gpu_weight,
            cpu_billing_weight=cpu_weight,
            mem_billing_weight_per_gb=mem_weight,
        )
        for fields in field_maps
        for gpu_weight, cpu_weight, mem_weight in [parse_billing_weights(fields.get("TRESBillingWeights", ""))]
        if gpu_weight > 0
    ]


def parse_gres_gpu(gres_text: str) -> tuple[str, int] | None:
    match = re.search(r"gpu:([a-z0-9_]+):(\d+)", gres_text)
    return (match.group(1), int(match.group(2))) if match else None


def load_gpu_nodes(partition_names: list[str]) -> list[GpuNode]:
    output = run_command([
        "sinfo", "-h", "-N", "-p", ",".join(partition_names),
        "-O", "Partition:30,NodeList:40,StateCompact:14,Gres:60,GresUsed:80",
    ])
    nodes: list[GpuNode] = []
    for line in output.splitlines():
        partition_name, node_name, state, gres_text, gres_used_text = line.split()[:5]
        gpu_total = parse_gres_gpu(gres_text)
        gpu_used = parse_gres_gpu(gres_used_text)
        if gpu_total is None or state not in USABLE_NODE_STATES:
            continue
        nodes.append(GpuNode(
            name=node_name, partition_name=partition_name.rstrip("*"), gpu_type=gpu_total[0],
            total_gpus=gpu_total[1], used_gpus=gpu_used[1] if gpu_used else 0,
        ))
    return nodes


def expand_hostlist(hostlist: str) -> list[str]:
    return run_command(["scontrol", "show", "hostnames", hostlist]).split()


def load_standby_gpus_by_node() -> dict[str, int]:
    output = run_command(["squeue", "-h", "-t", "R", "-q", STANDBY_GPU_QOS, "-O", "NumNodes:8,NodeList:200,tres-alloc:300"])
    standby_gpus_by_node: dict[str, int] = {}
    for line in output.splitlines():
        node_count_text, hostlist, tres_text = line.split()[:3]
        gpu_match = re.search(r"gres/gpu=(\d+)", tres_text)
        if not gpu_match:
            continue
        gpus_per_node = int(gpu_match.group(1)) // int(node_count_text)
        for node_name in expand_hostlist(hostlist):
            standby_gpus_by_node[node_name] = standby_gpus_by_node.get(node_name, 0) + gpus_per_node
    return standby_gpus_by_node


def load_eligible_pending_jobs() -> list[PendingJob]:
    output = run_command(["squeue", "-h", "-t", "PD", "-O", "Partition:200,QOS:30,Reason:60"])
    rows = [line.split() for line in output.splitlines() if line.strip()]
    return [
        PendingJob(partition_names=frozenset(row[0].split(",")), qos_name=row[1])
        for row in rows
        if len(row) >= 3 and row[2] not in NOT_YET_ELIGIBLE_REASONS
    ]


def gpu_job_qos_names(user_qos_names: frozenset[str]) -> dict[str, str]:
    submit_name_by_effective_qos = {name: name for name in user_qos_names if name not in ("standby", STANDBY_GPU_QOS)}
    if "standby" in user_qos_names:
        submit_name_by_effective_qos[STANDBY_GPU_QOS] = "standby"
    return submit_name_by_effective_qos


def billing_per_hour(partition: Partition, gpu_count: int, request: JobRequest) -> float:
    return max(
        partition.gpu_billing_weight * gpu_count,
        partition.cpu_billing_weight * request.cpus,
        partition.mem_billing_weight_per_gb * request.mem_gb,
    )


def combined_max_wall_hours(qos: Qos, partition: Partition) -> float:
    limits = [limit for limit in (qos.max_wall_hours, partition.max_time_hours) if limit is not None]
    return min(limits) if limits else math.inf


def is_gpu_allowed(gpu_type: str, partition_name: str, request: JobRequest) -> bool:
    spec = GPU_SPEC_BY_TYPE.get(gpu_type)
    return (
        spec is not None
        and gpu_type not in request.excluded_gpu_types
        and request.cuda >= spec.min_cuda
        and (spec.max_cuda is None or request.cuda <= spec.max_cuda)
        and CPU_ARCH_BY_PARTITION.get(partition_name, "x86_64") == request.cpu_arch
    )


def count_pending_jobs_ahead(partition_name: str, qos: Qos, pending_jobs: list[PendingJob], qos_by_name: dict[str, Qos]) -> int:
    return sum(
        1 for job in pending_jobs
        if partition_name in job.partition_names
        and job.qos_name in qos_by_name
        and qos_by_name[job.qos_name].priority >= qos.priority
    )


def build_candidates(
    request: JobRequest,
    partitions: list[Partition],
    gpu_nodes: list[GpuNode],
    standby_gpus_by_node: dict[str, int],
    pending_jobs: list[PendingJob],
    qos_by_name: dict[str, Qos],
    submit_name_by_effective_qos: dict[str, str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for partition in partitions:
        partition_nodes = [node for node in gpu_nodes if node.partition_name == partition.name]
        if not partition_nodes:
            continue
        gpu_type = partition_nodes[0].gpu_type
        if not is_gpu_allowed(gpu_type, partition.name, request):
            continue
        gpu_count = max(1, math.ceil(request.vram_gb / GPU_SPEC_BY_TYPE[gpu_type].memory_gb))
        if gpu_count > max(node.total_gpus for node in partition_nodes):
            continue
        for effective_qos_name, submit_qos_name in submit_name_by_effective_qos.items():
            qos = qos_by_name.get(effective_qos_name)
            if qos is None or effective_qos_name not in partition.allowed_qos_names:
                continue
            if effective_qos_name == STANDBY_GPU_QOS and not request.requeue_ok:
                continue
            max_wall_hours = combined_max_wall_hours(qos, partition)
            if request.hours > max_wall_hours:
                continue
            can_preempt_standby = STANDBY_GPU_QOS in qos.preempted_qos_names
            free_gpus_by_node = {
                node.name: node.total_gpus - node.used_gpus
                + (standby_gpus_by_node.get(node.name, 0) if can_preempt_standby else 0)
                for node in partition_nodes
            }
            hourly_billing = billing_per_hour(partition, gpu_count, request) * qos.usage_factor
            candidates.append(Candidate(
                partition_name=partition.name,
                gpu_type=gpu_type,
                qos=qos,
                submit_qos_name=submit_qos_name,
                gpu_count=gpu_count,
                max_wall_hours=max_wall_hours,
                idle_gpus=sum(node.total_gpus - node.used_gpus for node in partition_nodes),
                preemptable_gpus=sum(standby_gpus_by_node.get(node.name, 0) for node in partition_nodes) if can_preempt_standby else 0,
                startable_now=any(free >= gpu_count for free in free_gpus_by_node.values()),
                pending_jobs_ahead=count_pending_jobs_ahead(partition.name, qos, pending_jobs, qos_by_name),
                billing_per_hour=hourly_billing,
                total_billing=hourly_billing * request.hours,
            ))
    return candidates


def qos_preference_rank(candidate: Candidate, fast_turnaround: bool) -> int:
    if candidate.qos.name == "cs":
        return 0
    if candidate.qos.name == STANDBY_GPU_QOS:
        return 3 if fast_turnaround and candidate.pending_jobs_ahead > 0 else 1
    return 2


def rank_candidates(candidates: list[Candidate], fast_turnaround: bool) -> list[Candidate]:
    return sorted(candidates, key=lambda candidate: (
        not candidate.startable_now,
        candidate.gpu_count,
        qos_preference_rank(candidate, fast_turnaround),
        candidate.total_billing,
        candidate.pending_jobs_ahead,
    ))


def format_hours_as_slurm_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    clock = f"{remaining_minutes // 60:02d}:{remaining_minutes % 60:02d}:00"
    return f"{days}-{clock}" if days else clock


def suggest_sbatch_command(ranked_candidates: list[Candidate], request: JobRequest) -> str:
    best = ranked_candidates[0]
    same_submission = [
        candidate.partition_name for candidate in ranked_candidates
        if candidate.submit_qos_name == best.submit_qos_name
        and candidate.gpu_count == best.gpu_count
        and candidate.startable_now == best.startable_now
    ]
    gpu_flag = f"--gpus {best.gpu_type}:{best.gpu_count}" if len(same_submission) == 1 else f"--gpus {best.gpu_count}"
    return (
        f"sbatch -p {','.join(same_submission)} --qos {best.submit_qos_name} {gpu_flag} "
        f"--time {format_hours_as_slurm_time(request.hours)}"
    )


def format_candidate_table(ranked_candidates: list[Candidate]) -> str:
    header = f"{'#':>2}  {'partition':<9} {'gpu':<6} {'n':>2}  {'--qos':<9} {'prio':>4}  {'now?':<4} {'idle':>4} {'preempt':>7} {'jobs ahead':>10}  {'cost/hr':>8} {'cost':>9}  {'max wall':>8}"
    rows = [
        f"{index:>2}  {c.partition_name:<9} {c.gpu_type:<6} {c.gpu_count:>2}  {c.submit_qos_name:<9} {c.qos.priority:>4}  "
        f"{'yes' if c.startable_now else 'no':<4} {c.idle_gpus:>4} {c.preemptable_gpus:>7} {c.pending_jobs_ahead:>10}  "
        f"{c.billing_per_hour:>8.1f} {c.total_billing:>9.0f}  {format_hours_as_slurm_time(c.max_wall_hours) if math.isfinite(c.max_wall_hours) else '-':>8}"
        for index, c in enumerate(ranked_candidates, start=1)
    ]
    return "\n".join([header, *rows])


def parse_cuda_version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")
    return int(major), int(minor)


def parse_request() -> tuple[JobRequest, int]:
    parser = argparse.ArgumentParser(description="Rank partition/QOS options for a single-node GPU job, using live Slurm state.")
    parser.add_argument("--hours", type=float, required=True, help="Wall time the job needs")
    parser.add_argument("--vram-gb", type=float, default=0, help="Total GPU memory needed; sets the GPU count per GPU type")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gb", type=float, default=0, help="Host memory; counts toward billing under MAX_TRES")
    parser.add_argument("--requeue-ok", action=argparse.BooleanOptionalAction, default=True,
                        help="Job survives preemption+requeue (allows --qos standby)")
    parser.add_argument("--fast", action="store_true", help="Rank contended standby options below full-price ones")
    parser.add_argument("--arch", choices=["x86_64", "aarch64"], default="x86_64")
    parser.add_argument("--cuda", type=parse_cuda_version, default=(12, 6), help="CUDA version the env is built for, e.g. 12.6")
    parser.add_argument("--exclude-gpu", action="append", default=[], help="GPU type to skip, e.g. p100 (repeatable)")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    return JobRequest(
        vram_gb=args.vram_gb, hours=args.hours, cpus=args.cpus, mem_gb=args.mem_gb,
        requeue_ok=args.requeue_ok, fast_turnaround=args.fast, cpu_arch=args.arch, cuda=args.cuda,
        excluded_gpu_types=frozenset(gpu.lower() for gpu in args.exclude_gpu),
    ), args.top


def main() -> None:
    request, top_count = parse_request()
    qos_by_name = load_qos_by_name()
    partitions = load_partitions()
    candidates = build_candidates(
        request=request,
        partitions=partitions,
        gpu_nodes=load_gpu_nodes([partition.name for partition in partitions]),
        standby_gpus_by_node=load_standby_gpus_by_node(),
        pending_jobs=load_eligible_pending_jobs(),
        qos_by_name=qos_by_name,
        submit_name_by_effective_qos=gpu_job_qos_names(load_user_qos_names()),
    )
    if not candidates:
        print("No partition/QOS satisfies these constraints.")
        return
    ranked_candidates = rank_candidates(candidates, request.fast_turnaround)
    print(format_candidate_table(ranked_candidates[:top_count]))
    print()
    print(suggest_sbatch_command(ranked_candidates, request))


if __name__ == "__main__":
    main()