"""Run binding-ID metrics from a MIRROR eval-style config, submitting to Slurm when on a login node.

Usage (from the repo root):
    python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/llama3.2-1b.yaml

On a login node with `slurm.job_type: compute`, this submits itself as a job and exits.
Compute nodes are offline, so download the model once first with
`--slurm.job_type local-download` on a login node.
"""

import sys

ACTIVATE_CMD = "mamba activate binding_ids"


def main(args: list[str]) -> None:
    # Heavy imports (torch, transformers) come after submission, as in MIRROR's main.py,
    # so submitting from a login node stays fast.
    from mirror.slurm_util import parse_slurm_config

    from selfconcept.common.slurm import submit_slurm_job

    submit_slurm_job(parse_slurm_config(args), "selfconcept.binding_ids.run", args, ACTIVATE_CMD)

    from mirror.cli_parsers import build_parser
    from mirror.config import get_config, init_config
    from mirror.fabric_util import cpu_safe_strategy, make_fabric
    from mirror.models.model_util import instantiate_model
    from mirror.subcommands import evaluation
    from mirror.util import is_login_node, resolve_config_args

    parser = build_parser("eval")
    cfg = parser.parse_args(resolve_config_args(args))
    if hasattr(cfg, "config"):
        del cfg.config

    init_config(cfg.device)
    init = parser.instantiate_classes(cfg)

    device = get_config()["device"]
    fabric = make_fabric(
        # MIRROR's eval parser defaults strategy to None, which Fabric rejects
        cpu_safe_strategy(init.strategy or "auto", device),
        device,
        devices=init.slurm.ntasks_per_node or 1,
        num_nodes=init.slurm.nodes or 1,
    )
    fabric.launch()
    model = instantiate_model(init.model, fabric=fabric)

    if is_login_node() and init.slurm.job_type == "local-download":
        print("Model downloaded/cached. Re-run with slurm.job_type compute.")
        return

    evaluation(
        model=model,
        metrics=init.metrics,
        fabric=fabric,
        checkpoint_path=init.checkpoint_path,
        slurm=init.slurm,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
