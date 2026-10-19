from jsonargparse import auto_cli

from selfconcept.arm_search.toy.experiment import run_replicate
from selfconcept.arm_search.toy.experiment_report import summarize_experiment

if __name__ == "__main__":
    auto_cli({"run": run_replicate, "summarize": summarize_experiment})
