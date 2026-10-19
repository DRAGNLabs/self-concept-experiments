"""Application-agnostic Bayesian batch search over arms by tiered Thompson sampling: fit a ladder of relaxed targets,
pick each batch at the strictest tier expected to hit, run it, refit, repeat."""
