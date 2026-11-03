# Sourced by the sbatch scripts here: write_run_info <dir> <arm> <prompt file or empty> [problem files...]
# Writes <dir>/run_info/ so a run can be checked later: run_info.txt (date, Slurm job, arm, git commit, branch and
# status, plus whatever the caller passes in RUN_INFO_EXTRA), uncommitted.diff (git diff HEAD of the checkout),
# arms.tsv, prompt.txt (the arm's prompt, if any) and problems.sha256 (checksums of the problem files, if given).
write_run_info() {
    local dir=$1/run_info name=$2 prompt_file=$3
    shift 3
    local repo=${SLURM_SUBMIT_DIR:?}
    mkdir -p "$dir"
    {
        echo "date: $(date -Is)"
        echo "slurm job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname)"
        echo "arm: $name  prompt file: ${prompt_file:-none}"
        echo "${RUN_INFO_EXTRA:-}"
        echo "git commit: $(git -C "$repo" rev-parse HEAD)  branch: $(git -C "$repo" rev-parse --abbrev-ref HEAD)"
        echo "git status:"
        git -C "$repo" status --short
    } > "$dir/run_info.txt"
    git -C "$repo" diff HEAD > "$dir/uncommitted.diff"
    cp "$ARMS" "$dir/arms.tsv"
    [ -z "$prompt_file" ] || cp "$prompt_file" "$dir/prompt.txt"
    [ $# -eq 0 ] || sha256sum "$@" > "$dir/problems.sha256"
}
