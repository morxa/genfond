#!/bin/bash
#
#SBATCH --partition=rleap_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=%x-%j.out

# --signal=B:TERM@600 asks SLURM to send SIGTERM to this batch script 600s (10 minutes) before
# the --time limit is reached, so genfond (genfond/shutdown.py, wired up in __main__.py; see
# --max-wall-time in genfond/config/default.yaml) gets a chance to stop gracefully -- pickle its
# last policy, verify it, write a stats row -- instead of being SIGKILLed mid-round with nothing
# written. `B:` sends the signal to this batch shell only, not to any srun steps, so it has to be
# forwarded to the apptainer child explicitly here; apptainer in turn forwards it to genfond.
echo "Running: $@"
apptainer run --bind $PWD genfond_env.sif "$@" &
child=$!
trap 'echo "Forwarding SIGTERM to apptainer (pid $child)"; kill -TERM "$child" 2>/dev/null' TERM
wait "$child"
