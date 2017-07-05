#! /bin/bash

distrib="$1"
odydir="$2"
root=$(cd $odydir/../.. && pwd)
exec $root/launch $odydir/compute.py "$distrib" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
