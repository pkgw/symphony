#! /bin/bash

odydir="$1"
root=$(cd $odydir/../.. && pwd)
exec $root/launch $odydir/compute.py "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
