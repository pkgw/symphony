#! /bin/bash

root=$(cd $(dirname $0)/../.. && pwd)
exec $root/launch $(dirname $0)/compute.py "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
