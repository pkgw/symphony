#!/bin/bash

if [ "$1" = test ] ; then
    # 0.05 cpu-hours
    array_spec=0-1
    time_lim=1
    partition=interact
    set -x
elif [ "$1" = prod ] ; then
    # ~1000 cpu-hours
    array_spec=0-1023
    time_lim=60
    partition=serial_requeue
elif [ "$1" = med ] ; then
    # ~20 cpu-hours
    array_spec=0-128
    time_lim=10
    partition=serial_requeue
else
    echo >&2 "first argument must be either \"test\" or \"prod\" or \"med\""
    exit 1
fi

case "$2" in
    powerlaw|pitchy|pitchy-jv|plfaraday|pitchy-f) ;;
    *)
	echo >&2 "distribution type \"$2\" not recognized"
	exit 1 ;;
esac

mod=intel/17.0.2-fasrc01

if ldd $(dirname $0)/../symphonyPy.so |grep -q 'not found' ; then
    echo "Compiler module $mod needs loading."
    module load $mod
    if ldd $(dirname $0)/../symphonyPy.so |grep -q 'not found' ; then
	echo >&2 "error: still have dylib problems"
	exit 1
    fi
fi

exec sbatch \
    -a $array_spec \
    -o '%A.out' \
    -J symphony \
    --mem=512 \
    -N 1 \
    -n 1 \
    -p $partition \
    -t $time_lim \
    -o '%A.log' \
    --open-mode=append \
    $(dirname $0)/runner.sh "$2" $(cd $(dirname $0) && pwd)
