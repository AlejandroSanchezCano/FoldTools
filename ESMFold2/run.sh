#!/bin/bash

#SBATCH --partition gpu_a100
#SBATCH --nodes 1
#SBATCH --time 24:05:00
#SBATCH --ntasks 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu=1
#SBATCH --output /home/asanchez/chonky/tools/ESMFold2/output.log

# Date
datetime=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job started at: $datetime"

# Environment
source "$(conda info --base)/etc/profile.d/conda.sh" 
conda activate biohub

# Arguments
fasta=""
out_dir="."
name=""
biohub=""
fast=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fasta) fasta="$2"; shift ;;
        -o|--out_dir) out_dir="$2"; shift ;;
        -n|--name) name="$2"; shift ;;
        -b|--biohub) biohub=--biohub ;;
        -F|--fast) fast=--fast ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use job array index if available
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    idx=${SLURM_ARRAY_TASK_ID}
    #fasta=${fasta}/${idx}.fasta
    name=${idx}
    echo 'Running idx: ' $idx
fi

# Run
python -u ~/chonky/tools/ESMFold2/run.py \
            --fasta "$fasta" \
            --out_dir "$out_dir" \
            --name "$name" \
            $biohub \
            $fast
echo "Finished job at $(date '+%Y-%m-%d %H:%M:%S')"
