#!/bin/bash

#SBATCH --partition gpu_a100
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --ntasks 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu=1
#SBATCH --output /home/asanchez/chonky/tools/Boltz-2/output.log

# Date
datetime=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job started at: $datetime"

# Environment
source "$(conda info --base)/etc/profile.d/conda.sh" 
conda activate boltz2

# Arguments
yaml=""
out_dir=""
name=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--yaml) yaml="$2"; shift ;;
        -o|--out_dir) out_dir="$2"; shift ;;
        -n|--name) name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run
boltz predict "$yaml" \
              --out_dir "$out_dir" \
              --use_msa_server \
              --output_format mmcif 
              #--use_potentials

yaml_name=$(basename "$yaml")
mv "$out_dir"_"$yaml_name"/predictions/"$yaml_name"_model_0.mmcif" "$out_dir"/"$name".mmcif
