#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_TWOLEAK_22
#SBATCH -N 4 --ntasks=50
#SBATCH --time  24:00:00
#SBATCH --output=outTwoLeakCalAll/NAIDES_%A_%a.out
#SBATCH --error=outTwoLeakCalAll/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --cpus-per-task 1
#SBATCH --mem=32G
#SBATCH --mail-user=stanonik
#SBATCH --array=1-201
module load 2021
module load Python/3.9.5-GCCcore-10.3.0


#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["CalAll"]="CalAll.inp"

Mod='CalAll'

echo "Running WDN Network - " $Mod
echo 'Running WDN - ' ${NetworkRuns[$Mod]}
echo "$SLURM_ARRAY_TASK_ID"

cp  $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"
DataDirectory="/scratch-shared/NAIADES/ijs_simulations_v1/$Mod"
echo "Parameters: $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod"

python Main2.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod
echo "Finished"
