#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_11
#SBATCH -N 4 --ntasks=50
#SBATCH --time  24:00:00
#SBATCH --output=outGA/NAIDES_%A_%a.out
#SBATCH --error=outGA/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --cpus-per-task 1
#SBATCH --mem=4G
#SBATCH --mail-user=stanonik
#SBATCH --array=1-201
module load 2021
module load Python/3.9.5-GCCcore-10.3.0


#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["Braila"]="Braila_V2022.inp"
NetworkRuns["CalGA"]="Cal_A_RN1_GA.inp"
NetworkRuns["CalRN2"]="Cal_A_RN1_RN2.inp"
NetworkRuns["CalAll"]="CalAll.inp"

Mod='CalGA'

echo "Running WDN Network - " $Mod
echo 'Running WDN - ' ${NetworkRuns[$Mod]}
echo "$SLURM_ARRAY_TASK_ID"

cp  $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"
DataDirectory="/scratch-shared/NAIADES/$Mod"
python Main2.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod
