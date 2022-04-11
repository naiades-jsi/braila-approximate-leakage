#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_TWOLEAK_33
#SBATCH -N 4 --ntasks=50
#SBATCH --time  24:00:00
#SBATCH --output=outTwoLeakCalGa/NAIDES_%A_%a.out
#SBATCH --error=outTwoLeakCalGa/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --cpus-per-task 1
#SBATCH --mem=4G
#SBATCH --mail-user=stanonik
#SBATCH --array=1-201
module load 2021
module load Python/3.9.5-GCCcore-10.3.0


#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["CalGA"]="Cal_A_RN1_GA.inp"

Mod='CalGA'

echo "Running WDN Network - " $Mod
echo 'Running WDN - ' ${NetworkRuns[$Mod]}
echo "$SLURM_ARRAY_TASK_ID"

cp  $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"
DataDirectory="/scratch-shared/NAIADES/ijs_simulations_v1/$Mod"
python Main2.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod
