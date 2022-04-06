#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_22
#SBATCH -N 4 --ntasks=25
#SBATCH --time  3:00:00
#SBATCH --output=outCalAll/NAIDES_%A_%a.out
#SBATCH --error=outCalAll/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --mem=4G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stanonik
#SBATCH --array=110-220

module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["CalAll"]="CalAll.inp"

#--------------------------
Mod='CalAll'
#--------------------------
echo 'Running WDN - ' ${NetworkRuns[$Mod]}

# Change this in EPanettools also
DataDirectory="/scratch-shared/NAIADES/ijs_simulations_v1/$Mod"

#-------------------------------------------------
cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"


python Main.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod


echo "Finished CalAll"