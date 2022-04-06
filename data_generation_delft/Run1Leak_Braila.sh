#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_11
#SBATCH -N 4 --ntasks=25
#SBATCH --time  3:00:00
#SBATCH --output=outBraila/NAIDES_%A_%a.out
#SBATCH --error=outBraila/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --mem=4G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stanonik
#SBATCH --array=110-220

module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["Braila"]="Braila_V2022.inp"

Mod='Braila'
echo 'Running WDN - ' ${NetworkRuns[$Mod]}

# TODO merge files and make this all modular
# Change this in EPanettools also
DataDirectory="/scratch-shared/NAIADES/ijs_simulations_v1/$Mod"

#-------------------------------------------------
cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"


python Main.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod

echo "Finished Braila"