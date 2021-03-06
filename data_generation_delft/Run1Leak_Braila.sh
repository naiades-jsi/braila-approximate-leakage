#!/bin/bash
#SBATCH -N 1 --ntasks=100
#SBATCH --time 1:00:00
#SBATCH --mem=16G
#SBATCH --array=110-220
#SBATCH --job-name=NAIADES_JOB_ZAN_11
#SBATCH --output=outBraila/NAIDES_%A_%a.out
#SBATCH --error=outBraila/NAIADES_%A_%a.err
#SBATCH --partition thin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stanonik

# Documentation about nodes:https://servicedesk.surfsara.nl/wiki/display/WIKI/Snellius+usage+and+accounting
# TODO reduce resource consumption to a minimum to be assigned space more frequently
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
cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"

echo "Parameters: $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod"
python Main.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod

echo "Finished Braila"