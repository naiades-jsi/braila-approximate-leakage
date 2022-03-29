#!/bin/bash
#SBATCH --job-name=NAIADES2
#SBATCH -N 4 --ntasks=25
#SBATCH --time  1:00:00
#SBATCH --output=outGA/NAIDES_%A_%a.out
#SBATCH --error=outGA/NAIADES_%A_%a.err
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
NetworkRuns["CalGA"]="Cal_A_RN1_GA.inp"
NetworkRuns["CalRN2"]="Cal_A_RN1_RN2.inp"
NetworkRuns["CalAll"]="CalAll.inp"

#Mod='Braila'
Mod='CalGA'
#Mod='CalRN2'
#--------------------------
#Mod='CalAll'
#--------------------------
echo 'Running WDN - ' ${NetworkRuns[$Mod]}

DataDirectory="/scratch-shared/NAIADES/$Mod"


#cp  $HOME/NAIADES/Start/Networks/Cal_A_RN1_GA.inp "$TMPDIR"
#cp  $HOME/NAIADES/Start/Networks/Braila_V2022.inp "$TMPDIR"
#cp  $HOME/NAIADES/Start/Networks/Cal_All.inp "$TMPDIR"
#-------------------------------------------------
cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR"


python Main.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod


echo "Finished"