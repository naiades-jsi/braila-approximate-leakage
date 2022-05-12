#!/bin/bash
#SBATCH --job-name=NAIADES_JOB_ZAN_TWOLEAK_11 -> Give job a name
#SBATCH -N 4 --ntasks=50  -> Allocate 4 nodes to the job each with 50 tasks
#SBATCH --time  24:00:00  -> Set max time to 24 hours
#SBATCH --output=outTwoLeakBraila/NAIDES_%A_%a.out  -> Set output directory and file name for .out files
#SBATCH --error=outTwoLeakBraila/NAIADES_%A_%a.err  -> Set output directory and file name for .err files
#SBATCH --partition thin  -> Partition to use (thin, fat, long), this is suppose to alloce resources to the nodes
#SBATCH --mem=32G -> Set max memory usage to 32 GB
#SBATCH --mail-user=stanonik  # Email me when job ends (doesn't work)
#SBATCH --array=1-201 # Set the number of jobs to run
module load 2021
module load Python/3.9.5-GCCcore-10.3.0   # Loading all modules, and python, I have installed my packages with pip already


#NameoftheSystem
declare -A NetworkRuns
NetworkRuns["Braila"]="Braila_V2022.inp"    # Saving the name of the network and file name in a dictionary

Mod='Braila'

echo "Running WDN Network - " $Mod        # just prints
echo 'Running WDN - ' ${NetworkRuns[$Mod]}
echo "$SLURM_ARRAY_TASK_ID"

cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]} "$TMPDIR" # Copying the network to the tmp directory
DataDirectory="/scratch-shared/NAIADES/ijs_simulations_v1/$Mod" # Specifying the directory where the data will be saved
echo "Parameters: $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod"  # Printing the parameters

python Main2.py $SLURM_ARRAY_TASK_ID $TMPDIR $DataDirectory ${NetworkRuns[$Mod]} $Mod # Running the python script, with provided parameters
# Usually the parameters would be something like this:
# 111 /scratch-local/stanonik.809301 /scratch-shared/NAIADES/ijs_simulations_v1/Braila Braila_V2022.inp Braila
echo "Finished"
