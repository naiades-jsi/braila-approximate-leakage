# Instructions on how to connect to Snellius and run the data generation


## Preparations
1. Create a folder for the output in "<home_directory>
/NAIDADES/Start/Networks/" and copy the EPANET network files there.
>  mkdir -p NAIADES/Start/Networks

Command in script where it is used:
>  cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]}
  
2. Create a folder named "outGA" in the directory where the script was started 
Command in script where it is used:
> SBATCH --output=outGA and SBATCH --error=outGA

3. Create a folder on root "/scratch-shared/NAIADES/<folder_name>"
> mkdir /scratch-shared/NAIADES/ijs_simulations_v1 or any other folder name
4. -ntask should never be lower than 25

## Execution
1. Use SLURM
2. Command:
> sbatch Run1Leak.sh