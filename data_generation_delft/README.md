# Instructions on how to connect to Snellius and run the data generation


## Preparations
1. Create a folder for the EPANET networks in "<home_directory>/NAIDADES/Start/Networks/" and copy the EPANET network files there.
>  mkdir -p NAIADES/Start/Networks

Command in script where it is used:
>  cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]}
  
2. Create a folder named "outGA" in the directory where the script was started.    
Command in script where it is used:
> SBATCH --output=outGA and SBATCH --error=outGA

3. Create a folder on root "/scratch-shared/NAIADES/<folder_name>"
> mkdir /scratch-shared/NAIADES/ijs_simulations_v1 or any other folder name
4. -ntask should never be lower than 25


## Execution
1. Use SLURM
2. Preload installed packages 
> load 2021
3. Load correct Python version
> load Python/3.9.5-GCCcore-10.3.0
4. Install necessary python packages (file tools, a local package):   
> pip install wntr plotly
5. Command to submit job to sbatch:
> sbatch Run1Leak.sh  


## Helpful SLURM command
Website https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

List all jobs from user:
>squeue -u <user>

Cancel a job:
>scancel <process id>

Cancel all jobs from user:
> scancel -u <user>




