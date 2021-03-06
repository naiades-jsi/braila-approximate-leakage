# Instructions on how to connect to Snellius and run the data generation


## Preparations
1. Create a folder for the EPANET networks in "<home_directory>/NAIDADES/Start/Networks/" and copy the EPANET network files there.
>  mkdir -p NAIADES/Start/Networks

Command in script where it is used:
>  cp $HOME/NAIADES/Start/Networks/${NetworkRuns[$Mod]}
  
2. Create output folders named "outBraila" etc. in the directory where the script was started.    
Command in script where it is used (sixt of these folders need to be created):
> SBATCH --output=outBraila and SBATCH --error=outGA
 
` mkdir outBraila outCalAll outCalGa`   
`mkdir outTwoLeakBraila outTwoLeakCalAll outTwoLeakCalGa`

3. Create a folder on root "/scratch-shared/NAIADES/<folder_name>"
> mkdir /scratch-shared/NAIADES/ijs_simulations_v1 or any other folder name but code has to be changed then
`mkdir Braila CalAll CalGA`
4. -ntask should never be lower than 25


## Execution
1. Use SLURM
2. Preload installed packages 
> load 2021
3. Load correct Python version
> load Python/3.9.5-GCCcore-10.3.0
4. Install necessary python packages (file tools, a local package):   
> pip install wntr plotly
5. Commands to submit jobs for one leak to sbatch:
> sbatch Run1Leak_Braila.sh  
> sbatch Run1Leak_CalAll.sh  
> sbatch Run1Leak_CalGA.sh
6. Commands to submit jobs for two leaks to sbatch:
> sbatch Run2Leaks_Braila.sh  
> sbatch Run2Leaks_CalAll.sh  
> sbatch Run2Leaks_CalGA.sh


## Helpful SLURM command
Website https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

List all jobs from user:
>squeue -u <user>

Display more information about jobs:
> squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me

Cancel a job:
>scancel <process id>

Cancel all jobs with name:
> scancel --name < job_name >

Cancel all jobs from user:
> scancel -u < user >


## Data transferring

### Rsync
Use rync since it is faster and more robust to errors.
> rsync -avzr <username>@<server_adress>:/scratch-shared/NAIADES/ijs_simulations_v1/ \simulation & disown

To reroute the output to a file and put the process in the background execution:
> rsync -avzr < username >@snellius.surf.nl:/scratch-shared/NAIADES/ijs_simulations_v1/ \simulations_one_leak_4_25_2022 > logs_25_4.txt 2>&1 &

### Scp
Fastest method should be to use rsync, but if the server doesn't have it installed, we can use scp.
> scp -r  < username >@<server_adress>:/scratch-shared/NAIADES/ijs_simulations_v1/ \simulations\ & disown

Where the first path is the path on the linux server from which we are copying from and the second is the 
path on the windows machine where we want to copy to.

## Correct parameters to python

One leak:
Parameters: 111 /scratch-local/stanonik.809301 /scratch-shared/NAIADES/ijs_simulations_v1/Braila Braila_V2022.inp Braila
Two leaks:
Parameters: 1 /scratch-local/stanonik.809504 /scratch-shared/NAIADES/ijs_simulations_v1/Braila Braila_V2022.inp Braila


