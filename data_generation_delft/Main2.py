# %%
from EPANETTools import *
from FileTools import *
import wntr
import sys
import pickle

# %%
RunId = int(sys.argv[1])
DirInput = sys.argv[2] + '/'
DirOutput = sys.argv[3]
FileInp = sys.argv[4]
Mod = sys.argv[5]

# %%
print(f'Run number {RunId}')
print(f'Path of input {DirInput}')
print(f'Outputs are located in {DirOutput}')
print(f'Input File Name {FileInp}')

File = DirInput + FileInp  # network_PDA.inp'  # File used for the model
print(File)
Net1 = wntr.network.WaterNetworkModel(File)
BD, MB, Lnz = BaseDemand(Net1)  # Average Demand of the system in m3/s
# Lnz will be were there is demand so is a sensor node

with open(DirInput + "wn.pkl", "wb") as f:
    pickle.dump(Net1, f)

Net1 = RunNet(Net1, "")
with open(DirInput + "ns.pkl", "wb") as f:
    pickle.dump(Net1, f)

print('Run cycle2')
RunCycle2_v2(Run=RunId, Threshold=0.5, DirPath=DirInput, Lnz=Lnz, ModName=Mod)
