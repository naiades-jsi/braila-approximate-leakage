# =============================================================================
# Libraries Required.
# =============================================================================
import copy

import wntr
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from platypus import NSGAII, Problem, Subset, unique, nondominated
import time
from collections import Counter
import pickle
# import plotly.express as px
# from plotly.offline import plot
import random


# from pyitlib import discrete_random_variable as drv


def RunNet(x, OutPath):
    '''
    This function receives as variable an instance of a network created with the 
    Net() function.
    This returns an object with all the results of the model, the queries have to 
    be related to the instance created with this function. 
    How to do the querries, visit: 
    https://wntr.readthedocs.io/en/latest/waternetworkmodel.html
    
    Example: Results = RunNet(MyInstance)'''
    sim = wntr.sim.EpanetSimulator(x)
    if OutPath == '':
        pre = OutPath + 'temp'
    else:
        pre = OutPath + '/temp'

    sim = sim.run_sim(version=2.2, file_prefix=pre)
    return sim


def BaseDemand(Net, y="No_defined"):
    '''
    This function rturns the base demand of the nodes y (in list format) to 
    be consulted. If any node is specified then all the base-demands are showed.       
    '''
    x = Net
    BD = []
    Listnz = []
    if y == "No_defined":
        for i in x.junction_name_list:
            Node = x.get_node(i)
            if Node.base_demand > 0:
                BD.append(Node.base_demand)
                Listnz.append(Node.name)
            else:
                BD.append(0)

    else:
        for i in y:
            Node = x.get_node(i)
            if Node.base_demand > 0:
                BD.append(Node.base_demand)
            else:
                BD.append(0)
    BD = np.array(BD)
    MBD = BD.mean()

    return BD, MBD, Listnz


def RunCycle_v2(Run=1, Threshold=0.5, DirPath='test', Lnz='', ModName='NotSent'):
    '''
    RunCycle(Run=1,nd=1,Method ='PDD',Threshold=0.5,DirPath='test'):
    '''
    number_of_days = 1  # number of days
    Method = 'PDD'

    start_time = time.time()  # start of the Timer, just for evaluate code performance.
    # print(start)

    with open(DirPath + 'wn.pkl', 'rb') as f:
        Net = pickle.load(f)

    with open(DirPath + 'ns.pkl', 'rb') as f:
        Net_Sim = pickle.load(f)

    Net.options.time.duration = number_of_days * 24 * 3600  # Time of simulation

    # New network setting (This will be used in for)
    base_leak_instance = copy.deepcopy(Net)
    # base_leak_instance.options.time.duration = 24 * number_of_days * 3600  # Time of simulation

    Net.options.hydraulic.demand_model = Method  # Using PDD as demand method
    St = int(Net.options.time.duration / Net.options.time.hydraulic_timestep)

    # Base pressures
    BPM = Net_Sim.node['pressure'].loc[1:St * 3600, Net.junction_name_list]
    BPM = BPM[Lnz]

    LPM = []  # Leak pressure matrix
    LM = []  # Leakage Matrix
    DM = []  # Divergence matrix
    main_data_dict = {}

    Leak_Nodes = Lnz  # (Net.junction_name_list
    Sensor_Nodes = Lnz
    print(len(Leak_Nodes))
    print(len(Sensor_Nodes))
    Leakmin = Run / 10
    Leakmax = (Run + 1) / 10
    # dl=(Leakmax-Leakmin)/10
    LeakFlows = np.arange(Leakmin, Leakmax, 0.001)

    to_row = St * 3600
    round_leak_to = 4
    for i in Leak_Nodes:
        start2 = time.time()
        for curr_leak_flow in range(len(LeakFlows)):
            # __________________
            LeakFlow = [LeakFlows[curr_leak_flow] / 1000] * (24 * number_of_days + 1)  # array of the leak flow (m3/s)

            Net = copy.deepcopy(base_leak_instance)
            Net.add_pattern(name='New', pattern=LeakFlow)  # Add New Patter To the model
            Net.get_node(i).add_demand(base=1, pattern_name='New')  # Add leakflow
            # print(f'before run leak node no={i}, leak flow{k}')
            Net_New = RunNet(Net, DirPath)  # Run new model
            # __________________________________________
            # Give name to the dataframe
            Net2 = Net_New.node['pressure'].loc[1:to_row, Sensor_Nodes]. \
                rename_axis('Node_' + i + ', ' + str(round(LeakFlows[curr_leak_flow], round_leak_to)) + 'LPS', axis=1)
            LPM.append(Net2[Lnz])  # Save pressure results

            Difference = BPM[Sensor_Nodes].sub(Net2[Lnz], fill_value=0)  # Create divergence matrix
            # Save Divergence M.
            DM.append(Difference.abs().rename_axis('Node_' + i + ', ' +
                                                   str(round(LeakFlows[curr_leak_flow], round_leak_to)) + 'LPS',
                                                   axis=1))

            lf = pd.DataFrame([k * 1000 for k in LeakFlow[1:]], columns=['LeakFlow']
                              , index=list(range(3600, to_row + 3600, 3600))).rename_axis('Node: ' + i, axis=1)
            LM.append(lf)  # Save leakflows used

            print(f'leakflow = {curr_leak_flow} and value {LeakFlows[curr_leak_flow]}, LeakNode={i}')
        print('____**____')
        print(f'All leaks nodes {i} Time= {time.time() - start2}')

    main_data_dict['LPM'] = LPM
    main_data_dict['DM'] = DM
    main_data_dict['LM'] = LM

    print(f'Finish Time 1= {time.time() - start_time}')
    TM_ = []  # time when the leak was identified
    WLM = []  # Water loss Matrix (L/s), how much water is wasted

    start_3 = time.time()
    # TODO this can be moved in the upper for, or calculated without for loop
    for i in range(len(DM)):
        TMtemp = set()
        WLMtemp = set()
        for j in Sensor_Nodes:
            WLMtemp2 = []
            for curr_leak_flow in range(len(DM[0])):
                if DM[i][j][(curr_leak_flow + 1) * 3600] <= Threshold:
                    WLMtemp2.append(LM[i].LeakFlow[(curr_leak_flow + 1) * 3600] * 3600)
                else:
                    WLMtemp2.append(LM[i].LeakFlow[(curr_leak_flow + 1) * 3600] * 3600)
                    break
            TMtemp.add(len(DM[0]) + 1)
            WLMtemp.add(sum(WLMtemp2))
        TM_.append(list(TMtemp))
        WLM.append(list(WLMtemp))
    print(f'Finish Time 2= {time.time() - start_3}')

    main_data_dict['LPM'] = LPM
    main_data_dict['DM'] = DM
    main_data_dict['LM'] = LM
    main_data_dict['Meta'] = {'Leakmin': Leakmin, 'Leakmax': Leakmax, 'Run': Run, 'Run Time': time.time() - start_time}
    main_data_dict['TM_l'] = TM_
    main_data_dict['WLM'] = WLM

    # Better way to save the data
    out_f_name = f"/scratch-shared/NAIADES/ijs_simulations_v1/{ModName}/1Leak_{str(Run)}_{ModName}_{str(Leakmin)}.pkl"
    with open(out_f_name, 'wb') as end_file:
        pickle.dump(main_data_dict, end_file)

    return "success"


def RunCycle2_v2(Run=1, Threshold=0.5, DirPath='test', Lnz='', ModName='NoNameSentToCycle2'):
    '''
    RunCycle2(Run=1,nd=1,Method ='PDD',Threshold=0.5,DirPath='test'):
     Two on cyclic and one random node
    '''
    number_of_days = 1  # days of simulation
    Method = 'PDD'

    start = time.time()  # start of the Timer, just for evaluate code performance.
    # print(start)

    with open(DirPath + 'wn.pkl', 'rb') as f:
        Net = pickle.load(f)

    with open(DirPath + 'ns.pkl', 'rb') as f:
        Net_Sim = pickle.load(f)

    Net.options.time.duration = number_of_days * 24 * 3600  # Time of simulation

    # New network setting (This will be used in for)
    base_leak_instance = copy.deepcopy(Net)

    Net.options.hydraulic.demand_model = Method  # Using PDD as demand method
    St = int(Net.options.time.duration / Net.options.time.hydraulic_timestep)

    # Base pressures
    BPM = Net_Sim.node['pressure'].loc[1:St * 3600, Net.junction_name_list]
    BPM = BPM[Lnz]

    LPM = []  # Leak pressure matrix
    LM = []  # Leakage Matrix
    DM = []  # Divergence matrix
    main_data_dict = {}

    Leak_Nodes = Lnz  # (Net.junction_name_list
    Sensor_Nodes = Lnz
    # print(len(Leak_Nodes))
    # print(len(Sensor_Nodes))
    Leakmin = Run / 10
    Leakmax = (Run + 1) / 10
    # dl=(Leakmax-Leakmin)/10
    LeakFlows = np.arange(Leakmin, Leakmax, 0.001)
    to_row = St * 3600
    round_leak_to = 4
    for i, NN in enumerate(Leak_Nodes):
        # Scenario 1
        # A leakflow randomly selected
        # ListName 2 =LN2
        TempLeak = Leak_Nodes   # TODO Not sure if this does a correct copy, it can just do a reference
        TempLeak.pop(i)
        print(i)
        # print(LN2)
        print(f"the node before cycle is {NN}")
        for NN2 in TempLeak:
            start2 = time.time()
            for k in range(len(LeakFlows)):
                # __________________
                LeakFlow = [LeakFlows[k] / 1000] * (24 * number_of_days + 1)  # array of the leak flow (m3/s)
                Net = copy.deepcopy(base_leak_instance)
                Net.add_pattern(name='New', pattern=LeakFlow)  # Add New Patter To the model
                Net.get_node(NN2).add_demand(base=1, pattern_name='New')  # Add leakflow

                Net.get_node(NN).add_demand(base=1, pattern_name='New')  # Add leakflow
                # print(f'before run leak node no={i}, leak flow{k}')
                Net_New = RunNet(Net, DirPath)  # Run new model
                # __________________________________________
                # Give name to the dataframe
                Net2 = Net_New.node['pressure'].loc[1:to_row, Sensor_Nodes].rename_axis('Node_' + NN + '-' + NN2 +
                                                                                           ',' + str(
                    round(LeakFlows[k], round_leak_to)) + 'LPS', axis=1)

                LPM.append(Net2[Lnz])  # Save pressure results

                Difference = BPM[Sensor_Nodes].sub(Net2[Lnz], fill_value=0)  # Create divergence matrix

                # Save Divergence M.
                DM.append(Difference.abs().rename_axis('Node_' + NN + '-' + NN2 + ', '
                                                       + str(round(LeakFlows[k], round_leak_to)) + 'LPS',
                                                       axis=1))

                lf = pd.DataFrame([k * 1000 for k in LeakFlow[1:]], columns=['LeakFlow'],
                                  index=list(range(3600, to_row + 3600, 3600))). \
                    rename_axis('Node: ' + NN + '-' + NN2, axis=1)
                LM.append(lf)  # Save leakflows used
            print(f'leakflow = {len(LeakFlows) - 1} and value {LeakFlows[len(LeakFlows) - 1]}, LeakNode={NN} with LN2{NN2}')
        print(f'Finish Time 1= {time.time() - start}')
        TempLeak = Leak_Nodes
        print('____**____')
        print(f'All leaks nodes {NN}-{NN2} Time= {time.time() - start2}')

    TM_ = []  # time when the leak was identified
    WLM = []  # Water loss Matrix (L/s), how much water is wasted

    # TODO same as for one node leak
    for i in range(len(DM)):
        TMtemp = []
        WLMtemp = []
        for j in Sensor_Nodes:
            WLMtemp2 = []
            for k in range(len(DM[0])):
                if DM[i][j][(k + 1) * 3600] <= Threshold:
                    WLMtemp2.append(LM[i].LeakFlow[(k + 1) * 3600] * 3600)
                else:
                    WLMtemp2.append(LM[i].LeakFlow[(k + 1) * 3600] * 3600)
                    break
            TMtemp.append(k + 1)
            WLMtemp.append(sum(WLMtemp2))
        TM_.append(TMtemp)
        WLM.append(WLMtemp)
    print(f'Finish Time 2= {time.time() - start2}')

    main_data_dict['LPM'] = LPM
    main_data_dict['DM'] = DM
    main_data_dict['LM'] = LM
    main_data_dict['Meta'] = {'Leakmin': Leakmin, 'Leakmax': Leakmax, 'Run': Run, 'Run Time': time.time() - start2,
                 'Two Nodes': "Yes"}
    main_data_dict['TM_l'] = TM_
    main_data_dict['WLM'] = WLM

    out_f_name = f"/scratch-shared/NAIADES/ijs_simulations_v1/{ModName}/2Nodes_{str(Run)}_{ModName}_{str(Leakmin)}.pkl"
    with open(out_f_name, 'wb') as f:
        pickle.dump(main_data_dict, f)

    return 'Success'
