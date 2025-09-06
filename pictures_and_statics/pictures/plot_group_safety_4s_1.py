import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
labels = ['MARL-ACBQ','MARL-Lagrange','MARL-Barrier',
          'MARL-Reachability','MARL-CBQ','MARL-ABQ']

# colors = ['r','dodgerblue','limegreen','violet']
colors = ['r','dodgerblue','limegreen','violet','gray','m']

data1 = []
data1.append(np.load('stat4/4_acbq_safety_4s.npy'))
data1.append(np.load('stat4/4_cost_safety_4s.npy'))
data1.append(np.load('stat4/4_noreach_safety_4s.npy'))
data1.append(np.load('stat4/4_nocbf_safety_4s.npy'))
data1.append(np.load('stat4/4_cbq_safety_4s.npy'))
data1.append(np.load('stat4/4_abq_safety_4s.npy'))
data1_s = []
data1_s.append(np.load('stat4/4_acbq_steps_4s.npy'))
data1_s.append(np.load('stat4/4_cost_steps_4s.npy'))
data1_s.append(np.load('stat4/4_noreach_steps_4s.npy'))
data1_s.append(np.load('stat4/4_nocbf_steps_4s.npy'))
data1_s.append(np.load('stat4/4_cbq_steps_4s.npy'))
data1_s.append(np.load('stat4/4_abq_steps_4s.npy'))
print('adv=2')
for i in range(len(data1)):
    print(str(labels[i]),' safety: ',np.mean(data1[i]),' step: ',np.mean(data1_s[i]))
data2 = []
data2.append(np.load('stat4/4_random_acbq_safety_4s.npy'))
data2.append(np.load('stat4/4_random_cost_safety_4s.npy'))
data2.append(np.load('stat4/4_random_noreach_safety_4s.npy'))
data2.append(np.load('stat4/4_random_nocbf_safety_4s.npy'))
data2.append(np.load('stat4/4_random_cbq_safety_4s.npy'))
data2.append(np.load('stat4/4_random_abq_safety_4s.npy'))
data2_s = []
data2_s.append(np.load('stat4/4_random_acbq_steps_4s.npy'))
data2_s.append(np.load('stat4/4_random_cost_steps_4s.npy'))
data2_s.append(np.load('stat4/4_random_noreach_steps_4s.npy'))
data2_s.append(np.load('stat4/4_random_nocbf_steps_4s.npy'))
data2_s.append(np.load('stat4/4_random_cbq_steps_4s.npy'))
data2_s.append(np.load('stat4/4_random_abq_steps_4s.npy'))
print('adv=1')
for i in range(len(data2)):
    print(str(labels[i]),' safety: ',np.mean(data2[i]),' step: ',np.mean(data2_s[i]))

print('hello')