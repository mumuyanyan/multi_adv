import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def smooth(data,weight=0.9,var = 'Value'):
    scalar = data[var].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return pd.DataFrame(smoothed)

labels = ['MARL-ACBQ','MARL-Lagrange','MARL-Barrier',
          'MARL-Reachability','MARL-CBQ','MARL-ABQ']

colors = ['r','dodgerblue','limegreen','violet','gray','m']

reward_files = []




reward_files.append(['maddpg_acbq_reward1.csv',
                     'maddpg_acbq_reward2.csv',
                     'maddpg_acbq_reward3.csv'])
reward_files.append(['maddpg_cost_reward1.csv',
                     'maddpg_cost_reward4.csv',
                     'maddpg_cost_reward3.csv'])
reward_files.append(['maddpg_noreach_reward1.csv',
                     'maddpg_noreach_reward2.csv',
                     'maddpg_noreach_reward3.csv'])
reward_files.append(['maddpg_nocbf_reward1.csv',
                     'maddpg_nocbf_reward2.csv',
                     'maddpg_nocbf_reward3.csv'])
reward_files.append(['maddpg_cbq_reward1.csv',
                     'maddpg_cbq_reward2.csv',
                     'maddpg_cbq_reward3.csv'])
reward_files.append(['maddpg_abq_reward1.csv',
                     'maddpg_abq_reward2.csv',
                     'maddpg_abq_reward3.csv'])
rewards_all = []
for i in range(len(reward_files)):
    rewards = []
    successes = []
    for j in range(len(reward_files[i])):
        reward_one = pd.read_csv(reward_files[i][j])
        reward_one['smooth'] = smooth(reward_one, 0.85)
        

        rewards.append(reward_one)

    rewards_all.append(rewards)

rewards_all_np = []
successes_all_np = []
for i in range(len(rewards_all)):
    rewards_np = []
    successes_np = []
    for j in range(len(rewards_all[i])):
        if i == 4:
            rewards_np.append(np.array(rewards_all[i][j]-1))
        else:
            rewards_np.append(np.array(rewards_all[i][j]))


    rewards_all_np.append(
        np.concatenate([one[:,3:] for one in rewards_np],axis=1)
    )

plts = []

x_np = np.squeeze(np.array(rewards_all[0][0]['Step']))
total_steps = int(20000/30000*1000)
for i in range(len(rewards_all_np)-1,-1,-1):
    s_mean = np.mean(rewards_all_np[i], axis=1)
    s_std = np.std(rewards_all_np[i], axis=1)
    s_max = s_mean + s_std * 0.95
    s_min = s_mean - s_std * 0.95
    p,=plt.plot(x_np[:total_steps], s_mean[:total_steps], label=labels[i], color=colors[i])
    plts.append(p)

    plt.fill_between(x_np[:total_steps], s_max[:total_steps], s_min[:total_steps], alpha=0.3, color=colors[i],edgecolor=None)

plt.grid(linestyle="--")

plts.reverse()
plt.legend(plts,labels,loc='lower right')
plt.ylim(-16,15)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('Episodes')
plt.ylabel('Average Episode Reward')


plt.savefig('barrier_all_rewards.pdf',dpi=1200,format='pdf')
plt.savefig('barrier_all_rewards.png',dpi=400,format='png')
plt.savefig('barrier_all_rewards.svg',dpi=400,format='svg')

plt.show()

print('hello')