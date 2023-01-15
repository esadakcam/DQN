import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('waterworld_rewards.csv',index_col=None)
plt.plot(df['Reward'])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
input()