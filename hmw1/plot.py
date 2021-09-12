import matplotlib.pyplot as plt

token_file = "./hmw1/num_tokens.txt" 
type_file = "./hmw1/num_types.txt"
time_tile = "./hmw1/time_list.txt"

with open(token_file) as f:
    tokens = f.readlines()

with open(type_file) as f:
    types = f.readlines()

with open(time_tile) as f:
    times = f.readlines()

tokens = [int(tok[:-2]) for tok in tokens]
types = [int(typ[:-2]) for typ in types]
times = [float(t[:-2]) for t in times]

plt.plot(tokens)
plt.ylabel('counts')
plt.show()