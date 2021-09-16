import matplotlib.pyplot as plt
import numpy as np

token_file = "./hmw1/num_tokens.txt" 
type_file = "./hmw1/num_types.txt"
time_file = "./hmw1/time_list.txt"

with open(token_file) as f:
    tokens = f.readlines()
    tokens = [token.rstrip() for token in tokens]

with open(type_file) as f:
    types = f.readlines()
    types = [typ.rstrip() for typ in types]

with open(time_file) as f:
    times = f.readlines()
    times = [tim.rstrip() for tim in times]

tokens = [int(tok) for tok in tokens]
types = [int(typ) for typ in types]
times = [float(t) for t in times]
slices = np.arange(0, 99, 1)

print(tokens[0], type(tokens[0]))
print(types[0], type(types[0]))
print(times[0], type(times[0]))

# plt.plot(tokens)
# plt.plot(types)
plt.plot(slices, times, 'r--')
plt.xlabel('number of slices')
plt.ylabel('time')
plt.show()

plt.plot(slices, tokens, 'bs', slices, types, 'g^')
# plt.plot(types)
# plt.plot(times)
plt.ylabel('counts of tokens/types')
plt.xlabel('number of slices')
plt.show()

# plt.plot(slices, np.log(tokens), 'bs', slices, np.log(types), 'g^')
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('logarithmic counts of tokens/types')
# plt.xlabel('number of slices')
# plt.show()

plt.plot(slices, tokens, 'bs')
# plt.plot(types)
# plt.plot(times)
plt.ylabel('counts of tokens')
plt.xlabel('number of slices')
plt.show()

plt.plot(slices, types, 'g^')
# plt.plot(types)
# plt.plot(times)
plt.ylabel('counts of types')
plt.xlabel('number of slices')
plt.show()