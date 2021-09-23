import matplotlib.pyplot as plt
import numpy as np

# lowercase_type_file = "./hmw1/lowercase_num_types.txt"
# lowercase_time_file = "./hmw1/lowercase_time_list.txt"

# remove_punc_type_file = "./hmw1/remove_punct_num_types.txt"
# remove_punc_time_file = "./hmw1/remove_punc_time_list.txt"

# replace_num_type_file = "./hmw1/replace_num_num_types.txt"
# replace_num_time_file = "./hmw1/replace_num_time_list.txt"

# replace_url_type_file = "./hmw1/replace_url_num_types.txt"
# replace_url_time_file = "./hmw1/replace_url_time_list.txt"


# with open(lowercase_type_file) as f:
#     lowercase_types = f.readlines()
#     lowercase_types = [typ.rstrip() for typ in lowercase_types]

# with open(lowercase_time_file) as f:
#     lowercase_times = f.readlines()
#     lowercase_times = [tim.rstrip() for tim in lowercase_times]

# with open(remove_punc_type_file) as f:
#     remove_punc_types = f.readlines()
#     remove_punc_types = [typ.rstrip() for typ in remove_punc_types]

# with open(remove_punc_time_file) as f:
#     remove_punc_times = f.readlines()
#     remove_punc_times = [tim.rstrip() for tim in remove_punc_times]

# with open(replace_num_type_file) as f:
#     replace_num_types = f.readlines()
#     replace_num_types = [typ.rstrip() for typ in replace_num_types]

# with open(replace_num_time_file) as f:
#     replace_num_times = f.readlines()
#     replace_num_times = [tim.rstrip() for tim in replace_num_times]

# with open(replace_url_type_file) as f:
#     replace_url_types = f.readlines()
#     replace_url_types = [typ.rstrip() for typ in replace_url_types]

# with open(replace_url_time_file) as f:
#     replace_url_times = f.readlines()
#     replace_url_times = [tim.rstrip() for tim in replace_url_times]


# lowercase_types = [int(typ) for typ in lowercase_types]
# lowercase_times = [float(t) for t in lowercase_times]

# remove_punc_types = [int(typ) for typ in remove_punc_types]
# remove_punc_times = [float(t) for t in remove_punc_times]

# replace_num_types = [int(typ) for typ in replace_num_types]
# replace_num_times = [float(t) for t in replace_num_times]

# replace_url_types = [int(typ) for typ in replace_url_types]
# replace_url_times = [float(t) for t in replace_url_times]

# slices = np.arange(0, 99, 1)
slices = np.arange(1, 10, 1)

# print(tokens[0], type(tokens[0]))
# print(types[0], type(types[0]))
# print(times[0], type(times[0]))

# # plt.plot(tokens)
# # plt.plot(types)
# # plt.plot(slices, times, 'r--')
# # plt.xlabel('number of slices')
# # plt.ylabel('time')
# # plt.show()
# plt.plot(slices, lowercase_types, 'bs', slices, remove_punc_types, 'g^', slices, replace_num_types, 'r^', slices, replace_url_types, 'y^',)
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('vocabulary size')
# plt.xlabel('number of slices')
# plt.show()
train_time = [6.129, 10.057, 13.563, 17.182, 21.927, 24.023, 25.529, 30.420, 33.667]
train_time = [float(t) for t in train_time]

disk_size = [42, 67, 88, 106, 123, 139, 154, 168, 182]
disk_size = [float(d) for d in disk_size]

ave_perplexity = [519.21, 459.77, 452.84, 445.56, 433.43, 432.02, 418.99, 416.19, 412.93]
ave_perplexity = [float(p) for p in ave_perplexity]

plt.plot(slices, train_time, 'bs')
plt.ylabel('train time')
plt.xlabel('number of slices')
plt.show()

plt.plot(slices, disk_size, 'bs')
plt.ylabel('disk size')
plt.xlabel('number of slices')
plt.show()

plt.plot(slices, ave_perplexity, 'bs')
plt.ylabel('Average perplexity')
plt.xlabel('number of slices')
plt.show()



# plt.plot(slices, lowercase_types, 'bs', slices, remove_punc_types, 'g^', slices, replace_num_types, 'r^', slices, replace_url_types, 'y^',)
# plt.ylabel('vocabulary size')
# plt.xlabel('number of slices')
# plt.show()


# plt.plot(slices, lowercase_times, 'bs', slices, remove_punc_times, 'g^', slices, replace_num_times, 'r^', slices, replace_url_times, 'y^',)
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('time taken to preprocess')
# plt.xlabel('number of slices')
# plt.show()

# plt.plot(slices, lowercase_times, 'bs', slices, remove_punc_times, 'g^', slices, replace_num_times, 'r^', slices, replace_url_times, 'y^',)
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('time taken to preprocess')
# plt.xlabel('number of slices')
# plt.show()

# plt.plot(slices, np.log(tokens), 'bs', slices, np.log(types), 'g^')
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('logarithmic counts of tokens/types')
# plt.xlabel('number of slices')
# plt.show()

# plt.plot(slices, tokens, 'bs')
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('counts of tokens')
# plt.xlabel('number of slices')
# plt.show()

# plt.plot(slices, types, 'g^')
# # plt.plot(types)
# # plt.plot(times)
# plt.ylabel('counts of types')
# plt.xlabel('number of slices')
# plt.show()