import numpy as np


data = np.load("/home/miguel/Desktop/PIPES/1575380167829077_train.npy",allow_pickle=True)  # xyzrgbl, N*7

print(type(data))
print(len(data))
print(data[0])
print(type(data[0][0]))
print(type(data[0][1]))
print(type(data[0][2]))
print(type(data[0][3]))
print(type(data[0][4]))
print(type(data[0][5]))
print(type(data[0][6]))

idx_list = []
for i in range(len(data)):
	if data[i][6] not in idx_list:
		idx_list.append(data[i][6])
print(idx_list)

