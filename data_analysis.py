import pandas as pd
import numpy as np

data = pd.read_csv("seattle/TractTravelRawNumbers.csv", header=None).values

# numbers = data.to_numpy()
MU = 0.01
J_raw = -(data/2)*np.log(1-MU)
print(J_raw[0])
quit()
top_numbers = {}
avg_top = {}
avg = {}
count = 0
for row in numbers:
    # collect max three numbers
    top_numbers[count] = sorted(row, reverse=True)[:2]
    indeces = np.argsort(row)[-2:]
    print(top_numbers[count])
    print(indeces)
    # collect average of max three numbers
    # avg_top[count] = np.mean(sorted(row)[-2:])
    # # collect average of row
    # avg[count] = np.mean(row)
    # print(top_numbers[count])
    # print(indeces)
    # print('interpretation: max flows from tract {} to tract {} is {} people'.format(count, indeces[0], top_numbers[count][0]))
    # print(avg_top[count])
    # print(avg[count])
    count+=1
    quit()


# top_travels = np.max(data)
# print(top_travels)
