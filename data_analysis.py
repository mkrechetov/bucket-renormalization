import pandas as pd
import numpy as np

data = pd.read_csv("seattle/TractTravelRawNumbers.csv", header=None)

numbers = data.to_numpy()

top_numbers = {}
avg_top = {}
avg = {}
count = 0
for row in numbers:
    # collect max three numbers
    top_numbers[count] = sorted(row)[-3:]
    avg_top[count] = np.mean(sorted(row)[-3:])
    avg[count] = np.mean(row)
    print(top_numbers[count])
    print(avg_top[count])
    print(avg[count])
    count+=1
    # quit()
    # collect average of max three numbers
    # collect average of row
# top_travels = np.max(data)
# print(top_travels)
