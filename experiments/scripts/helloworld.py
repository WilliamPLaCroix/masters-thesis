print("Here is a list")
list = [1,2,3,4,5]
print(list)

import pandas as pd
print("pandas imported, converting list to dataframe")
dataframe = pd.DataFrame(list)
print(dataframe)

import numpy as np
print("numpy imported, converting list to array")
array = np.array(list)
print(array)

print("completed without issue")