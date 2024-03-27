import pandas as pd

# 读取BGL文件，并取前100000行
bgl_df = pd.read_csv('Dataset/BGL.log_structured.csv', nrows=100001)
# 将取出的数据保存到新的CSV文件中
bgl_df.to_csv('Dataset/test_BGL.log_structured.csv', index=False)

# 读取Thunderbird文件，并取前100000行
thunderbird_df = pd.read_csv('Dataset/Thunderbird.log_structured.csv', nrows=100001)
# 将取出的数据保存到新的CSV文件中
thunderbird_df.to_csv('Dataset/test_Thunderbird.log_structured.csv', index=False)

print("Files have been created with the first 100,000 rows.")
