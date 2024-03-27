import pandas as pd

# 创建数据
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

# 将DataFrame转换为LaTeX表格
table_tex = df.to_latex()
print(table_tex)