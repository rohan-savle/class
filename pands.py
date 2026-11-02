import pandas as  pd

import os


marks = pd.Series([78,85,92,67,88])



print(marks)

marks = pd.Series([78,85,25,25,6,4,90], index = ['a','b','c','d','e','f','g',])

print("\nseries with customer index")

print(marks)

print("\nvalue of index 'b' is ",marks['b'])

print("\nvalue of index 'c' is ",marks['c'])

print("\nfirst value using loc: ")
print(marks.loc['a'])


data = {
    "Name":['johan','rahul','sachin','dada'],
    "Age":[12,15,26,25],
    "Marks":[78,85,92,67]
}

df = pd.DataFrame(data)
print("\nDataframe:")
print(df)


print("\nfirst 2 rows:")
print(df.head(2))


print("\nShape:")
print(df.shape)

print("\nData Types:")

print(df.dtypes)
 
print("\nInformation:")

df.info()
 
print("\nStatistical Summary:")

print(df.describe())



print("\nfirst row.")
print(df.iloc[0])

print("\nfirst row and third column.")
print(df.iloc[1,2])

print("\nfirst row using loc.")
print(df.loc[2])

print("\nName of first Student using loc.")
print(df.loc[3, 'Name'])


#Replace "students.csv" with your actual file path
 df_csv = pd.read_csv("students.csv")
 df_csv = pd.read_csv("students.csv")

print("\nCSV data:")
print(df_csv)