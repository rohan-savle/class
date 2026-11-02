import pandas as pd

import numpy as np
 
# Create a realistic employee dataset

data = {

    "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110],

    "Name": [

        "Amit Sharma", "Rahul Kumar", "Dip Singh", "Priya Verma",

        "Neha Khan", "Arjun Mehta", "Sara Ali", "Rohan Gupta",

        "Anjali Das", "Vikas Jain", "Vikas Jain"

    ],

    "Age": [

        25, 30, np.nan, 28, 35, 42, 29, np.nan, 31, 45, 45

    ],

    "Gender": [

        "Male", "male", "M", "Female", "FEMALE",

        "Male", "female", "M", "Female", "Male", "Male"

    ],

    "City": [

        "Delhi", "Delhi NCR", "Kolkata", "Mumbai", "Delhi",

        "Kolkata City", "Mumbai", "Delhi", "Kolkata", "Delhi", "Delhi"

    ],

    "Department": [

        "IT", "HR", "IT", "Finance", "HR",

        "IT", "Finance", "IT", "HR", "IT", "IT"

    ],

    "Salary": [

        55000, 45000, 60000, 50000, np.nan,

        75000, 65000, 58000, 48000, 999999, 999999

    ],

    "Experience": [

        2, 5, 3, 4, 8, 15, 6, 4, 7, 20, 20

    ],

    "Email": [

        "amit@gmail.com",

        "rahul@gmail.com",

        "dip@gmail.com",

        "priya@gmail.com",

        "neha@gmail.com",

        "arjun@gmail.com",

        "sara@gmail.com",

        "rohan@gmail.com",

        "anjali@gmail.com",

        "vikas@gmail.com",

        "vikas@gmail.com"

    ],

    "Date": [

        "2025-01-10", "2025-02-15", "2025-03-12",

        "2025-03-20", "2025-04-10", "2025-04-25",

        "2025-05-15", "2025-06-10", "2025-07-20",

        "2025-08-10", "2025-08-10"

    ]

}
 
df = pd.DataFrame(data)
 
print(df)

high_Salary = df[df['Salary'] > 60000]
print (high_Salary)



df[
    (df["Age"] > 25)
    |
    
    (df["City"] == "Delhi")
]

print(df)




df.isnull()
print(df.isnull())


df.isnull().sum
print(df.isnull().sum)


df_drop_Age = df.dropna(subset=["Age"])
print(df_drop_Age)



df['Salary']=df["Salary"].fillna(df["Salary"].median())

df_forward_fill = df.ffill()
print(df_forward_fill)


df_sorted = df.sort_values(by=["Salary"])
print(df_sorted)

df_sorted = df.sort_values(by=["Experience"])
print(df_sorted)




employee_info = pd.DataFrame({

    "Employee_ID": [101, 102, 103, 104, 105],

    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"]

})
 
salary_info = pd.DataFrame({

    "Employee_ID": [101, 102, 103, 104, 105],

    "Salary": [55000, 45000, 60000, 50000, 70000]

})


merged_df = pd.merge(employee_info, salary_info, on="Employee_ID", how="inner")