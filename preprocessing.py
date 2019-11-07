import pandas as pd

# Use of Imputer

df = pd.read_csv('E:/PyCharmProjects/NewPyProj/csv_data.csv')

print(df)

print(df.isnull())

print(df.isnull().sum())

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean')

imputer = imputer.fit(df.values)

imputed_data = imputer.transform(df.values)

print(imputed_data)

# Handling Categorical Data

df_cat = pd.DataFrame(data=[['green','M',10.1,'class1'],
                            ['blue','L',20.1,'class2'],
                            ['white','M',30.1,'class1']])
df_cat.columns = ['color','size','price','classlabel']

print(df_cat)

    # Mapping for Ordinal Features

size_mapping = {'M': 1, 'L':2}

df_cat['size'] = df_cat['size'].map(size_mapping)

print(df_cat)

    # LabelEncoder for Ordinal Features

from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

df_cat['classlabel'] = class_le.fit_transform(df_cat['classlabel'].values)

print(df_cat)

    # One-Hot Encoding for Nominal Categorical Variables

df_cat = pd.get_dummies(df_cat[['color', 'size', 'price']])

print(df_cat)

# Standardization

from sklearn.preprocessing import StandardScaler

df_stand = pd.read_csv('E:/PyCharmProjects/NewPyProj/stand.csv')

print(df_stand)

std = StandardScaler()

print(df_stand.shape)

print(df_stand['age'])

print(df_stand['age'].reshape(-1, 1))

df_stand['age'] = std.fit_transform(df_stand['age'].values.reshape(-1, 1))

print(df_stand['age'])

print(df_stand['salary'])

df_stand['salary'] = std.fit_transform(df_stand['salary'].values.reshape(-1, 1))

print(df_stand['salary'])

df_stand['purchased'] = class_le.fit_transform(df_stand['purchased'].values)

df_stand = pd.get_dummies(df_stand[['country', 'age', 'salary', 'purchased']], drop_first=True)

print(df_stand)

