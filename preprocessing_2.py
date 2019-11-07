import pandas as pd

# 1. Encoding Ordinal Data

data = pd.DataFrame({'Exam_Grade' : ['A', 'B', 'A', 'C', 'A']})
print(data)

mapping = {'A' : 5, 'B' : 4, 'C' : 3, 'D' : 2, 'F' : 1 }

data['Exam_Grade'] = data['Exam_Grade'].map(mapping)

print(data)

# 2. Encoding Nominal Data

# 2.1 One-Hot Encoding

data = pd.DataFrame({'Car_Manufacturer' : ['Toyota', 'Ford', 'Ford', 'Marcedes', 'Ford']})

print(data)

one_hot_encodings = pd.get_dummies(data, columns=['Car_Manufacturer'])

print(one_hot_encodings)

# 2.2 Label Encoding

import numpy as np

data = pd.DataFrame({'Car_Manufacturer' : ['Toyota', 'Ford', 'Ford', 'Marcedes', 'Ford']})

mapping = {label : idx for idx, label in enumerate(np.unique(data['Car_Manufacturer'].dropna()))}

print(mapping)

data['Car_Manufacturer'] = data['Car_Manufacturer'].map(mapping)

print(data)

#         Using sklearn

from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({'Car_Manufacturer' : ['Toyota', 'Ford', 'Ford', 'Marcedes', 'Ford']})

encoder = LabelEncoder().fit(data['Car_Manufacturer'])
data['Car_Manufacturer'] = encoder.transform(data['Car_Manufacturer'])

print(data)

# 2.3 Target Encoding

# pip install category_encoders
import category_encoders as ce

data = pd.DataFrame({'State' : ['CA', 'NY', 'CA', 'TX', 'TX', 'NY', "NY"],
                     'Income' : [70000, 64000, 72000, 59000, 57000, 67000, 62000]})
print(data)

encoder = ce.target_encoder.TargetEncoder(cols=['State'])
encoder.fit(data['State'], data['Income'])

data['State'] = encoder.transform(data['State'])


data['State'] = data['State'].round() # rounding for simplification

print(data)






