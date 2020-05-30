import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


data_info = pd.read_csv('lending_club_info.csv', index_col='LoanStatNew')
df = pd.read_csv('lending_club_loan_two.csv')

# sns.countplot(x='loan_status', data=df)
# plt.figure(figsize=(12,4))
# sns.distplot(df['loan_amnt'], kde=False, bins=40)

# print(df.corr()['loan_amnt'].sort_values(ascending=False))
# plt.figure(figsize=(12,7))
# sns.heatmap(df.corr(), annot=True, cmap='viridis')

# feat_info('installment')
# feat_info('loan_amnt')

# sns.scatterplot(x='installment', y='loan_amnt', data=df, alpha=0.5)

# sns.boxplot(x='loan_status', y='loan_amnt', data=df)

# print(df.groupby('loan_status')['loan_amnt'].describe())
# print(df['grade'].unique())
# print(df['sub_grade'].unique())

# sns.countplot(x='grade', data=df, hue='loan_status')
# plt.figure(figsize=(12,4))
# sns.countplot(x='sub_grade', data=df, palette='coolwarm', order=sorted(df['sub_grade'].unique()))

df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
# df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

# print(100*(df.isnull().sum()/len(df)))
df = df.drop('emp_title', axis=1)
# sns.countplot(x='emp_length', data=df, order=['< 1 year',
#                                              '1 year',
#                                              '2 years',
#                                              '3 years',
#                                              '4 years',
#                                              '5 years',
#                                              '6 years',
#                                              '7 years',
#                                              '8 years',
#                                              '9 years',
#                                              '10+ years'], hue='loan_repaid')
# plt.show()
# emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
# emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
# print(emp_co/emp_fp)

df = df.drop('emp_length', axis=1)
# similar ratios between different loan statuses
df = df.drop('title', axis=1)
# similar to purpose also it has null values
# print(df['mort_acc'].value_counts())

# print(corr()['mort_acc'])      total_acc has highest correlation with mort_acc
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df = df.dropna()
# drop the NaN values as they are very less in number probably around 500
# print(df.isnull().sum())

# print(df.select_dtypes(['object']).columns)
# returns columns with string datatype

df['term'] = df['term'].apply(lambda term: int(term[:3]))
df = df.drop('grade', axis=1)

dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1), dummies], axis=1)

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

df['zip_code'] = df['address'].apply(lambda z: z[-5:])
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = pd.concat([df.drop('zip_code', axis=1), dummies], axis=1)
df = df.drop('address', axis=1)

df = df.drop('issue_d', axis=1)
# case of data leakage
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda d: int(d[-4:]))
df = df.drop('earliest_cr_line', axis=1)
df = df.drop('loan_status', axis=1)
#print(df['purpose'].value_counts())

# conversion of dataframe values to numpy array is important step as tensorflow only works on arrays
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
# first layer add same number of neurons as number of rows then decrease in the later layers
model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))
model.save('loan.h5')

losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
