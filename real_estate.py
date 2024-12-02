import pickle, json
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
pd.set_option('display.max_columns', 200)


df = pd.read_csv('Bengaluru_House_Data.csv')

# print(df.columns)
# print(df.shape)
df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1, inplace=True)

## Data Cleaning ##

# print(df.isnull().sum())
df.bath = df.bath.fillna(df.bath.median())
df.dropna(inplace=True)

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1])/2)
    try:
        return float(x)
    except:
        return None

df.total_sqft = df.total_sqft.apply(convert_sqft_to_num)

## Feature Engineering ##

df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

df.location = df.location.apply(lambda x: x.strip())
location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

## Outlier Removal ##

df = df[~(df.total_sqft / df.bhk < 300)]
#print(df.price_per_sqft.describe())

def remove_outliers(df):
    df_out = pd.DataFrame()
    for i, subdf in df.groupby('location'):
        mean = np.mean(subdf.price_per_sqft)
        standard_deviation = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft >= (mean-standard_deviation)) & (subdf.price_per_sqft <= (mean+standard_deviation))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out

df1 = remove_outliers(df)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df2 = remove_bhk_outliers(df1)

## Visualization ## 

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df2, "Hebbal")
plt.show()

matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df2.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

df2 = df2[df2.bath<df2.bhk+1]

## Model Building ##

df2 = df2.drop(['size', 'price_per_sqft'], axis=1)
dummies = pd.get_dummies(df2['location'], drop_first=True)
for col in dummies.select_dtypes(include=['bool']).columns:
    dummies[col] = dummies[col].astype(int)
df2 = pd.concat([df2.drop('location', axis=1), dummies.drop('other', axis=1)], axis=1)

X = df2.drop(['price'], axis=1)
y = df2.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['squared_error','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

best_model = find_best_model_using_gridsearchcv(X_train, y_train)
print(best_model)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model_lr.predict([x])[0]

# print(predict_price('Indira Nagar', 1000, 3, 3))

## Pickle file ##
with open('real_estate_predict_price', 'wb') as f:
    pickle.dump(model_lr, f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))
