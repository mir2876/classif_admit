import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
admission= pd.read_csv("D:/Bootcamp ML & AI/misi_mira/admission_data.csv")
admission.head()
admission["admit_status"].value_counts()
admission.info()
admission.describe()
admission.head()
# check duplicated data overall
admission.duplicated().sum()
admission[admission.duplicated(keep=False)]
#Ada 5 data yang duplicated
#drop data duplicated
admission = admission.drop_duplicates()
# check missing values in each column
admission.isna().sum()
# percentage of missing values in each column
admission.isna().mean()
# drop all rows with missing values
admission = admission.dropna()
admission.hist(figsize=(15, 10))
plt.show()
#semua sudah berdistribusi normal
#Cek Outlier
# draw boxplot for each numeric column
plt.figure(figsize=(12,6))

# plotting
features = ['gre_score','toefl_score','motiv_letter_strength','recommendation_strength','gpa','research_exp']
for i in range(0, len(features)):
    plt.subplot(1, len(features), i+1)
    sns.boxplot(y=admission[features[i]], color='red')
    plt.tight_layout()
# drop rows that have outliers of recommendation_strength
# Using IQR method
Q1 = admission['recommendation_strength'].quantile(0.25)
Q3 = admission['recommendation_strength'].quantile(0.75)
IQR = Q3 - Q1

admission = admission[~((admission['recommendation_strength'] < (Q1 - 1.5 * IQR)) | (admission['recommendation_strength'] > (Q3 + 1.5 * IQR)))]
# value counts of categorical columns in admission
features = ['research_exp','admit_status','univ_tier']

for feature in features:
    print("***"*10)
    print(f'Value Counts of {feature}')
    print(admission[feature].value_counts())
    print('\n')
#Feature encoding
# label encode research_exp
research_exp_map = {
    'yes': 1,
    'no': 0
}

admission['research_exp'] = admission['research_exp'].map(research_exp_map)
# label encode admit_status
admit_status_map = {
    'yes': 1,
    'no': 0
}
admission['admit_status'] = admission['admit_status'].map(admit_status_map)
# label encode univ_tier
univ_tier_map = {
    'high': 1,
    'low': 0
}
admission['univ_tier'] = admission['univ_tier'].map(univ_tier_map)
# min-max scaling all column
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for col in admission.columns:
    admission[col] = scaler.fit_transform(admission[[col]])
admission.describe()
#save data to csv
admission.to_csv('D:/Bootcamp ML & AI/misi_mira/data_clean.csv', index=False)
# split train test
from sklearn.model_selection import train_test_split

feature = admission.drop(columns='admit_status')
target = admission[['admit_status']]
#Memastikan target berbentuk 1D array
target=target.values.ravel()
feature_admit_train, feature_admit_test, target_admit_train, target_admit_test = train_test_split(feature, target, test_size=0.20, random_state=42)
#Multicolinearity handling
# calculate vif score for each column in feature_admit_train
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

X = add_constant(feature_admit_train)

vif_add = (pd.DataFrame(
            [vif(X.values, i) for i in range(len(X.columns))]
            ,index=X.columns)
            .reset_index())

vif_add.columns = ['feature','vif_score']
vif_add = vif_add.loc[vif_add.feature!='const']
vif_add
# heatmap correlation
#admit_train = pd.concat([feature_admit_train, target_admit_train], axis=1)
admit_train = pd.concat([feature_admit_train, pd.DataFrame(target_admit_train, columns=['admit_status'])], axis=1)
corr = admit_train.corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()
# initiate logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=42)

# train the model
logreg.fit(feature_admit_train, target_admit_train)
# retrieve the coefficients
# show as a nice dataframe

data = feature_admit_train
model = logreg

coef_df = pd.DataFrame({
    'feature':['intercept'] + data.columns.tolist(),
    'coefficient':[model.intercept_[0]] + list(model.coef_[0])
})

coef_df
# classification report on training data
from sklearn.metrics import classification_report

target_predict_train = logreg.predict(feature_admit_train)

print(classification_report(target_admit_train,target_predict_train))

# classification report on test data
from sklearn.metrics import classification_report

target_predict_test = logreg.predict(feature_admit_test)

print(classification_report(target_admit_test,target_predict_test))
#ROC AUC
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# Hitung nilai ROC AUC
auc_score_rl_tr = roc_auc_score(target_admit_train, target_predict_train)
auc_score_rl_ts = roc_auc_score(target_admit_test, target_predict_test)
print("ROC AUC RL train:", auc_score_rl_tr)
print("ROC AUC RL test:", auc_score_rl_ts)
# plotting confusion matrix on test data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(7,5))

cm = confusion_matrix(target_admit_test, target_predict_test, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=logreg.classes_)
disp.plot(ax=ax)
plt.show()

#random forest
# train random forest model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 200, random_state=42)
rf.fit(feature_admit_train, target_admit_train)
# classification report on training data
from sklearn.metrics import classification_report

trf_predict_train = rf.predict(feature_admit_train)

print(classification_report(target_admit_train,trf_predict_train))
# classification report on test data
from sklearn.metrics import classification_report

trf_predict_test = rf.predict(feature_admit_test)

print(classification_report(target_admit_test,trf_predict_test))
# plotting confusion matrix on test data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(7,5))

cm = confusion_matrix(target_admit_test, trf_predict_test,
                      labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf.classes_)
disp.plot(ax=ax)
plt.show()

#Gradient Boosted Trees
# train a gradient boosted tree model on smote training data
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=42)

gbc.fit(feature_admit_train, target_admit_train)
# classification report on training data
from sklearn.metrics import classification_report

tgb_predict_train = gbc.predict(feature_admit_train)

print(classification_report(target_admit_train,tgb_predict_train))
# classification report on test data
from sklearn.metrics import classification_report

tgb_predict_test = gbc.predict(feature_admit_test)

print(classification_report(target_admit_test,tgb_predict_test))

# plotting confusion matrix on test data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(7,5))

cm = confusion_matrix(target_admit_test, tgb_predict_test, labels=gbc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=gbc.classes_)
disp.plot(ax=ax)
plt.show()

#model terpilih adalah model logreg, sehingga yang disimpan cukup model ini saja
#Simpan hasil train
import pickle
with open("D:/Bootcamp ML & AI/misi_mira/hasil_rl.pkl", "wb") as file:
    pickle.dump(model, file)