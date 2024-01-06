import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.style.use('classic')
from dataprep import eda as dpeda
import os
import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score #計算混淆矩陣及分數
from sklearn.model_selection import StratifiedKFold

### UCI 測試範例
# Taiwanese Bankruptcy Prediction
data_path = os.path.join('input', 'data.csv')
df_data = pd.read_csv(data_path)
df_data.rename(columns={'Bankrupt?':'label'}, inplace=True)

# 簡單eda
df_data.describe()
df_data.info()
df_data.shape
df_data.isna().any().any()
df_data['label'].value_counts()

# # 輸出eda html
# eda_report_path = os.path.join('output', 'eda_report')
# report = dpeda.create_report(df_data, title = 'eda_report')
# report.save(eda_report_path)

# # 繪分布圖存檔，確認資料在label = 0, 1時，feature分布有差異，順便看離群值概況
# df_tmp = df_data.sample(3000)
# for num, i in enumerate(df_tmp.columns[1:]):
#     sns.kdeplot(data=df_tmp, x=i, hue='label')
#     plt.title(i)
#     plt.xlabel(i)
#
#     i = i.replace('/', ' ')  # 會有特殊字元造成路徑錯誤
#     output_path = os.path.join('output', f'{num+1}.{i}.png')
#     plt.savefig(output_path)
#     plt.show()
#     # plt.pause(1)  # 等1秒


### 處理訓練、測試集
from sklearn.model_selection import train_test_split   #切割訓練集、測試集
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# stratify 讓測試、訓練集的Y的01樣本佔比相同
X_train, X_test, Y_train, Y_test = train_test_split(df_data.drop(columns='label'), df_data['label'],
                                                    test_size=0.25, stratify=df_data['label'], random_state=0)
print(X_train.shape, Y_train.shape)  #75%訓練集
print(X_test.shape, Y_test.shape)    #25%測試集

# 不平衡樣本處理
print(f"未處理前正負樣本")
print(Y_train.value_counts())
print(f"正樣本占比:{Y_train.value_counts()[1]/len(Y_train)}")      # 確認為不平衡樣本，Out: 0.03226437231130231

# SMOTE+TomekLinks
# Oversampling (SMOTE) : 概念是在少數樣本位置近的地方，人工合成一些樣本
# Undersampling (Tomek Link) : 找出邊界鑑別度不高的樣本，認為這些樣本屬雜訊應該剔除

X_train, Y_train = SMOTE(sampling_strategy=0.2, random_state=5).fit_resample(X_train, Y_train)
print(f"oversampling後的正負樣本分布")
print(Y_train.value_counts())
print(f"正樣本占比:{Y_train.value_counts()[1]/len(Y_train)}")

X_train, Y_train = TomekLinks().fit_resample(X_train, Y_train)
print(f"undersampling後的正負樣本分布")
print(Y_train.value_counts())
print(f"正樣本占比:{Y_train.value_counts()[1]/len(Y_train)}")

### 建立4個類別預測模型，Decision Tree, Random Forest, Gradient Boosting Tree, Support Vector Machine
# 1. Decision Tree, 決策樹
from sklearn.tree import DecisionTreeClassifier

# 找最佳參數
def objective(trial, X, y):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1),                  # 最小分裂需求樣本
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),                  # 葉片上最小樣本
        'class_weight': {0: 1, 1: 20},  # 正樣本權重增強
    }

    # Cross Validation
    metric_val = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in cv.split(X, y):
        # 拆訓練、驗證
        X_train_opt, X_val_opt = X.iloc[train_index], X.iloc[val_index]
        y_train_opt, y_val_opt = y.iloc[train_index], y.iloc[val_index]

        optuna_model = DecisionTreeClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt)

        # 預測結果，優化目標:設為f1_score
        metric = f1_score(y_val_opt, optuna_model.predict(X_val_opt))
        # metric = log_loss(y_val_opt, optuna_model.predict_proba(X_val_opt))
        metric_val.append(metric)

    # 回傳這組超參數的平均 log_loss
    return np.mean(metric_val)


# 創建 Optuna 學習器，開始優化
study = optuna.create_study(direction='maximize')  # 設置目標是最大化f1
# study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X = X_train, y = Y_train), n_trials=100, n_jobs = -1)

# 獲取最佳超參數
best_params = study.best_params
print("Best Hyperparameters of DecisionTreeClassifier:", best_params)

# 用最佳參數建模，偏重1的樣本
clf = DecisionTreeClassifier(**best_params)
clf.fit(X_train, Y_train)

# 建立記錄各個模型成效的df
df_result = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1'])
model1_result = {
    'model': 'DecisionTree',
    'accuracy': round(accuracy_score(Y_test, clf.predict(X_test)),4),
    'precision': round(precision_score(Y_test, clf.predict(X_test)),4),
    'recall': round(recall_score(Y_test, clf.predict(X_test)),4),
    'f1': round(f1_score(Y_test, clf.predict(X_test)),4)
}
df_result = df_result.append(model1_result, ignore_index=True)

print(f"==========DecisionTree 結果==========")
print(classification_report (Y_test, clf.predict(X_test)))
print(f"測試集正負樣本分布 :\n {Y_test.value_counts()}")


# 計算各個feature重要度
importance = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(clf.feature_importances_,2)})
importance = importance.sort_values('importance', ascending=False)
importance.head(10)

# 繪圖儲存
# sns.barplot(x = 'importance', y = 'feature', data = importance)
# plt.savefig('決策樹特徵值重要度.png',bbox_inches='tight',pad_inches=0.0,dpi=300)


### 2. Random Forest, 隨機森林
from sklearn.ensemble import RandomForestClassifier
def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),                    # 樹的樹量
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        'class_weight' : {0: 1, 1: 20},                                                # 正樣本權重增強
    }

    # Cross Validation
    metric_val = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in cv.split(X, y):
        # 拆訓練、驗證
        X_train_opt, X_val_opt = X.iloc[train_index], X.iloc[val_index]
        y_train_opt, y_val_opt = y.iloc[train_index], y.iloc[val_index]

        optuna_model = RandomForestClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt)

        # 預測結果，優化目標:設為f1_score
        metric = f1_score(y_val_opt, optuna_model.predict(X_val_opt))
        # metric = log_loss(y_val_opt, optuna_model.predict_proba(X_val_opt))
        metric_val.append(metric)

    # 回傳這組超參數的平均 log_loss
    return np.mean(metric_val)


# 創建 Optuna 學習器，開始優化
study = optuna.create_study(direction='maximize')  # 設置目標是最大化f1
# study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X = X_train, y = Y_train), n_trials=30, n_jobs = -1)

# 獲取最佳超參數
best_params = study.best_params
print("Best Hyperparameters of RandomForestClassifier:", best_params)

# 用最佳參數建模
rfc = RandomForestClassifier(**best_params)
rfc.fit(X_train, Y_train)

# 記錄各個模型成效的df
model1_result = {
    'model': 'RandomForest',
    'accuracy': round(accuracy_score(Y_test, rfc.predict(X_test)),4),
    'precision': round(precision_score(Y_test, rfc.predict(X_test)),4),
    'recall': round(recall_score(Y_test, rfc.predict(X_test)),4),
    'f1': round(f1_score(Y_test, rfc.predict(X_test)),4)
}
df_result = df_result.append(model1_result, ignore_index=True)


print(f"==========RandomForest 結果==========")
print(classification_report (Y_test, rfc.predict(X_test)))

#計算各個feature重要度
importance = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(rfc.feature_importances_,2)})
importance = importance.sort_values('importance', ascending=False)
importance.head(10)


### 3. XGBoost, Gradient Boosting Tree, 梯度提升樹
from xgboost import XGBClassifier
def objective(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        # 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),      # 控制每棵樹訓練時使用的特徵比例
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),           # 最小子葉節點的權重和，控制生長速度
        'lambda': trial.suggest_float('lambda', 1e-5, 100),  # L2 正則化項
        'alpha': trial.suggest_float('alpha', 1e-5, 100),  # L1 正則化項
    }

    # Cross Validation
    metric_val = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in cv.split(X, y):
        # 拆訓練、驗證
        X_train_opt, X_val_opt = X.iloc[train_index], X.iloc[val_index]
        y_train_opt, y_val_opt = y.iloc[train_index], y.iloc[val_index]

        optuna_model = XGBClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt)

        # 預測結果，優化目標:設為f1_score
        metric = f1_score(y_val_opt, optuna_model.predict(X_val_opt))
        # metric = log_loss(y_val_opt, optuna_model.predict_proba(X_val_opt))
        metric_val.append(metric)

    # 回傳這組超參數的平均 log_loss
    return np.mean(metric_val)


# 創建 Optuna 學習器，開始優化
study = optuna.create_study(direction='maximize')  # 設置目標是最大化f1
# study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X = X_train, y = Y_train), n_trials=30, n_jobs = -1)

# 獲取最佳超參數
best_params = study.best_params
print("Best Hyperparameters of XGBoost:", best_params)

# 用最佳參數建模
xgb = XGBClassifier(**best_params)
xgb.fit(X_train, Y_train)

# 記錄各個模型成效的df
model1_result = {
    'model': 'XGBoost',
    'accuracy': round(accuracy_score(Y_test, xgb.predict(X_test)),4),
    'precision': round(precision_score(Y_test, xgb.predict(X_test)),4),
    'recall': round(recall_score(Y_test, xgb.predict(X_test)),4),
    'f1': round(f1_score(Y_test, xgb.predict(X_test)),4)
}
df_result = df_result.append(model1_result, ignore_index=True)

print(f"==========XGBoost 結果==========")
print(classification_report (Y_test, xgb.predict(X_test)))

#計算各個feature重要度
importance = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(xgb.feature_importances_,2)})
importance = importance.sort_values('importance', ascending=False)
importance.head(10)

### 4. LGBM, Gradient Boosting Tree, 梯度提升樹
from lightgbm import LGBMClassifier
# warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")

def objective(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'num_leaves': trial.suggest_int('num_leaves', 80, 300, step=20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),   # 最小子葉節點的權重和，控制生長速度
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100),         # 正則化
    }

    # Cross Validation
    metric_val = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in cv.split(X, y):
        # 拆訓練、驗證
        X_train_opt, X_val_opt = X.iloc[train_index], X.iloc[val_index]
        y_train_opt, y_val_opt = y.iloc[train_index], y.iloc[val_index]

        optuna_model = LGBMClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt)

        # 預測結果，優化目標:設為f1_score
        metric = f1_score(y_val_opt, optuna_model.predict(X_val_opt))
        # metric = log_loss(y_val_opt, optuna_model.predict_proba(X_val_opt))
        metric_val.append(metric)

    # 回傳這組超參數的平均 log_loss
    return np.mean(metric_val)


# 創建 Optuna 學習器，開始優化
study = optuna.create_study(direction='maximize')  # 設置目標是最大化f1
# study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X = X_train, y = Y_train), n_trials=30, n_jobs = -1)

# 獲取最佳超參數
best_params = study.best_params
print("Best Hyperparameters of LGBM:", best_params)

# 用最佳參數建模
lgbm = LGBMClassifier(**best_params)
lgbm.fit(X_train, Y_train)

# 記錄各個模型成效的df
model1_result = {
    'model': 'LGBM',
    'accuracy': round(accuracy_score(Y_test, lgbm.predict(X_test)),4),
    'precision': round(precision_score(Y_test, lgbm.predict(X_test)),4),
    'recall': round(recall_score(Y_test, lgbm.predict(X_test)),4),
    'f1': round(f1_score(Y_test, lgbm.predict(X_test)),4)
}
df_result = df_result.append(model1_result, ignore_index=True)

print(f"==========LGBMoost 結果==========")
print(classification_report (Y_test, lgbm.predict(X_test)))

#計算各個feature重要度
importance = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(lgbm.feature_importances_,2)})
importance = importance.sort_values('importance', ascending=False)
importance.head(10)

### 5. Support Vector Machine，支持向量機
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def objective(trial, X, y):
    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),              # 對錯誤的懲罰力度
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e5),      # 支持向量的影響範圍
        'class_weight': {0: 1, 1: 20},                              # 正樣本權重增強
        'kernel': 'rbf',                                            # 怕無法收斂，限定用高斯核
        'max_iter': 100000,                                         # 限制最多迭代100000次
    }

    # Cross Validation
    metric_val = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in cv.split(X, y):
        # 拆訓練、驗證
        X_train_opt, X_val_opt = X.iloc[train_index], X.iloc[val_index]
        y_train_opt, y_val_opt = y.iloc[train_index], y.iloc[val_index]

        optuna_model = SVC(**param)
        optuna_model.fit(X_train_opt, y_train_opt)

        # 預測結果，優化目標:設為f1_score
        metric = f1_score(y_val_opt, optuna_model.predict(X_val_opt))
        # metric = log_loss(y_val_opt, optuna_model.predict_proba(X_val_opt))
        metric_val.append(metric)

    # 回傳這組超參數的平均 log_loss
    return np.mean(metric_val)


# 先做標準化
float_columns = X_train.select_dtypes(include=['float']).columns        # 抓出浮點數欄位
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[float_columns] = scaler.fit_transform(X_train_scaled[float_columns])
X_test_scaled[float_columns] = scaler.transform(X_test_scaled[float_columns])


# 創建 Optuna 學習器，開始優化
study = optuna.create_study(direction='maximize')  # 設置目標是最大化f1
# study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X = X_train_scaled, y = Y_train), n_trials=30, n_jobs = -1)

# 獲取最佳超參數
best_params = study.best_params
print("Best Hyperparameters of SVM:", best_params)

# 用最佳參數建模
svm = SVC(**best_params)
svm.fit(X_train_scaled, Y_train)

# 記錄各個模型成效的df
model1_result = {
    'model': 'SVM',
    'accuracy': round(accuracy_score(Y_test, svm.predict(X_test_scaled)),4),
    'precision': round(precision_score(Y_test, svm.predict(X_test_scaled)),4),
    'recall': round(recall_score(Y_test, svm.predict(X_test_scaled)),4),
    'f1': round(f1_score(Y_test, svm.predict(X_test_scaled)),4)
}
df_result = df_result.append(model1_result, ignore_index=True)

print(f"==========SVM 結果==========")
print(classification_report (Y_test, svm.predict(X_test_scaled)))


### 隨機亂猜正負樣本的基線比較，50%猜猜1，10%猜1

print(f"測試集正負樣本分布 :\n {Y_test.value_counts()}")
def guess_metric(pos_portion):
    pos_count_guess = int(len(Y_test)*pos_portion)     # 亂猜的正樣本數
    neg_count_guess = len(Y_test) - pos_count_guess    # 亂猜的負樣本數
    # 創建包含 neg_count_guess 個 0 和 pos_count_guess 個 1 的 array
    guess_array = np.concatenate((np.zeros(neg_count_guess), np.ones(pos_count_guess)))
    np.random.seed(0)
    np.random.shuffle(guess_array)
    print(f"隨機亂猜{pos_portion*100}%為正樣本的指標表現")
    print(classification_report(Y_test, guess_array))

    # 記錄各個模型成效的df
    model1_result = {
        'model': f'guess {pos_portion*100} % positive',
        'accuracy': round(accuracy_score(Y_test, guess_array), 4),
        'precision': round(precision_score(Y_test, guess_array), 4),
        'recall': round(recall_score(Y_test, guess_array), 4),
        'f1': round(f1_score(Y_test, guess_array), 4)
    }
    return model1_result


model1_result = guess_metric(0.5)
df_result = df_result.append(model1_result, ignore_index=True)

model1_result = guess_metric(0.1)
df_result = df_result.append(model1_result, ignore_index=True)

df_result.set_index('model', inplace=True)

print(df_result)

if __name__ == '__main__':
    pass


