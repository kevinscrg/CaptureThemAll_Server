import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import joblib
from scipy.stats import t

def prepare_cross_validation_folds(df, procent_train = 0.7, nr_folds = 5):
    X = df.drop(columns=['label']).reset_index(drop=True)
    y = df['label'].reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=42) 
    folds = []
    
    
    for fold_idx, (_, fold_indices) in enumerate(kf.split(X, y)):
        
        fold_df = df.iloc[fold_indices].reset_index(drop=True)
        
        fold_train, fold_val = train_test_split(
            fold_df,
            train_size=procent_train,
            stratify=fold_df['label'],
            shuffle=True,
            random_state=fold_idx  
        )
        
        val_normal, val_attack = fold_val[fold_val['label'] == 0], fold_val[fold_val['label'] == 1]
        n = min(len(val_normal), len(val_attack))
        
        fold_val = pd.concat([val_normal.sample(n=n, random_state=fold_idx), val_attack.sample(n=n, random_state=fold_idx)], ignore_index=True)
        fold_val = fold_val.sample(frac=1, random_state=fold_idx).reset_index(drop=True)
        
        
        x_train = fold_train.drop(columns=['label']).reset_index(drop=True)
        y_train = fold_train['label'].reset_index(drop=True)
        x_val = fold_val.drop(columns=['label']).reset_index(drop=True)
        y_val = fold_val['label'].reset_index(drop=True)
        
        folds.append((x_train, y_train, x_val, y_val))
    
    return folds

def randomForestTrain(folds):
    
    precisions_per_class = []
    recalls_per_class = []
    f1_scores_per_class = []
    conf_matrices = []
    accuracies = []
    rf_models = []
    for i, (x_train_fold, y_train_fold, x_val_fold, y_val_fold) in enumerate(folds):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(x_train_fold, y_train_fold)
        y_pred = rf_model.predict(x_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)
        rf_models.append(rf_model)
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    
    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)

    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    interval_incredere = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)
    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval_incredere': interval_incredere
    }

def decisionTreeTrain(folds):

    precisions_per_class = []
    recalls_per_class = []
    f1_scores_per_class = []
    conf_matrices = []
    accuracies = []
    dt_models = []
    for i, (x_train_fold, y_train_fold, x_val_fold, y_val_fold) in enumerate(folds):
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(x_train_fold, y_train_fold)
        y_pred = dt_model.predict(x_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)
        dt_models.append(dt_model)
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    
    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)
    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    interval_incredere = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)
    
    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval_incredere': interval_incredere
    }
    
def svmTrain(folds):
    precisions_per_class = []
    recalls_per_class = []
    f1_scores_per_class = []
    conf_matrices = []
    accuracies = []
    svm_models = []
    scalers = []
    
    for i, (x_train_fold, y_train_fold, x_val_fold, y_val_fold) in enumerate(folds):
        
        scaler = StandardScaler()
        x_train_fold = scaler.fit_transform(x_train_fold)
        x_val_fold = scaler.transform(x_val_fold)
        
        svm_model = SVC(random_state=42)
        svm_model.fit(x_train_fold, y_train_fold)
        y_pred = svm_model.predict(x_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)
        svm_models.append(svm_model)
        scalers.append(scaler)
    
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    
    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)

    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)
    
    interval_incredere = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)
    
    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval_incredere': interval_incredere
    }

    

def TrainModel(df, model, procent_train, model_name, nr_folds=5):
    folds = prepare_cross_validation_folds(df, procent_train, nr_folds)
    results = model(folds)
    if not retrain_and_save_final_model(df, model_name, procent_train):
        print("Warning: model was evaluated, but not saved correctly.")
    return results
   
   
def retrain_and_save_final_model(df, model_name, procent_train):
    try:
        df_rest = joblib.load(f'dfs/df_rest_{model_name}.pkl')
        df_train, df_test = train_test_split(df, train_size=procent_train, stratify=df['label'], shuffle=True, random_state=42)
        X = df_train.drop(columns=['label']).reset_index(drop=True)
        y = df_train['label'].reset_index(drop=True)

        if df_rest is not None:
            test_malign = df_test[df_test['label'] == 1]
            test_benign = df_test[df_test['label'] == 0]
            n_test = min(len(test_malign), len(test_benign))
            df_test = pd.concat([test_malign.sample(n=n_test, random_state=42),
                                    test_benign.sample(n=n_test, random_state=42)], ignore_index=True)
            df_rest = pd.concat([df_rest, df_test], ignore_index=True)
            joblib.dump(df_rest, f'dfs/df_rest_{model_name}.pkl', compress=3)
            

        if model_name == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, "models/rf_model.pkl")
        
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X, y)
            joblib.dump(model, "models/dt_model.pkl")
        
        elif model_name == "SVM":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = SVC(random_state=42)
            model.fit(X_scaled, y)
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(model, "models/svm_model.pkl")
        return True
    except Exception as e:
        print(f"Error during retraining and saving final model: {e}")
        return False
   
def getResultsRF(x_test, y_test):
    rf_model = joblib.load('models/rf_model.pkl')
    y_pred = rf_model.predict(x_test)
    
    return {"class_report" : classification_report(y_test, y_pred, output_dict=True), 
            "conf_matrix" : confusion_matrix(y_test, y_pred).tolist() }
    
def getResultsDT(x_test, y_test):
    dt_model = joblib.load('models/dt_model.pkl')
    y_pred = dt_model.predict(x_test)
    
    return {"class_report" : classification_report(y_test, y_pred, output_dict=True), 
            "conf_matrix" : confusion_matrix(y_test, y_pred).tolist() }
    
def getResultsSVM(x_test, y_test):
    svm_model = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    x_test = scaler.transform(x_test)
    y_pred = svm_model.predict(x_test)
    
    return {"class_report" : classification_report(y_test, y_pred, output_dict=True), 
            "conf_matrix" : confusion_matrix(y_test, y_pred).tolist() }
    
def getResultsModel(model, df_test):
    
    x_test = df_test.drop(columns=['label']).reset_index(drop=True)
    y_test = df_test['label'].reset_index(drop=True)
    
    return model(x_test, y_test)
    
    
def getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val):
    interval_incredere_acc = [
    max(0,round(np.mean(accuracies) - val, 4)),
    min(1,round(np.mean(accuracies) + val, 4))
    ]
    
    interval_incredere_f1_normal = [
    max(0, round(np.mean(f1_scores_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(f1_scores_per_class[:, 0]) + val, 4))
    ]
    
    interval_incredere_f1_attack = [
    max(0, round(np.mean(f1_scores_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(f1_scores_per_class[:, 1]) + val, 4))
    ]
    
    interval_incredere_precision_normal = [
    max(0, round(np.mean(precisions_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(precisions_per_class[:, 0]) + val, 4))
    ]
    
    interval_incredere_precision_attack = [
    max(0, round(np.mean(precisions_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(precisions_per_class[:, 1]) + val, 4))
    ]
    
    interval_incredere_recall_normal = [
    max(0, round(np.mean(recalls_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(recalls_per_class[:, 0]) + val, 4))
    ]
    
    interval_incredere_recall_attack = [
    max(0, round(np.mean(recalls_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(recalls_per_class[:, 1]) + val, 4))
    ]
    
    return{
        'accuracy': interval_incredere_acc,
        'f1_score_normal': interval_incredere_f1_normal,
        'f1_score_attack': interval_incredere_f1_attack,
        'precision_normal': interval_incredere_precision_normal,
        'precision_attack': interval_incredere_precision_attack,
        'recall_normal': interval_incredere_recall_normal,
        'recall_attack': interval_incredere_recall_attack
    }
    
    