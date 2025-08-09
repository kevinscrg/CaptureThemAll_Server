from category_encoders import TargetEncoder
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd



def preprocess_data(df, df_test):
    transform_cols = ['proto', 
                  'state'   
                  ]

    encoder = TargetEncoder(cols=transform_cols, handle_unknown='value')

    df_encoded = encoder.fit_transform(df[transform_cols], df['label'])
    df_rest = df.drop(columns=transform_cols).reset_index(drop=True)
    df = pd.concat([df_rest, df_encoded], axis=1)
    
    df_test_encoded = encoder.transform(df_test[transform_cols])
    df_test = pd.concat([df_test.drop(columns=transform_cols).reset_index(drop=True), df_test_encoded], axis=1)
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    if 'label' in to_drop:
        to_drop.remove('label')
    df = df.drop(columns=to_drop)

    
    X = df.drop(columns=['label'])
    y = df['label']

    giwrf = RandomForestClassifier(
        n_estimators=100,          
        class_weight='balanced',   
        random_state=42,
        n_jobs=-1               
    )


    giwrf.fit(X, y)

    feature_importances = pd.Series(giwrf.feature_importances_, index=X.columns)


    threshold = 0.02
    selected_features_gini = feature_importances[feature_importances > threshold].index.tolist()

    if 'label' not in selected_features_gini:
        selected_features_gini.append('label')

    df = df[selected_features_gini]
    df_test = df_test[selected_features_gini]
    
    n_test = min(len(df_test[df_test['label'] == 0]), len(df_test[df_test['label'] == 1]))
    df_test = pd.concat([df_test[df_test['label'] == 0].sample(n=n_test, random_state=42), 
                         df_test[df_test['label'] == 1].sample(n=n_test, random_state=42)], ignore_index=True)
    
    
    joblib.dump(encoder, 'models/Tencoder.pkl')
    joblib.dump(selected_features_gini, 'models/selected_features.pkl')

    return df, df_test