from flask import Flask, request #type: ignore[import]
import pandas as pd
import joblib
from flask_cors import CORS #type: ignore[import]

import Models
import Preprocess

app = Flask(__name__)
CORS(app, supports_credentials=True)



def load_data():
    dfs = []
    for i in range(1,5):
        path = './data/UNSW-NB_complet/UNSW-NB15_{}.csv'  
        dfs.append(pd.read_csv(path.format(i), header = None))
    df = pd.concat(dfs).reset_index(drop=True)

    df_col = pd.read_csv('./data/UNSW-NB_complet/NUSW-NB15_features.csv', encoding='ISO-8859-1')
    df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())

    df.columns = df_col['Name']

    cols_to_drop_total = [
        'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime',
        'stcpb', 'dtcpb', 'attack_cat',
        'service', 'trans_depth', 'res_bdy_len',
        'is_ftp_login', 'ct_flw_http_mthd', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]


    df.drop(columns=cols_to_drop_total, inplace=True)

    df = df.drop_duplicates().reset_index(drop=True)

    print("done")
    return df
    
df = None

if df is None:
    df = load_data()


@app.route("/train_model", methods=["POST"])
def train_model():
    global df
    df_rest = None
    
    
    data = request.get_json()

    nr_of_instances = data.get("nr_of_instances")
    procent_train = data.get("procent_train")
    model_name = data.get("model_name")
    procent_attack = data.get("procent_attack")
    folds = data.get("folds")
    print(len(df[df['label'] == 1]))
    print(nr_of_instances* procent_attack)
    df_sample_attack = df[df['label'] == 1].sample(n=int(nr_of_instances* procent_attack), random_state=42)
    df_sample_normal = df[df['label'] == 0].sample(n=int(nr_of_instances* (1 - procent_attack)), random_state=42)
    
    df_sample = pd.concat([df_sample_attack, df_sample_normal])
    df_rest = df.drop(df_sample.index)
    
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    df_rest = df_rest.reset_index(drop=True)
    

    df_sample, df_rest = Preprocess.preprocess_data(df_sample, df_rest)
    joblib.dump(df_rest, f'dfs/df_rest_{model_name}.pkl', compress=3)
    
    model = None
    
    
    match model_name:
        case "RandomForest":
            model = Models.randomForestTrain
        case "DecisionTree":
            model = Models.decisionTreeTrain
        case "SVM":
            model = Models.svmTrain
        case _:
           return"Invalid model name", 400  
    
    return Models.TrainModel(df_sample, model, procent_train, model_name, folds), 200

@app.route("/get_results", methods=["POST"])
def get_results():
    
    data = request.get_json()
    model_name = data.get("model_name")
    
    df_rest = joblib.load(f'dfs/df_rest_{model_name}.pkl')
    
    model = None
    
    match model_name:
        case "RandomForest":
            model = Models.getResultsRF
        case "DecisionTree":
            model = Models.getResultsDT
        case "SVM":
            model = Models.getResultsSVM
        case _:
            return "Invalid model name", 400
    response = Models.getResultsModel(model, df_rest)
        
    return response, 200
    
    
if __name__ == "__main__":
    app.run(debug=True)