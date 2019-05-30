# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import preprocessing

def load_dataset():
    fifa_filepath = "data.csv"
    data = pd.read_csv(fifa_filepath)
    data.head()
    
    # Seleciona apenas algumas features de interesse
    df2 = data.loc[:, 'Crossing':'Release Clause']
    df1 = data[['Age', 'Overall', 'Value', 'Wage', 'Preferred Foot', 'Skill Moves', 'Position', 'Height', 'Weight']]
    df = pd.concat([df1, df2], axis=1)
    # Excluit todos os exemplos que possuem features ausentes
    df = df.dropna()
    

    # REaliza alguns procedimentos para a padronização de algumas features
    def value_to_int(df_value):
        try:
            value = float(df_value[1:-1])
            suffix = df_value[-1:]
    
            if suffix == 'M':
                value = value * 1000000
            elif suffix == 'K':
                value = value * 1000
        except ValueError:
            value = 0
        return value
    # Realiza alguns procedimentos para a padronização de algumas features
    df['Value_float'] = df['Value'].apply(value_to_int)
    df['Wage_float'] = df['Wage'].apply(value_to_int)
    df['Release_Clause_float'] = df['Release Clause'].apply(lambda m: value_to_int(m))
    
    def weight_to_int(df_weight):
        value = df_weight[:-3]
        return value
      
    df['Weight_int'] = df['Weight'].apply(weight_to_int)
    df['Weight_int'] = df['Weight_int'].apply(lambda x: int(x))
    
    def height_to_int(df_height):
        try:
            feet = int(df_height[0])
            dlm = df_height[-2]
            if dlm == "'":
                height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
            elif dlm != "'":
                height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
        except ValueError:
            height = 0
        return height
    
    df['Height_int'] = df['Height'].apply(height_to_int)
    
    df = df.drop(['Value', 'Wage', 'Release Clause', 'Weight', 'Height'], axis=1)
    
    # Label encoder na feature Preferred Foot
    le_foot = preprocessing.LabelEncoder()
    df["Preferred Foot"] = le_foot.fit_transform(df["Preferred Foot"].values)
    
    # Transforma o problema em um problema de 3 classes, separadas por setor do campo
    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
      df.loc[df.Position == i , 'Pos'] = 'Strikers' 
    
    for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:
      df.loc[df.Position == i , 'Pos'] = 'Midfielder' 
    
    for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:
      df.loc[df.Position == i , 'Pos'] = 'Defender' 
    
    return df