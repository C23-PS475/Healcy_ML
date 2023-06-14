import pandas as pd
data = pd.read_csv('arrrghh.csv')
json_data = data.to_json(orient='records')
with open('arrrghh.json', 'w') as file:
    file.write(json_data)

