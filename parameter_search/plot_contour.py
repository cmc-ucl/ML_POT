import os, sys
import plotly.express as px
import pandas as pd

list_dir = [os.path.join(os.getcwd(), x) for x in os.listdir('./') if 'GAP_' in x and os.path.isdir(x)]


list_dir = [x for x in list_dir if int(x.split('_')[-1]) != 0 ]

df = pd.DataFrame(columns=['cutoff', 'sparse'])

MSE = []
cutoff = []
sparse = []
for i in list_dir:
    mse_file =[os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
    cutoff.append(float(mse_file[0].split('_')[-2]))
    sparse.append(float(mse_file[0].split('_')[-1].split('/')[-2]))
    with open(mse_file[0], 'r') as f:
        line = f.readlines()[0]
        MSE.append(float(line))

df['cutoff'] = cutoff
df['sparse'] = sparse
df['mse'] = MSE
df.to_csv('for_contour.csv', index=False)

print(df)

import plotly.graph_objects as go

fig = go.Figure(data=
                go.Contour(
                    z=MSE,
                    x=cutoff, # horizontal axis
                    y=sparse, # vertical axis
                    colorbar=dict(
                        title='RMSE of ±0.5 ang from the equilibrium bond distance',
                        titleside='right'
                    ),
                    colorscale='Hot',
                    contours=dict(
                        start=0.1,
                        end=0.3,          # 0.4
                        size=0.01         # 0.01
                    )
                ))
fig.write_html(f'contour.html')

