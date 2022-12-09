import os
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

if 'htmls' in os.listdir('./'):
    shutil.rmtree('htmls')
    os.mkdir('htmls')
else:
    os.mkdir('htmls')


#############################
# n_sparse vs cutoff vs MSE #
#############################
list_dir = [os.path.join(os.getcwd(), x) for x in os.listdir('./') if 'GAP_' in x and os.path.isdir(x)]
list_dir = [x for x in list_dir if int(x.split('_')[-1]) != 0 ]
list_dir = sorted(list_dir, key=lambda x: (float(x.split('_')[-2]), int(x.split('_')[-1])))

df = pd.DataFrame(columns=['cutoff', 'sparse'])

MSE = []
cutoff = []
sparse = []
for i in list_dir:
    print(i)
    mse_file =[os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
    cutoff.append(float(mse_file[0].split('_')[-2]))
    sparse.append(float(mse_file[0].split('_')[-1].split('/')[-2]))
    with open(mse_file[0], 'r') as f:
        line = f.readlines()[0]
        MSE.append(float(line))

df['cutoff'] = cutoff
df['sparse'] = sparse
df['mse'] = MSE
df.sort_values(by=['cutoff', 'sparse'], inplace=True)
df.to_csv('for_sparse_cutoff_mse_Al-F.csv', index=False)

fig = go.Figure(data =
    go.Contour(
        z=MSE,
        x=cutoff, # horizontal axis
        y=sparse, # vertical axis
        colorbar=dict(
            title='MSE between ±0.5Å from the equilibrium bond dist',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.01
        )
    ))

fig.update_layout(
    title="MSE as a function of no. of sparse and cutoff distance parameters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
        )
    )

fig.write_html(f'htmls/n_sparse-cutoff-MSE_Al-F.html')

#########################
#########################

MSE = []
cutoff = []
sparse = []
for i in list_dir:
    print(i)
    mse_file =[os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
    cutoff.append(float(mse_file[0].split('_')[-2]))
    sparse.append(float(mse_file[0].split('_')[-1].split('/')[-2]))
    with open(mse_file[0], 'r') as f:
        line = f.readlines()[1]
        MSE.append(float(line))

df['cutoff'] = cutoff
df['sparse'] = sparse
df['mse'] = MSE
df.sort_values(by=['cutoff', 'sparse'], inplace=True)
df.to_csv('for_sparse_cutoff_mse_F-F.csv', index=False)

fig = go.Figure(data =
    go.Contour(
        z=MSE,
        x=cutoff, # horizontal axis
        y=sparse, # vertical axis
        colorbar=dict(
            title='MSE between ±0.5Å from the equilibrium bond dist',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.01
        )
    ))

fig.update_layout(
    title="MSE as a function of no. of sparse and cutoff distance parameters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
        )
    )

fig.write_html(f'htmls/n_sparse-cutoff-MSE_F-F.html')



##############################################
# n_sparse, cutoff, |E(GAP=1.58)-E(BM=1.58)| #
##############################################
df = df
df = df.sort_values(by=['cutoff', 'sparse'], ignore_index=True)

list_dirs = [x for x in os.listdir('./') if os.path.isdir(x)]
list_dirs.remove('htmls')
list_dirs = sorted(list_dirs, key=lambda x: (float(x.split('_')[-2]), int(x.split('_')[-1])))

df2 = pd.DataFrame(columns=['r',  'Al-F(GAP)',  'Al-Al(GAP)',  'F-F(GAP)', 'Al-F(BM)'])
for i in list_dirs:
    df_pot = pd.read_csv(os.path.join(i, 'GAP_pot_tabulated.csv'))
    df_temp = df_pot.loc[(df_pot['r'] >= 1.58) & (df_pot['r'] < 1.59)]
    df2 = pd.concat([df2, df_temp], axis=0, ignore_index=True)

df3 = pd.concat([df, df2], axis=1, join='inner')
df3.to_csv('for_sparse_cutoff_E.csv', index=False)

fig_1 = go.Figure(
    data=
    go.Contour(
        z=abs(df3['Al-F(GAP)']-df3['Al-F(BM)']),
        x=df3['cutoff'],
        y=df3['sparse'],
        colorbar=dict(
            title='|E(GAP=1.58) - E(BM=1.58)| (Equilibrium bond dist Al-F = 1.57705 Å)',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.05
        )
       ))

fig_1.update_layout(
    title="E(Al-F(GAP)) as a function of no. of sparse and cutoff distance paramters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
    )
)

fig_1.write_html(f'htmls/GAP_sparse-cutoff-energy_Al-F.html')

#########################
#########################

fig_2 = go.Figure(
    data=
    go.Contour(
    z=abs(df3['F-F(GAP)']-df3['F-F(buck4)']),
            x=df3['cutoff'],
        y=df3['sparse'],
        colorbar=dict(
            title='|E(GAP=1.58) - E(BM=1.58)| (Equilibrium bond dist Al-F = 1.57705 Å)',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=2,
            size=0.05
        )
       ))

fig_2.update_layout(
    title="E(Al-F(GAP)) as a function of no. of sparse and cutoff distance paramters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
    )
)

fig_2.write_html(f'htmls/GAP_sparse-cutoff-energy_F-F.html')


#######################################
# r, cutoff, |E(GAP)-E(BM)|           #
#######################################
r = []
GAP_AlF = []

Range = [round(x*0.2, 2) for x in range(7, 31)]

list_dirs_cut = [x for x in list_dirs if x.split('_')[-1] == '10']
for i in list_dirs_cut:
    path = os.path.join(i, 'GAP_pot_tabulated.csv')
    df_4 = pd.read_csv(path)
    df_4['r'] = df_4['r'].round(2)
    df_4 = df_4.loc[df_pot['r'].isin(Range)]     # df_4['r'] which is in the range
    gap = df_4['Al-F(GAP)']
    gulp = df_4['Al-F(BM)']

    gap = [str(x) for x in gap]
    gulp = [str(x) for x in gulp]
    SQRT = []
    for numj, j in enumerate(gap):
        sqrt = np.sqrt(np.square(np.subtract(float(gap[numj]), float(gulp[numj])).mean()))
        sqrt = np.log(sqrt)
        SQRT.append(sqrt)
    GAP_AlF.append(SQRT)

r.append(df_4['r'].tolist())
distance = np.array(r)[0]
cutoff = df3['cutoff'].to_numpy()
cutoff = np.unique(cutoff)
DISTANCE, CUTOFF = np.meshgrid(distance, cutoff)
GAP_CUT_DIST = np.array(GAP_AlF).T

fig_1 = go.Figure(
    data=
    go.Contour(
        x=cutoff,
        y=distance,
        z=GAP_CUT_DIST,                    ######
        colorbar=dict(
            title='E(GAP) at equilibrium bond dist for Al-F',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=-10,
            end=10,
            size=0.2,
        )
       )
)

fig_1.update_layout(
    title="E(Al-F(GAP)) as a function of r and cutoff distance paramters",
    xaxis_title="GAP cutoff parameter",
    yaxis_title="r (Al-F)",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
    )
)

fig_1.write_html('htmls/GAP_r-cutoff-energy.html')

#############################
# sparse vs cutoff (z=RMSE) #
#############################
list_dir = [os.path.join(os.getcwd(), x) for x in os.listdir('./')
            if 'GAP_' in x and os.path.isdir(x)]
list_dir = [x for x in list_dir if int(x.split('_')[-1]) != 0]

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

fig = go.Figure(data=
                go.Contour(
                    z=MSE,
                    x=cutoff, # horizontal axis
                    y=sparse, # vertical axis
                    colorbar=dict(
                        title='RMSE of ±0.5Å from the equilibrium bond distance',
                        titleside='right'
                    ),
                    colorscale='Hot',
                    contours=dict(
                        start=0.0,
                        end=0.3, # 0.4
                        size=0.01 # 0.01
                    )
                ))
fig.write_html('htmls/contour_Al-F.html')

#########################
#########################

df2 = pd.DataFrame(columns=['cutoff', 'sparse'])
MSE = []
cutoff = []
sparse = []
for i in list_dir:
    mse_file = [os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
    cutoff.append(float(mse_file[0].split('_')[-2]))
    sparse.append(float(mse_file[0].split('_')[-1].split('/')[-2]))

    with open(mse_file[0], 'r') as f:
        line = f.readlines()[1]
        MSE.append(float(line))

df['cutoff'] = cutoff
df['sparse'] = sparse
df['mse'] = MSE
df.to_csv('for_contour.csv', index=False)

print(df)

fig = go.Figure(data=
                go.Contour(
                    z=MSE,
                    x=cutoff, # horizontal axis
                    y=sparse, # vertical axis
                    colorbar=dict(
                        title='RMSE of ±0.5Å from the equilibrium bond distance',
                        titleside='right'
                    ),
                    colorscale='Hot',
                    contours=dict(
                        start=0.0,
                        end=0.3, # 0.4
                        size=0.01 # 0.01
                    )
                )
               )
fig.write_html('htmls/contour_F-F.html')
