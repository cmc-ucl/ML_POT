import os
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

if 'htmls' in os.listdir('./'):
    shutil.rmtree('htmls')
    os.mkdir('htmls')
else:
    os.mkdir('htmls')


#############
# RMSE Al-F # 
#############
list_dir = [os.path.join(os.getcwd(), x) for x in os.listdir('./') if 'GAP_7' in x and os.path.isdir(x)]
list_dir = [x for x in list_dir if int(x.split('_')[-1]) != 0 ]
list_dir = sorted(list_dir, key=lambda x: (float(x.split('_')[-2]), int(x.split('_')[-1])))
df_AlF = pd.DataFrame(columns=['cutoff', 'sparse'])

MSE_AlF = []
MSE_FF = []
cutoff = []
sparse = []
for i in tqdm(list_dir, desc="Get RMSE for Al-F and F-F"):
    mse_file = [os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
    if len(mse_file) == 0:

        ALL = i.split('-')
        ALL = [x.split('_') for x in ALL]
        ALL = [x for sub in ALL for x in sub]

        C = ALL[-2]
        S = ALL[-1]

        mode = ALL[1:-5]
        step = ALL[-5]
        structs = ALL[-4:-2]

        mode = ' '.join(mode)
        structs = ' '.join(structs)

        shutil.rmtree(i)
        os.system(f"python /home/uccatka/auto/for_GAP/GULP_GAP.py '{mode}' {step} {structs} {C} {S} y y y")

        mse_file = [os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x]
        if len(mse_file) == 0:
            shutil.rmtree(i)
            continue
    else: pass
    cutoff.append(float(mse_file[0].split('_')[-2]))
    sparse.append(float(mse_file[0].split('_')[-1].split('/')[-2]))
    with open(mse_file[0], 'r') as f:
        lines = f.readlines()
        MSE_AlF.append(float(lines[0]))
        MSE_FF.append(float(lines[1]))
df_AlF['cutoff'] = cutoff
df_AlF['sparse'] = sparse
df_AlF['Al-F MSE'] = MSE_AlF
df_AlF.sort_values(by=['cutoff', 'sparse'], inplace=True)

fig = go.Figure(data =
    go.Contour(
        z=MSE_AlF,
        x=cutoff, # horizontal axis
        y=sparse, # vertical axis
        colorbar=dict(
            title='RMSE between ±0.5 Å from the equilibrium bond dist',
            titleside='right'),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.01)
            )
    )

fig.update_layout(
    title="RMSE between the Al-F GAP potential energy and the reference Al-F potential energy in the <br>\
range of ±0.5 Å from the equilibrium bond distances as a function of {n_sparse} and {cutoff} parameters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black")
        )

df_AlF.to_csv('htmls/mse_Al-F.csv', index=False)
fig.write_html(f'htmls/2b-MSE_Al-F.html')

############
# RMSE F-F #
############
df_FF = pd.DataFrame(columns=['cutoff', 'sparse'])
MSE = []
#cutoff = []
#sparse = []
#for i in tqdm(list_dir, desc= "Get RMSE for F-F"):
#    mse_file = [os.path.join(i, x) for x in os.listdir(i) if 'MSE' in x][0]
#    cutoff.append(float(mse_file.split('_')[-2]))
#    sparse.append(float(mse_file.split('_')[-1].split('/')[-2]))
#    with open(mse_file, 'r') as f:
#        line = f.readlines()[1]     # 2nd line is for mse between buck and GAP F-F
#        MSE.append(float(line))
df_FF['cutoff'] = cutoff
df_FF['sparse'] = sparse
df_FF['F-F MSE'] = MSE_FF
df_FF.sort_values(by=['cutoff', 'sparse'], inplace=True)

fig = go.Figure(data =
    go.Contour(
        z=MSE_FF,
        x=cutoff, # horizontal axis
        y=sparse, # vertical axis
        colorbar=dict(
            title='RMSE between ±0.5 Å from the equilibrium bond dist',
            titleside='right'),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.01)
            )
            )

fig.update_layout(
    title="RMSE between the F-F GAP potential energy and the reference potential energy in the <br>\
range of ±0.5 Å from the equilibrium bond distances as a function of {n_sparse} and {cutoff} parameters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
        )
    )

df_FF.to_csv('htmls/mse_F-F.csv', index=False)
fig.write_html(f'htmls/2b-MSE_F-F.html')


##################################
# 2b vs |E(GAP=1.58)-E(BM=1.58)| #
##################################
list_dirs = [x for x in os.listdir('./') if os.path.isdir(x) if 'htmls' not in x if 'top_structures' not in x if 'GAP_copy' not in x]
list_dirs = sorted(list_dirs, key=lambda x: (float(x.split('_')[-2]), int(x.split('_')[-1])))

df_temp_AlF = pd.DataFrame(columns=['r',  'Al-F(GAP)',  'Al-Al(GAP)',  'F-F(GAP)', 'Al-F(BM)'])
df_temp_FF = pd.DataFrame(columns=['r',  'Al-F(GAP)',  'Al-Al(GAP)',  'F-F(GAP)', 'Al-F(BM)'])
for i in tqdm(list_dirs, desc= "Get {GAP_pot_tabulated.csv} files for Al-F and F-F"):
    df_pot = pd.read_csv(os.path.join(i, 'FIT', 'GAP_pot_tabulated.csv'))
    df_temp2 = df_pot.loc[(df_pot['r'] >= 1.58) & (df_pot['r'] < 1.59)]
    df_temp_AlF = pd.concat([df_temp_AlF, df_temp2], axis=0, ignore_index=True)
    df_temp3 = df_pot.loc[(df_pot['r'] >= 2.73) & (df_pot['r'] < 2.74)]
    df_temp_FF = pd.concat([df_temp_FF, df_temp3], axis=0, ignore_index=True)

df_joint_AlF = pd.concat([df_AlF, df_temp_AlF], axis=1, join='inner')
df_joint_FF = pd.concat([df_FF, df_temp_FF], axis=1, join='inner')


fig_1 = go.Figure(
    data=
    go.Contour(
        z=abs(df_joint_AlF['Al-F(GAP)']-df_joint_AlF['Al-F(BM)']),
        x=df_joint_AlF['cutoff'],
        y=df_joint_AlF['sparse'],
        colorbar=dict(
            title='|E(GAP(Al-F)=1.58) - E(BM(Al-F)=1.58)| (Equilibrium bond dist Al-F = 1.57705 Å)',
            titleside='right'),
        colorscale='Hot',
        contours=dict(
            start=0,
            end=0.2,
            size=0.05)
            )
            )

fig_1.update_layout(
    title="|E(GAP(Al-F)=1.58) - E(BM(Al-F)=1.58)| as a function of {n_sparse} and {cutoff} paramters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black")
)

df_joint_AlF.to_csv('htmls/E_Al-F.csv', index=False)
fig_1.write_html(f'htmls/energy_Al-F.html')


#####################################
# 2b vs |E(GAP=1.58)-E(buck4=1.58)| #
#####################################
fig_2 = go.Figure(
    data=
    go.Contour(
    z=abs(df_joint_FF['F-F(GAP)']-df_joint_FF['F-F(buck4)']),
        x=df_joint_FF['cutoff'],
        y=df_joint_FF['sparse'],
        colorbar=dict(
            title='|E(GAP(F-F)=2.73) - E(buck4(F-F)=2.73)| (Equilibrium bond dist F-F = 2.73154 Å)',
            titleside='right'),
        colorscale='Hot',
        contours=dict(
            start=0.04,
            end=0.05,
            size=0.001)
)
)

fig_2.update_layout(
    title="|E(GAP(F-F)=2.73) - E(buck4(F-F)=2.73)| as a function of {n_sparse} and {cutoff} paramters",
    xaxis_title="cutoff distance",
    yaxis_title="n_sparse",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black")
)

df_joint_FF.to_csv('htmls/E_F-F.csv', index=False)
fig_2.write_html(f'htmls/energy_F-F.html')


##################################
# r vs cutoff  vs |E(GAP)-E(BM)| #
##################################
GAP_AlF = []
df_AlF.sort_values(by=['Al-F MSE'], inplace=True)
min_AlF_mse = df_AlF.iloc[0]
C_AlF = str(np.around(min_AlF_mse['cutoff'], 2))
S_AlF = str(int(min_AlF_mse['sparse']))
list_dirs_cut = [x for x in list_dirs if x.split('_')[-1] == '10'] #S_AlF]  #n_sparse==10
#min_AlF_dir = [x for x in list_dir if C in x if S in x][0]

for i in tqdm(list_dirs_cut, desc=""):
    path = os.path.join(i, 'FIT', 'GAP_pot_tabulated.csv')
    df_temp = pd.read_csv(path)
    #df_temp = df_temp.loc[(df_pot['r'] >= 1.57705-0.5) & (df_pot['r'] <= 1.57705+0.5)]
    gap = df_temp['Al-F(GAP)'].to_numpy()
    gulp = df_temp['Al-F(BM)'].to_numpy()

    SQRT = []
    for j in range(len(gap)):
        sqrt = np.abs(np.subtract(gap[j], gulp[j]))
        sqrt = np.log(sqrt)
        SQRT.append(sqrt)
    GAP_AlF.append(SQRT)

distance = df_temp['r'].to_numpy()
cutoff = df_joint_FF['cutoff'].to_numpy()
cutoff = np.unique(cutoff)
GAP_CUT_DIST = np.array(GAP_AlF).T

fig_1 = go.Figure(
    data=
    go.Contour(
        x=cutoff,
        y=distance,
        z=GAP_CUT_DIST,                    ######
        colorbar=dict(
            title='log(E(GAP(Al-F)-BM(Al-F))) at the near (±0.5 Å) equilibrium bond dist for Al-F',
            titleside='right'),
        colorscale='Hot',
        contours=dict(
            start=-10,
            end=2,
            size=0.2)
)
)

fig_1.update_layout(
    title="log(E(GAP(Al-F)-BM(Al-F))) as a function of interatomic distance (Al-F)<br> and cutoff distance paramters",
    xaxis_title="GAP cutoff parameter",
    yaxis_title="r (Al-F)",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black")
)

fig_1.write_html('htmls/r-cutoff-energy_Al-F.html')
del df_temp, GAP_CUT_DIST, gap, gulp

#####################################
# r vs cutoff  vs |E(GAP)-E(Bcuk4)| #
#####################################
GAP_FF = []
df_FF.sort_values(by=['F-F MSE'], inplace=True)
min_AlF_mse = df_AlF.iloc[0]
C_FF = str(np.around(min_AlF_mse['cutoff'], 2))
S_FF = str(int(min_AlF_mse['sparse']))
list_dirs_cut = [x for x in list_dirs if x.split('_')[-1] == '10'] #S_FF] #'10']

for i in list_dirs_cut:
    path = os.path.join(i, 'FIT', 'GAP_pot_tabulated.csv')
    df_temp = pd.read_csv(path)
    #df_temp = df_temp.loc[(df_pot['r'] >= 2.73154-0.5) & (df_pot['r'] <= 2.73154+0.5)]
    gap = df_temp['F-F(GAP)']
    gulp = df_temp['F-F(buck4)']

    gap = [str(x) for x in gap]
    gulp = [str(x) for x in gulp]
    SQRT = []
    for numj, j in enumerate(gap):
        sqrt = np.abs(np.subtract(float(gap[numj]), float(gulp[numj])))
        sqrt = np.log(sqrt)
        SQRT.append(sqrt)
    GAP_FF.append(SQRT)

distance = df_temp['r'].to_numpy()
GAP_CUT_DIST = np.array(GAP_FF).T

fig_1 = go.Figure(
    data=
    go.Contour(
        x=cutoff,
        y=distance,
        z=GAP_CUT_DIST,                    ######
        colorbar=dict(
            title='log(E(GAP(F-F)-BM(Al-F))) at the near equilibrium bond dist for Al-F',
            titleside='right'
        ),
        colorscale='Hot',
        contours=dict(
            start=-10,
            end=2,
            size=0.2,
        )
       )
)

fig_1.update_layout(
    title="log(E(GAP(F-F)-Buck4(F-F))) as a function of interatomic distance (Al-F) <br>and cutoff distance paramt",
    xaxis_title="GAP cutoff parameter",
    yaxis_title="r (F-F)",
    font=dict(
        family="Arial, monospace",
        size=18,
        color="Black"
    )
)

fig_1.write_html('htmls/r-cutoff-energy_F-F.html')








