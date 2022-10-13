import gulp
import os, sys, random
from colored import fg, bg, attr

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# ase API
from ase import Atoms 
# quippy API
from quippy.potential import Potential

print()
print(f'{fg(15)} {bg(5)} Visualisation {attr(0)}')
print()
print()

binwidth = 0.05
sig2 = 0.005

Arg = sys.argv
EIGVEC = Arg[1]
STEP = int(Arg[2])
RANK_from = int(Arg[3])
RANK_to = int(Arg[4])
Breath = Arg[5]
cutoff= float(Arg[6])
sparse = int(Arg[7])

EIGVEC = EIGVEC.split()
_EIGVEC = '-'.join(EIGVEC)

wd_name = f'GAP_{_EIGVEC}_{STEP}_{RANK_from}-{RANK_to}_{Breath}_{cutoff}_{sparse}'

GULP = gulp.GULP(STEP, EIGVEC, SP='set')

cwd = os.getcwd()
wd = [os.path.join(cwd, x) for x in os.listdir('./') if os.path.isdir(x) and wd_name in x]
try:
    full_wd = wd[0]
except IndexError:
    print("WARNING: Probably you didn't run {gap_1} and {gap_2}")
    sys.exit()

ext_fpath = os.path.join(full_wd, 'ext_movie.xyz')
FIT_dir_path = os.path.join(full_wd, 'FIT')
FIT_dir_path = os.path.join(full_wd, 'FIT')

Training_xyz_path = os.path.join(FIT_dir_path, 'Training_set.xyz')
Valid_xyz_path = os.path.join(FIT_dir_path, 'Valid_set.xyz')

#############################################################
# prepare dimer curve data using the Interatomic  potential #
#############################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(f"{fg(1)} Al-F GAP pairwise interaction {attr(0)}")
dimers = [Atoms("AlF", positions=[[0,0,0], [x, 0,0]]) for x in np.arange(0.0, 6.02, 0.02)] #int(cutoff)+1.02, 0.02)]
print(full_wd)
pot = Potential('IP GAP', param_filename='%s/FIT/GAP.xml' % (full_wd))
dimer_curve = []
for dim in dimers:
    dim.set_calculator(pot)
    dimer_curve.append(dim.get_potential_energy())
x_axis = np.array([dim.positions[1,0] for dim in dimers])
y_axis = np.array(dimer_curve)
data = np.c_[x_axis, y_axis]
output_list = data.tolist()
df = pd.DataFrame (output_list, columns = ['r', 'Al-F(GAP)'])
del dim, dimers, dimer_curve, x_axis, y_axis, data, output_list

# Al-F scaled to match with scaled F-F
dimers_AlF_scaled = [Atoms("AlF", positions=[[0,0,0], [round(x/np.sqrt(3), 3),0,0]]) for x in np.arange(0.0, 6.02, 0.02)]
dimer_curve_AlF_scaled = []
for dim_AlF_scaled in dimers_AlF_scaled:
    dim_AlF_scaled.set_calculator(pot)
    dimer_curve_AlF_scaled.append(dim_AlF_scaled.get_potential_energy())
x_axis_AlF_scaled = np.array([dim_AlF_scaled.positions[1,0] for dim_AlF_scaled in dimers_AlF_scaled])
y_axis_AlF_scaled = np.array(dimer_curve_AlF_scaled)
data_AlF_scaled = np.c_[x_axis_AlF_scaled, y_axis_AlF_scaled]
output_list_AlF_scaled = data_AlF_scaled.tolist()
df_AlF_scaled = pd.DataFrame (output_list_AlF_scaled, columns = ['r_scaled', 'Al-F(GAP)_scaled'])
del dim_AlF_scaled, dimers_AlF_scaled, dimer_curve_AlF_scaled, x_axis_AlF_scaled, y_axis_AlF_scaled, data_AlF_scaled, output_list_AlF_scaled 
print()

print(f"{fg(3)} Al-Al GAP  pairwise interaction {attr(0)}")
dimers = [Atoms("AlAl", positions=[[0,0,0], [x, 0,0]]) for x in np.arange(0.0, 6.02, 0.02)] 
dimer_curve = []
for dim in dimers:
    dim.set_calculator(pot)
    dimer_curve.append(dim.get_potential_energy())
x_axis = np.array([dim.positions[1,0] for dim in dimers])
y_axis = np.array(dimer_curve)
data = np.c_[x_axis, y_axis]
output_list = data.tolist()
dff = pd.DataFrame (output_list, columns = ['x2', 'Al-Al(GAP)'])
del dimers, dimer_curve, x_axis, y_axis, data, output_list
print()

print(f"{fg(2)} F-F GAP pairwise interaction {attr(0)}")
dimers = [Atoms("FF", positions=[[0,0,0], [x,0,0]]) for x in np.arange(0.0, 6.02, 0.02)]
dimer_curve = []
for dim in dimers:
    dim.set_calculator(pot)
    dimer_curve.append(dim.get_potential_energy())
x_axis = np.array([dim.positions[1,0] for dim in dimers])
y_axis = np.array(dimer_curve)
data = np.c_[x_axis, y_axis]
output_list = data.tolist()
dfff = pd.DataFrame (output_list, columns = ['x3', 'F-F(GAP)'])
del dimers, dimer_curve, y_axis, data, output_list
print()

df = df.join(dff)
df = df.join(dfff)
df = df.join(df_AlF_scaled)
df.drop(columns=['x2', 'x3'], inplace=True)
del dff, dfff, df_AlF_scaled 

print('\n\n\nPlotting the pairwise interaction energy as a function of interatomic distances...\n\n\n')

###################################################
# calculating Interatomic distance and binning it #
###################################################
with open(Training_xyz_path, 'r') as f:
    lines = f.readlines()
no_of_atoms = int(lines[0])
lines = [x for x in lines if len(x) > 10 and 'Properties' not in x][:-2]

coordinates = []
ID = []
for i in lines:
    line = [float(x) for x in i.split()[1:]][:3]
    coordinates += line
    ID.append( i.split(' ')[0])
del lines

From = []
To = []
with open(ext_fpath, 'r') as f:
    lines = f.readlines()
for numi, i in enumerate(lines):
    if len(i) <= 10:
        From.append(numi)
        To.append(numi)

To.append(len(lines))
To = To[1:]

block = {From[i]: To[i] for i in range(len(From))}
del From, To
keys_list = list(block.keys())  # shuffle() wants a list
random.shuffle(keys_list)       # randomize the order of the keys

nkeys_80 = int(1.0 * len(keys_list))  # how many keys does 80% equal
keys_80 = keys_list[:nkeys_80]
keys_20 = keys_list[nkeys_80:]
del nkeys_80

train_80 = {k: block[k] for k in keys_80}
valid_20 = {k: block[k] for k in keys_20}
del keys_80, keys_20

coordinates = np.asarray(coordinates)
coordinates = np.reshape(coordinates, (len(train_80.keys()), no_of_atoms, 3))
ID = np.asarray(ID)
ID = np.reshape(ID, (len(train_80.keys()), no_of_atoms))

all_het_dist = []
all_homo_dist = []

for numi, i in enumerate(coordinates):
    nbins, npairs_all, npairs_het, npairs_homo, all_dist, het_dist, homo_dist, opdata = GULP.RDF(numi+1, no_of_atoms, i, ID[numi], binwidth, sig2)
    all_het_dist += het_dist
    all_homo_dist += homo_dist

#################################
# Plotting Born-Mayer, GAP, RDF #
#################################
# E vs r
fig = go.FigureWidget()

# Generated Born-Mayer potential
def BM(x):
    return 3760*np.exp(-x/0.222)

def deriv_BM(x):
    return -1880000*np.exp(-500*x/111)/111

def buck4(x):           # 2.73154 Å F-F distance
    if x.all() < 2.0:
        return 1127.7*np.exp(-x/0.2753)
    elif 2.0 <= x.all() < 2.726:
        return -3.976*x**5 + 49.0486*x4 -241.8573*x**3+ 597.2668*x**2 -741.117*x + 371.2706
    elif 2.726 <= x.all() < 3.031:
        return -0.361*x**3 +  3.2362*x**2 -9.6271*x +9.4816
    elif x.all() >= 3.031:
        return -15.83/P3035**6


# Born-Mayer Al-F potential
BM_color = 'rgb(10, 120, 24)'
df['Al-F(BM)'] = BM(x_axis)

trace1 = fig.add_scatter(x=x_axis, y=BM(x_axis), mode='lines', name='Al-F Born-Mayer',\
        line = dict(shape='linear', color=BM_color, dash='dot')
        )
# Buckingham 4-region F-F potential 
df['F-F(buck4)'] = buck4(x_axis)
df.to_csv(f'{full_wd}/GAP_pot_tabulated.csv', index=False)

trace2 = fig.add_scatter(x=x_axis, y=buck4(x_axis), mode='lines', name='F-F Buck4',\
        line = dict(shape='linear', color='firebrick', dash='dot')
        )


#####################
### GAP potential ###
#####################
# Al-F
trace3 = fig.add_scatter(x=df['r'], y=df[df.columns[1]], mode='lines', name='Al-F GAP potential',\
        line = dict(shape='linear', color = BM_color))#, secondary_y=False,)

## Al-F (same results but the data points' x-axis is match with F-F scaled)
#trace_ = fig.add_scatter(x=df['r_scaled'], y=df['Al-F(GAP)_scaled'], mode='lines', name='Al-F GAP potential_2',\
#        line=dict(shape='linear', color=BM_color))

# F-F original
trace4 = fig.add_scatter(x=df['r'], y=df['F-F(GAP)'], mode='lines', name='F-F GAP potential',\
line = dict(shape = 'linear', color = 'firebrick')) # , secondary_y=False,)

# F-F scaled
scaling = df['r'].tolist()
scaling = [round(x/np.sqrt(3), 3) for x in scaling]
trace5 = fig.add_scatter(x=df['r_scaled'], y=df['F-F(GAP)'], mode='lines', name='F-F GAP potential_scaling', visible='legendonly')
df.drop(columns=['Al-F(BM)', 'Al-Al(GAP)'], inplace=True)

# Al-F + F-F
df['GAP_sum'] = df['F-F(GAP)'] + df['Al-F(GAP)_scaled']
df.to_csv('df_gap_sum.csv')
trace6 = fig.add_scatter(x=df['r_scaled'], y=df['GAP_sum'], mode='lines', name='sum(GAP)', visible='legendonly')


## MSE (± 0.5 from the equilibrium interatomic distnace)
from_point = list(df['r']).index([x for x in df['r'] if 1.57705-0.5 < x][0])
to_point = list(df['r']).index([x for x in df['r'] if x < 1.57705+0.5][-1])
MSE_catan = round(mean_squared_error(df['Al-F(GAP)'][from_point:to_point], BM(x_axis)[from_point:to_point], squared=False), 4)
from_point_a = list(df['r']).index([x for x in df['r'] if 2.73154-0.5 < x < 2.73154+0.5][0])
to_point_a = list(df['r']).index([x for x in df['r'] if x < 2.73154+0.5][-1])
MSE_an = round(mean_squared_error(df['F-F(GAP)'][from_point_a:to_point_a], buck4(x_axis)[from_point_a:to_point_a], squared=False), 4)
with open(f'{wd_name}/MSE.txt', 'w') as f:
    f.write(str(MSE_catan))
    f.write('\n')
    f.write(str(MSE_an))
print(f'RMSE(cat-an): {MSE_catan}')
print(f'RMSE(an-an): {MSE_an}')


# Histogram above the potential figure (uppger panel)
trace7 = fig.add_histogram(x=all_het_dist, xbins=dict(start=0, end=6, size=0.005), marker_color=BM_color,\
                            name='Number of hetero species interatomic distance', yaxis='y2')
trace8 = fig.add_histogram(x=all_homo_dist, xbins=dict(start=0, end=6, size=0.005), marker_color='firebrick',\
                            name='Number of homo species interatomic distance', yaxis='y2')

fig.layout = dict(xaxis=dict(domain=[0, 0.8], showgrid=False, zeroline=False, title='Interatomic distance / Å'),
                  yaxis=dict(domain=[0, 0.8], range=[-20, 50], showgrid=False, zeroline=True, title='Potential energy / eV'),
                  margin=dict(l=80, r=80, t=80, b=150),
                  width=1800,
                  height=800,
                  hovermode='closest',
                  bargap=0.8,
                  xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, ),
                  yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, title='Count'),
                  font=dict(size=20),
                  )

# Transparent vertical region where from the shortest to longest interatomic distance covered by the training structure 
fig.add_vrect(
    x0=min(all_het_dist) ,x1=max(all_het_dist) ,
    fillcolor=BM_color, opacity=0.4,
    layer="below", line_width=1,)

fig.add_vrect(
    x0=min(all_homo_dist) ,x1=max(all_homo_dist) ,
    fillcolor='firebrick', opacity=0.4,
    layer="below", line_width=1,) 

# Vertial-dash line to show equilibrium interatomic distances for Al-F, F-F
fig.add_vline(x=1.57705, line_width=2, line_dash="dash", line_color=BM_color)
fig.add_vline(x=2.73154, line_width=2, line_dash="dash", line_color="firebrick")
fig.add_annotation(
            #size=15,
            x=0.6,
            y=0.7,
            text=f"RMSE(Al-F) = {MSE_catan} <br> RMSE(F-F) = {MSE_an}",
            xanchor='left',
            xref="paper",
            yref="paper")

#fig.add_annotation(
#            #size=15,
#            x=0,
#            y=-0.25,
#            text=f"RMSE = {MSE_catan} (at nearest distance > 1.57 Å \
#between V(GAP[Al-F]) and V(BM[Al-F])<br> RMSE = {MSE_an} \
#(at nearest distance > 2.7 Å between V(GAP[F-F]) and V(buck4[F-F])",
#            xanchor='left',
#            xref="paper",
#            yref="paper")
            


fig.write_html(f'./{wd_name}/plot.html')

print(f'\n\n\n The plot (./{wd_name}/plot.html) is saved -- Good luck!\n\n\n')


