import pandas as pd
import numpy as np
import os
from colored import fg, bg, attr

# Helper functions
def A_sort(x):
    return (float(x.split('/')[-1].split('_')[0]))

def rho_sort(x):
    return (float(x.split('/')[-1].split('_')[1].split('.')[1]))

def every_n(i):
    dir_no = int(i.split('/')[-2])
    if dir_no % 5 == 0:
        return dir_no
        

if 'Data_relax.csv' in os.listdir('./'):
    os.remove('Data_relax.csv')
elif 'Data_standard.csv' in os.listdir('./'):
    os.remove('Data_standard.csv')
else:
    pass

root = os.getcwd() + '/storage'
output = [x for x in os.listdir(f'{root}')]
output = sorted(output, key=int)

useless = ['sh', 'error', 'output', 'in1', 'in2']
Final_fit = []
for i in output:
    final = [x for x in os.listdir(f'{root}/{i}') if x.split('.')[1] not in useless] 
    final = [x for x in final if x.split('.')[3] not in useless]
     
    for j in final:
        final = f'{root}/{i}/{j}'
        Final_fit.append(final)


Final_fit = [x.replace('.fit', '') for x in Final_fit]
Final_fit = [x.replace('.relax', '') for x in Final_fit]
output_fit = sorted(Final_fit, key = lambda x: (A_sort(x), rho_sort(x))) 
 
print(f'{fg(1)} {bg(14)} Collecting data... {attr(0)}')
df1 = pd.DataFrame(columns={'A_f', 'rho_f', 'sos_fit',})
df2 = pd.DataFrame(columns={'A_r', 'rho_r', 'sos_relax'})

for numi, i in enumerate(output_fit):
    A = float(i.split('/')[-1].split('_')[0])
    rho = float(i.split('/')[-1].split('_')[1]) #.split('.')[1])
   
    if os.path.isfile(f'{i}.fit'): #in os.listdir(i):
        with open(f'{i}.fit', 'r', errors='replace') as f:
            lines = f.readlines()
    
            for j in lines:
                if 'Cycle:      0  Sum sqs:' in j:
                    sos = str((j.split())[4])
                    if numi % 500 ==0:
                        print(f'{i}.fit {sos}')
                    if '**' in sos:                                     # Replacing 'nan' to large int
                        sos = 9999999999999
                        df1 = df1.append({'A_f': A, 'rho_f': rho, 'sos_fit': sos}, ignore_index=True)
                    else:
                        df1 = df1.append({'A_f': A, 'rho_f': rho, 'sos_fit': sos}, ignore_index=True)
   

    if os.path.isfile(f'{i}.relax'):  
        with open(f'{i}.relax', 'r', errors='replace') as f:
            lines = f.readlines()
            
            for j in lines:
                if 'Cycle:      0  Sum sqs:' in j:
                    sos = str((j.split())[4])
                    if numi % 500 ==0:
                        print(f'{i}.relax {sos}')
                    if '**' in sos:
                        sos = 9999999999999
                        df2 = df2.append({'A_r': A, 'rho_r': rho, 'sos_relax': sos}, ignore_index=True)
                    else:
                        df2 = df2.append({'A_r': A, 'rho_r': rho, 'sos_relax': sos}, ignore_index=True)


df1 = df1[['A_f','rho_f','sos_fit']]
df2 = df2[['A_r', 'rho_r', 'sos_relax']]

df1 = df1.sort_values(["A_f", "rho_f"])
df2 = df2.sort_values(["A_r", "rho_r"])

df1.to_csv('Data_standard.csv', index=False)
df2.to_csv('Data_relax.csv', index=False)
print(f'{fg(1)} {bg(14)} Done {attr(0)} \n')

os.system('python /home/uccatka/auto/IP/IP_visparam.py') 




