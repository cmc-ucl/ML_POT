import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename="Data.csv"     #File you want to open
cm='coolwarm'           #Colour map you want to use

# Read data
data = pd.read_csv('Data.csv')

# Pull IPs 1 and 2 - this is just check that they are what I think they should be
A   = np.array(data['A'])   
rho = np.array(data['rho'])   

# Get values for the x and y axis
X = np.unique(data['A'])    
Y = np.unique(data['rho'])    

X_len = len(X)
Y_len = len(Y)

# Reshape the IP1 and 2 just so I can see that they look like they should
IP1_re = np.reshape(A, (X_len, Y_len))  
IP2_re = np.reshape(rho, (X_len, Y_len))

# Pull the data val1 and val2
val1 = np.array(data['sos_fit'])     
val2 = np.array(data['sos_relax'])

#Reshape the data to fit the grid


################
# Standard fit #
################
if pd.isna(data['sos_fit'][0]) == False:
    val1_re = np.reshape(val1, (X_len,Y_len))  
    Z1=val1_re.T
    
    fig1,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z1, levels=np.linspace(5000, 6000, 10), cmap=cm)
    cbar_1st = fig1.colorbar(cp)
    for l in cbar_1st.ax.yaxis.get_ticklabels():
        l.set_fontsize(20)
    
    fig1.set_size_inches(24, 12)
    ax.set_title('Standard fit option', fontsize=40)
    ax.set_xlabel('A', fontsize=35)
    ax.set_ylabel('ρ', fontsize=35)
    ax.tick_params(labelsize=20)
    plt.savefig('only_standard_fit_option.png')
    plt.show(fig1)
    plt.close(fig1)


#############
# Relax fit #
#############
if pd.isna(data['sos_relax'][0]) == False:
    val2_re = np.reshape(val2,(X_len,Y_len))
    Z2=val2_re.T
    
    fig2,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z2, levels=np.linspace(0, 300, 10), cmap=cm)
    cbar_2st = fig2.colorbar(cp)
    for l in cbar_2st.ax.yaxis.get_ticklabels():      # colour bar label size
        l.set_fontsize(20)
    
    fig2.set_size_inches(24, 12)
    ax.set_title('Relax fit option', fontsize=40)
    ax.set_xlabel('A', fontsize=35)
    ax.set_ylabel('ρ', fontsize=35)
    ax.tick_params(labelsize=20)           
    plt.savefig('only_relax_fit_option.png')
    plt.show(fig2)
    plt.close(fig2)



####################################
## Plot two plots at the same time #
####################################
#fig, ((ax1, ax2)) = plt.subplots(2,1)
#fig.set_size_inches(12, 6) 
#
## First set of data
#cplot1 = ax1.contourf(X, Y, Z1,50,cmap=cm)
#ax1.set_title('Standard fit option', fontsize=20)
#ax1.set_xlabel('A', fontsize=12.5)
#ax1.set_ylabel('ρ', fontsize=12.5)
#ax1.tick_params(labelsize=10)
#cbar1 = fig.colorbar(cplot1, ax=ax1)
#tick_font_size = 10
#cbar1.ax.tick_params(labelsize=tick_font_size)
#
## Second set of data
#cplot2 = ax2.contourf(X, Y, Z2,50,cmap=cm)
#ax2.set_title('Relax fit option', fontsize=20)
#ax2.set_xlabel('A', fontsize=12.5)
#ax2.set_ylabel('ρ', fontsize=12.5)
#ax2.tick_params(labelsize=10)
#cbar2 = fig.colorbar(cplot2, ax=ax2)
#tick_font_size = 10
#cbar2.ax.tick_params(labelsize=tick_font_size)
#
#fig.tight_layout()
#plt.savefig('Standard_fit_and_Relax_fit_options.png')
##plt.close()


