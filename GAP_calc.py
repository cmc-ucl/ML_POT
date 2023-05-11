import os

pot = 'GAP.xml'
#pot = 'GAP_AlF.xml'
os.system(
    "/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T \
atoms_filename=./FIT/Training_set.xyz param_filename=./FIT/%s \
| grep AT | sed 's/AT//'  > ./FIT/quip_train.xyz" % pot)

