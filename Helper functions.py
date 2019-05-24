import session_directory as sd
import os
import pickle
dir_use = sd.find_eraser_directory('Marble24','Shock','-1')
os.path.join(dir_use,'placefields_cml_manlims.pkl')
pf_file = os.path.join(dir_use, 'placefields_cm1_manlims.pkl')
with open(pf_file, 'rb') as file:
    PF = pickle.load(file)



