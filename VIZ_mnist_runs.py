from __future__ import print_function, division 
import numpy as np 
import matplotlib.pyplot as plt 
import os, sys, time, pickle 

files = os.listdir("./") 
clean_files = [] 
for f in files:
    if "run" in f:
        if os.path.isfile(os.path.join("./",f,"params.txt")) and os.path.isfile(os.path.join("./",f,"history.pkl")) :
            clean_files.append(f) 

print("Found folders: ", clean_files) 

for folder in clean_files:
    with open(os.path.join("./",folder,"params.txt"), "rt") as f:
        parameters_str = f.read() 
    with open(os.path.join("./", folder, "history.pkl"), "rb") as f:
        history_dict = pickle.load(f) 
    
    assert "val_preds_acc" in history_dict.keys(), "val_preds_acc not available "
    plt.plot(np.arange(20), history_dict["val_preds_acc"], label=parameters_str)
plt.legend() 
plt.show() 

