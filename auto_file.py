import subprocess
import fileinput
import sys
from pathlib import Path
import os


 
nr_stiffness_ratio = ["5","6"]
nr_growth_ratio = ["1","1.5","2","2.5"]
#nr_growth_rate = ["4.7e-4","5e-4","6e-4"]
#nr_growth_exponent =["1.5","1.2","1","2"]
#nr_divsion_rate_RG = ["60","120","30","150"]
#nr_divsion_rate_ORG = ["0","10","20","30","60"]
#nr_migration_speed = ["7"]
#nr_mst = ["0.02","0.01","0.03","0.04"]
#nr_cortex_thickness = ["0.1","0.07","0.05"]

#for growth_exponent in nr_growth_exponent:
#    for growth_rate in nr_growth_rate:
#        for mst in nr_mst:
#            for migration_speed in nr_migration_speed:
#                for cortex_thickness in nr_cortex_thickness:
#for migration_speed in nr_migration_speed:
#    for divsion_rate_ORG in nr_divsion_rate_ORG:
#        for divsion_rate_RG in nr_divsion_rate_RG:
for growth_ratio in nr_growth_ratio:
    for stiffness_ratio in nr_stiffness_ratio:
        folder_name= "SR"+stiffness_ratio+"_GR"+growth_ratio
        subfolder = "Constant"
        parent_dir_folder = os.getcwd()
        path_folder = os.path.join(parent_dir_folder, subfolder, folder_name)
                    
        if os.path.exists(path_folder):
            print(f"The folder '{path_folder}' exists.")

        else:
            print ("===================================================================================================================")
            print ("\n Stiffness rate "+stiffness_ratio+" ,growth ratio "+growth_ratio+"\n")
            print ("===================================================================================================================")
            for line in fileinput.input("Parameters.prm", inplace=1):
                if "set The ratio of stiffness" in line:
                    line = "        set The ratio of stiffness                       = "+stiffness_ratio+" \n"
                if "set  Growth ratio"in line:
                    line = "        set  Growth ratio                                = "+growth_ratio+" \n"
        #                                        if "set Cortex thickness"in line:
        #                                            line = "        set Cortex thickness                             = "+cortex_thickness+" \n"
        #                                        if "set Mitotic somal translocation factor"in line:
        #                                            line = "        set Mitotic somal translocation factor           ="+mst+" \n"
        #                                        if "set Growth rate"in line:
        #                                            line = "        set Growth rate                                  ="+growth_rate+" \n"
        #                                        if "set Growth exponent"in line:
#        #                                            line = "        set Growth exponent                              ="+growth_exponent+" \n"
#                            if "set Cell dvision rate of RGCs"in line:
#                                line = "        set Cell dvision rate of RGCs                    ="+divsion_rate_RG+" \n"
#                            if "set Cell dvision rate of Outer RGCs"in line:
#                                line = "        set Cell dvision rate of Outer RGCs              ="+divsion_rate_ORG+" \n"
#                            if "set Cell migration speed"in line:
#                                line = "        set Cell migration speed                         ="+migration_speed+" \n"
                sys.stdout.write(line)

                                            
            a = subprocess.run(['./Brain_growth', 'Parameters.prm',  sys.argv[1]])
                                                
            makefolder = subprocess.Popen('mkdir    '+folder_name, shell=True)
            makefolder.wait()
            move = subprocess.Popen('mv  Output*          '+folder_name, shell=True)
            move.wait()
            copy = subprocess.Popen('cp  Parameters.prm     '+folder_name, shell=True)
            copy.wait()
            move2 = subprocess.Popen('mv  timeing.csv     '+folder_name, shell=True)
            move2.wait()
            movefolder = subprocess.Popen('mv '+folder_name+'  '+subfolder, shell=True)
            movefolder.wait()
            print ("COPY DONE! \n")

