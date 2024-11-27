import subprocess
import fileinput
import sys

 
nr_c_k = ["0.33334"]
nr_betta = ["0","0.01","0.0335","0.05","0.1"]


make = subprocess.Popen('cmake CMakeLists.txt', shell=True)
make.wait()
subprocess.Popen('make', shell=True)

for c_k in nr_c_k:
    for betta in nr_betta:
        print ("===================================================================================================================")
        print ("\n Betta = "+betta+" , c_k = "+c_k+"\n")
        print ("===================================================================================================================")
        for line in fileinput.input("Parameters.prm", inplace=1):
            if "        set Output file name                              =" in line:
                line = "        set Output file name                              = growth \n"
            if "        set Stabilization constant                       =" in line:
                line = "        set Stabilization constant                       = "+betta+" \n"
            if "        set c_k factor                                   ="in line:
                line = "        set c_k factor                                   ="+c_k+" \n"
            sys.stdout.write(line)

            
        a = subprocess.Popen('make run', shell=True)
        a.wait()
            
        folder_name = "Betta"+betta
        makefolder = subprocess.Popen('mkdir    120_30/'+folder_name, shell=True)
        makefolder.wait()
        copy1 = subprocess.Popen('cp  Output*          120_30/'+folder_name, shell=True)
        copy1.wait()
        copy2 = subprocess.Popen('cp  Parameters.prm     120_30/'+folder_name, shell=True)
        copy2.wait()
        copy3 = subprocess.Popen('cp  timeing.csv     120_30/'+folder_name, shell=True)
        copy3.wait()
        remove = subprocess.Popen('rm  Output*  timeing.csv      ', shell=True)
        remove.wait()
        print ("COPY DONE! \n")

