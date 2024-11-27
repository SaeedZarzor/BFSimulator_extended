import subprocess
import fileinput
import sys

 
nr_stiffness_ratio = ["4","5","1","2","8"]
nr_growth_ratio = ["1","1.5","2","2.5","3"]


make = subprocess.Popen('cmake CMakeLists.txt', shell=True)
make.wait()
subprocess.Popen('make', shell=True)

for stiffness_ratio in nr_stiffness_ratio:
    for growth_ratio in nr_growth_ratio:
        print ("===================================================================================================================")
        print ("\n Stiffness rate "+stiffness_ratio+" ,growth ratio "+growth_ratio+"\n")
        print ("===================================================================================================================")
        for line in fileinput.input("Parameters.prm", inplace=1):
            if "        set The ratio of stiffness                       =" in line:
                line = "        set The ratio of stiffness                       = "+stiffness_ratio+" \n"
            if "        set  Growth ratio                                ="in line:
                line = "        set  Growth ratio                                = "+growth_ratio+" \n"
            sys.stdout.write(line)

        
        a = subprocess.run(['./Brain_growth', 'Parameters.prm',  sys.argv[1]])
            
        folder_name = "SR" + stiffness_ratio +"_GR" + growth_ratio
        makefolder = subprocess.Popen('mkdir    '+folder_name, shell=True)
        makefolder.wait()
        copy1 = subprocess.Popen('cp  Output*          '+folder_name, shell=True)
        copy1.wait()
        copy2 = subprocess.Popen('cp  Parameters.prm     '+folder_name, shell=True)
        copy2.wait()
        copy3 = subprocess.Popen('cp  timeing.csv     '+folder_name, shell=True)
        copy3.wait()
        remove = subprocess.Popen('rm  Output*  timeing.csv      ', shell=True)
        remove.wait()
        print ("COPY DONE! \n")

