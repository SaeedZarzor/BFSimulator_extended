import subprocess
import fileinput
import sys

 
nr_Cell_dvision_Inner_zone = ["30","60","120"]
nr_Cell_dvision_Outer_subzone = ["10","20","30"]


make = subprocess.Popen('cmake CMakeLists.txt', shell=True)
make.wait()
subprocess.Popen('make', shell=True)

for Cell_dvision_Outer_subzone in nr_Cell_dvision_Outer_subzone:
    for Cell_dvision_Inner_zone in nr_Cell_dvision_Inner_zone:  
        print ("===================================================================================================================")
        print ("\n RG rate "+Cell_dvision_Inner_zone+" ,ORG rate "+Cell_dvision_Outer_subzone+"\n")
        print ("===================================================================================================================")
        for line in fileinput.input("Parameters.prm", inplace=1):
            if "    set Output file name                              =" in line:
                line = "    set Output file name                              = growth \n"
            if "        set Cell dvision rate of RGCs                    =" in line:
                line = "        set Cell dvision rate of RGCs                    = "+Cell_dvision_Inner_zone+" \n"
            if "        set Cell dvision rate of Outer RGCs              ="in line:
                line = "        set Cell dvision rate of Outer RGCs              = "+Cell_dvision_Outer_subzone+" \n"
            sys.stdout.write(line)

        
        a = subprocess.Popen('make run', shell=True)
        a.wait()
            
        folder_name = "RG" + Cell_dvision_Inner_zone +"_ORG" + Cell_dvision_Outer_subzone
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

