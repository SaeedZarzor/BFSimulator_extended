#!/opt/homebrew/bin/Python3.10
# change the directory above #
######################################

import subprocess
import fileinput
import sys
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import  messagebox
import customtkinter
from PIL import Image
from PIL.Image import Resampling
import time
import psutil
import webbrowser
import numpy as np


global counter_1
global counter_2
global counter_3
global counter_4
counter_1 = 0
counter_2 = 0
counter_3 = 0
counter_4 = 0
#================================================== the main windows =================================================
Image.MAX_IMAGE_PIXELS = 1000000000
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("Brain model parameters")
root.geometry("1430x730")

#=============================== define the variables ================================================================
OSVZ_varying_options = ["Constant","Linear-gradient","Quadratic-gradient","Random1","Random2"]
def_vz_raduis = tk.StringVar(root, "")
def_svz_raduis = tk.StringVar(root, "")
def_cr_thickness = tk.StringVar(root, "")
def_intial_raduis = tk.StringVar(root, "")
def_MST_factor = tk.StringVar(root, "")
def_intial_division = tk.StringVar(root, "")
def_migration_threshold = tk.StringVar(root, "")
def_HV_exp = tk.StringVar(root, "")
def_shear_modulus = tk.StringVar(root, "")
def_stiffness_ratio = tk.StringVar(root, "")
def_poisson_ratio = tk.StringVar(root, "")
def_max_density = tk.StringVar(root, "")
def_stiffness_case = tk.StringVar(root, "")
def_case = tk.StringVar(root, "")
def_refinement = tk.StringVar(root, "")
def_degree = tk.StringVar(root, "")
def_stability_con = tk.StringVar(root, "")
def_c_k = tk.StringVar(root, "")
def_total_time = tk.StringVar(root, "")
def_delt_t = tk.StringVar(root, "")
def_nonlinear_it = tk.StringVar(root, "")
def_tol_u = tk.StringVar(root, "")
def_tol_c = tk.StringVar(root, "")
def_update_u = tk.StringVar(root, "")
def_solver_type = tk.StringVar(root, "")
def_linear_it = tk.StringVar(root, "")
def_k_growth = tk.StringVar(root, "")
def_growth_exp = tk.StringVar(root, "")
def_growth_ratio = tk.StringVar(root, "")
def_ORG_variation_case =  tk.StringVar(root, "   ")
def_RG_RG_n_ph1 = tk.StringVar(root, "")
def_RG_RG_n_ph2 = tk.StringVar(root, "")
def_RG_RG_n_ph3 = tk.StringVar(root, "")
def_RG_RG_n_ph4 = tk.StringVar(root, "")
def_RG_RG_n_ph5 = tk.StringVar(root, "")
def_IP_RG_n_ph1 = tk.StringVar(root, "")
def_IP_RG_n_ph2 = tk.StringVar(root, "")
def_IP_RG_n_ph3 = tk.StringVar(root, "")
def_IP_RG_n_ph4 = tk.StringVar(root, "")
def_IP_RG_n_ph5 = tk.StringVar(root, "")
def_OR_RG_n_ph1 = tk.StringVar(root, "")
def_OR_RG_n_ph2 = tk.StringVar(root, "")
def_OR_RG_n_ph3 = tk.StringVar(root, "")
def_OR_RG_n_ph4 = tk.StringVar(root, "")
def_OR_RG_n_ph5 = tk.StringVar(root, "")
def_NU_RG_n_ph1 = tk.StringVar(root, "")
def_NU_RG_n_ph2 = tk.StringVar(root, "")
def_NU_RG_n_ph3 = tk.StringVar(root, "")
def_NU_RG_n_ph4 = tk.StringVar(root, "")
def_NU_RG_n_ph5 = tk.StringVar(root, "")
def_IP_OR_n_ph1 = tk.StringVar(root, "")
def_IP_OR_n_ph2 = tk.StringVar(root, "")
def_IP_OR_n_ph3 = tk.StringVar(root, "")
def_IP_OR_n_ph4 = tk.StringVar(root, "")
def_IP_OR_n_ph5 = tk.StringVar(root, "")
def_OR_OR_n_ph1 = tk.StringVar(root, "")
def_OR_OR_n_ph2 = tk.StringVar(root, "")
def_OR_OR_n_ph3 = tk.StringVar(root, "")
def_OR_OR_n_ph4 = tk.StringVar(root, "")
def_OR_OR_n_ph5 = tk.StringVar(root, "")
def_NU_OR_n_ph1 = tk.StringVar(root, "")
def_NU_OR_n_ph2 = tk.StringVar(root, "")
def_NU_OR_n_ph3 = tk.StringVar(root, "")
def_NU_OR_n_ph4 = tk.StringVar(root, "")
def_NU_OR_n_ph5 = tk.StringVar(root, "")
def_IP_IP_n_ph1 = tk.StringVar(root, "")
def_IP_IP_n_ph2 = tk.StringVar(root, "")
def_IP_IP_n_ph3 = tk.StringVar(root, "")
def_IP_IP_n_ph4 = tk.StringVar(root, "")
def_IP_IP_n_ph5 = tk.StringVar(root, "")
def_NU_IP_n_ph1 = tk.StringVar(root, "")
def_NU_IP_n_ph2 = tk.StringVar(root, "")
def_NU_IP_n_ph3 = tk.StringVar(root, "")
def_NU_IP_n_ph4 = tk.StringVar(root, "")
def_NU_IP_n_ph5 = tk.StringVar(root, "")
def_IP_migration = tk.StringVar(root, "")
def_OR_migration = tk.StringVar(root, "")
def_NU_migration = tk.StringVar(root, "")
def_RG_diffusivity = tk.StringVar(root, "")
def_OR_diffusivity = tk.StringVar(root, "")
def_IP_diffusivity = tk.StringVar(root, "")
def_NU_diffusivity = tk.StringVar(root, "")
def_first_phase  = tk.StringVar(root, "")
def_second_phase  = tk.StringVar(root, "")
def_third_phase  = tk.StringVar(root, "")
def_fourth_phase  = tk.StringVar(root, "")


Error_1 = BooleanVar(root, value=False)
Error_2 = BooleanVar(root, value=False)
Error_3 = BooleanVar(root, value=False)
Error_4 = BooleanVar(root, value=False)
Error_5 = BooleanVar(root, value=False)
Error_6 = BooleanVar(root, value=False)
Error_7 = BooleanVar(root, value=False)
Error_8 = BooleanVar(root, value=False)
Error_9 = BooleanVar(root, value=False)
Error_10 = BooleanVar(root, value=False)
Error_11 = BooleanVar(root, value=False)
Error_12 = BooleanVar(root, value=False)
Error_13 = BooleanVar(root, value=False)
Error_14 = BooleanVar(root, value=False)

#================================= Open images ===========================================================

light_backround = Image.open('Images/Untitled-3.png')
dark_backround = Image.open('Images/Untitled-3-dark.png')
light_gemetry_vz=Image.open('Images/gemotry_vz_light.png')
dark_gemetry_vz=Image.open('Images/gemotry_vz_dark.png')
light_gemetry_isvz=Image.open('Images/gemotry_isvz_light.png')
dark_gemetry_isvz=Image.open('Images/gemotry_isvz_dark.png')
light_gemetry_tc=Image.open('Images/gemotry_tc_light.png')
dark_gemetry_tc=Image.open('Images/gemotry_tc_dark.png')
light_gemetry_R=Image.open('Images/gemotry_R_light.png')
dark_gemetry_R=Image.open('Images/gemotry_R_dark.png')
light_gemetry_mst=Image.open('Images/gemotry_mst_light.png')
dark_gemetry_mst=Image.open('Images/gemotry_mst_dark.png')
strain_image_mu_light = Image.open('Images/strain_Energy_mu_light.png')
strain_image_mu_dark = Image.open('Images/strain_Energy_mu_dark.png')
strain_image_nu_light = Image.open('Images/strain_Energy_nu_light.png')
strain_image_nu_dark = Image.open('Images/strain_Energy_nu_dark.png')
strain_image_ratio_light = Image.open('Images/strain_Energy_ratio_light.png')
strain_image_ratio_dark = Image.open('Images/strain_Energy_ratio_dark.png')
growth_image_ratio_light = Image.open('Images/growth_eqautions_ratio_light.png')
growth_image_ratio_dark = Image.open('Images/growth_eqautions_ratio_dark.png')
growth_image_ks_light = Image.open('Images/growth_eqautions_ks_light.png')
growth_image_ks_dark = Image.open('Images/growth_eqautions_ks_dark.png')
growth_image_exp_light = Image.open('Images/growth_eqautions_exp_light.png')
growth_image_exp_dark = Image.open('Images/growth_eqautions_exp_dark.png')
diffusion_image_gamma_light = Image.open('Images/adv_dif_eq_gamma_light.png')
diffusion_image_gamma_dark = Image.open('Images/adv_dif_eq_gamma_dark.png')
diffusion_image_c0_light = Image.open('Images/adv_dif_eq_c0_light.png')
diffusion_image_c0_dark = Image.open('Images/adv_dif_eq_c0_dark.png')
diffusion_image_v_light = Image.open('Images/adv_dif_eq_v_light.png')
diffusion_image_v_dark = Image.open('Images/adv_dif_eq_v_dark.png')
diffusion_image_d_light = Image.open('Images/adv_dif_eq_d_light.png')
diffusion_image_d_dark = Image.open('Images/adv_dif_eq_d_dark.png')
diffusion_image_Gvz_light = Image.open('Images/adv_dif_eq_G_vz_light.png')
diffusion_image_Gvz_dark = Image.open('Images/adv_dif_eq_G_vz_dark.png')
diffusion_image_Gosvz_light = Image.open('Images/adv_dif_eq_G_osvz_light.png')
diffusion_image_Gosvz_dark = Image.open('Images/adv_dif_eq_G_osvz_dark.png')
non_image_light = Image.open('Images/non.png')
non_image_light_2 = Image.open('Images/non_2.png')
two_d_light = Image.open('Images/2D_light.png')
two_d_dark = Image.open('Images/2D_dark.png')
three_d_light = Image.open('Images/3D_light.png')
three_d_dark = Image.open('Images/3D_dark.png')
ref_2d_light =  Image.open('Images/ref_2d_light.png')
ref_2d_dark =  Image.open('Images/ref_2d_dark.png')
varying_light = Image.open('Images/varying.png')
varying_dark = Image.open('Images/varying_dark.png')
varying_cmax_light = Image.open('Images/varying_cmax_light.png')
varying_cmax_dark = Image.open('Images/varying_cmax_dark.png')
logo =  Image.open('Images/Logo_BRAINIACS.png')
author =  Image.open('Images/my_photo.png')
OSVZ_constant = Image.open('Images/OSVZ_Constant.png')
OSVZ_Linear_gradient = Image.open('Images/OSVZ_Linear_gradient.png')
OSVZ_Quadratic_gradient = Image.open('Images/OSVZ_Quadratic_gradient.png')
OSVZ_Random1 = Image.open('Images/OSVZ_Random1.png')
OSVZ_Random2 = Image.open('Images/OSVZ_Random2.png')
#OSVZ_Random3 = Image.open('Images/OSVZ_Random3.png')
OSVZ_Linear_gradient_curve = Image.open('Images/Linear_gradient_curve.png')
OSVZ_Linear_gradient_curve_dark = Image.open('Images/Linear_gradient_curve_dark.png')
OSVZ_Quadratic_gradient_curve = Image.open('Images/Quadratic_gradient_curve.png')
OSVZ_Quadratic_gradient_curve_dark = Image.open('Images/Quadratic_gradient_curve_dark.png')
OSVZ_Random1_curve_dark = Image.open('Images/OSVZ_Random1_curve_dark.png')
OSVZ_Random1_curve = Image.open('Images/OSVZ_Random1_curve.png')
OSVZ_Random2_curve_dark = Image.open('Images/OSVZ_Random2_curve_dark.png')
OSVZ_Random2_curve = Image.open('Images/OSVZ_Random2_curve.png')




width, height = light_backround.size;
new_height  = 200
new_width = new_height * width / height

logo_width, logo_height = logo.size;
new_logo_height  = 140
new_logo_width = new_logo_height * logo_width / logo_height

author_width, author_height = author.size;
new_author_height  = 200
new_author_width = new_author_height * author_width / author_height

#=========================================== the Frames =========================================================

frame_backround = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=new_height+10,
                               corner_radius=10)
                               
frame_backround.place(x=10, y=10, anchor='nw')

frame_Gemotry = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=210,
                               corner_radius=10)
                               
frame_Gemotry.place(x=new_width+30, y=10, anchor='nw')

frame_diffusion = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=330,
                               corner_radius=10)
                               
frame_diffusion.place(x=10, y=new_height+30, anchor='nw')

frame_stiffness = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=210,
                               corner_radius=10)

frame_stiffness.place(x=2*new_width+50, y=10)


frame_mesh = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=280,
                               corner_radius=10)
frame_mesh.place(x=new_width+30, y=new_height+30, anchor='nw')

frame_solver = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=280,
                               corner_radius=10)
                               
frame_solver.place(x =2*new_width+50, y= 210+20 , anchor='nw')

fram_growth = customtkinter.CTkFrame(master=root,
                               width=new_width+10,
                               height=150,
                               corner_radius=10)
                               
fram_growth.place(x =10, y= new_height+330+40 , anchor='nw')

photo_info_fram = customtkinter.CTkFrame(master=root,
                               width=new_width+10+300,
                               height=200,
                               corner_radius=10)
                               
photo_info_fram.place(x=new_width+30, y=new_height+280+40, anchor='nw')


# ================================== instert images =========================================================

backround_image = customtkinter.CTkImage(light_image=light_backround, dark_image=dark_backround, size=(new_width, new_height))
label_img1 = customtkinter.CTkLabel(frame_backround, image = backround_image, text="")
label_img1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

gemotry_image_vz = customtkinter.CTkImage(light_image=light_gemetry_vz, dark_image=dark_gemetry_vz, size=dark_gemetry_vz.size)
gemotry_image_isvz = customtkinter.CTkImage(light_image=light_gemetry_isvz, dark_image=dark_gemetry_isvz, size=dark_gemetry_isvz.size)
gemotry_image_tc = customtkinter.CTkImage(light_image=light_gemetry_tc, dark_image=dark_gemetry_tc, size=dark_gemetry_tc.size)
gemotry_image_R = customtkinter.CTkImage(light_image=light_gemetry_R, dark_image=dark_gemetry_R, size=dark_gemetry_R.size)
gemotry_image_mst = customtkinter.CTkImage(light_image=light_gemetry_mst, dark_image=dark_gemetry_mst, size=dark_gemetry_mst.size)
strain_image_ratio = customtkinter.CTkImage(light_image=strain_image_ratio_light, dark_image=strain_image_ratio_dark, size=strain_image_ratio_dark.size)
strain_image_mu = customtkinter.CTkImage(light_image=strain_image_mu_light, dark_image=strain_image_mu_dark, size=strain_image_mu_dark.size)
strain_image_nu = customtkinter.CTkImage(light_image=strain_image_nu_light, dark_image=strain_image_nu_dark, size=strain_image_nu_dark.size)
diffusion_image_gamma = customtkinter.CTkImage(light_image=diffusion_image_gamma_light, dark_image=diffusion_image_gamma_dark, size=diffusion_image_gamma_dark.size)
diffusion_image_c0 = customtkinter.CTkImage(light_image=diffusion_image_c0_light, dark_image=diffusion_image_c0_dark, size=diffusion_image_c0_dark.size)
diffusion_image_v = customtkinter.CTkImage(light_image=diffusion_image_v_light, dark_image=diffusion_image_v_dark, size=diffusion_image_v_dark.size)
diffusion_image_d = customtkinter.CTkImage(light_image=diffusion_image_d_light, dark_image=diffusion_image_d_dark, size=diffusion_image_d_dark.size)
diffusion_image_Gvz = customtkinter.CTkImage(light_image=diffusion_image_Gvz_light, dark_image=diffusion_image_Gvz_dark, size=diffusion_image_Gvz_dark.size)
diffusion_image_Gosvz = customtkinter.CTkImage(light_image=diffusion_image_Gosvz_light, dark_image=diffusion_image_Gosvz_dark, size=diffusion_image_Gosvz_dark.size)
growth_image_ks = customtkinter.CTkImage(light_image=growth_image_ks_light, dark_image=growth_image_ks_dark, size=growth_image_ks_light.size)
growth_image_ratio = customtkinter.CTkImage(light_image=growth_image_ratio_light, dark_image=growth_image_ratio_dark, size=growth_image_ratio_light.size)
growth_image_exp = customtkinter.CTkImage(light_image=growth_image_exp_light, dark_image=growth_image_exp_dark, size=growth_image_exp_light.size)
non_image = customtkinter.CTkImage(light_image=non_image_light, dark_image=non_image_light, size=non_image_light.size)
non_image_2 = customtkinter.CTkImage(light_image=non_image_light_2, dark_image=non_image_light_2, size=non_image_light_2.size)
two_d = customtkinter.CTkImage(light_image=two_d_light, dark_image=two_d_dark, size=two_d_light.size)
three_d = customtkinter.CTkImage(light_image=three_d_light, dark_image=three_d_dark, size=three_d_dark.size)
ref_2d = customtkinter.CTkImage(light_image=ref_2d_light, dark_image=ref_2d_dark, size=ref_2d_dark.size)
varying_image = customtkinter.CTkImage(light_image=varying_light, dark_image=varying_dark, size=varying_dark.size)
varying_image_cmax = customtkinter.CTkImage(light_image=varying_cmax_light, dark_image=varying_cmax_dark, size=varying_cmax_dark.size)
logo_image = customtkinter.CTkImage(light_image=logo, dark_image=logo, size=(new_logo_width, new_logo_height))
author_image = customtkinter.CTkImage(light_image=author, dark_image=author, size=(new_author_width, new_author_height))
OSVZ_constant_image = customtkinter.CTkImage(light_image=OSVZ_constant, dark_image=OSVZ_constant, size=OSVZ_constant.size)
OSVZ_Linear_gradient_image = customtkinter.CTkImage(light_image=OSVZ_Linear_gradient, dark_image=OSVZ_Linear_gradient, size=OSVZ_Linear_gradient.size)
OSVZ_Quadratic_gradient_image = customtkinter.CTkImage(light_image=OSVZ_Quadratic_gradient, dark_image=OSVZ_Quadratic_gradient, size=OSVZ_Quadratic_gradient.size)
OSVZ_Random1_image = customtkinter.CTkImage(light_image=OSVZ_Random1, dark_image=OSVZ_Random1, size=OSVZ_Random1.size)
OSVZ_Random2_image = customtkinter.CTkImage(light_image=OSVZ_Random2, dark_image=OSVZ_Random2, size=OSVZ_Random2.size)
#OSVZ_Random3_image = customtkinter.CTkImage(light_image=OSVZ_Random3, dark_image=OSVZ_Random3, size=OSVZ_Random2.size)
OSVZ_Linear_gradient_curve_image = customtkinter.CTkImage(light_image=OSVZ_Linear_gradient_curve, dark_image=OSVZ_Linear_gradient_curve_dark, size=OSVZ_Linear_gradient_curve.size)
OSVZ_Quadratic_gradient_curve_image = customtkinter.CTkImage(light_image=OSVZ_Quadratic_gradient_curve, dark_image=OSVZ_Quadratic_gradient_curve_dark, size=OSVZ_Quadratic_gradient_curve.size)
OSVZ_Random1_curve_image = customtkinter.CTkImage(light_image=OSVZ_Random1_curve, dark_image=OSVZ_Random1_curve_dark, size=OSVZ_Random1_curve.size)
OSVZ_Random2_curve_image = customtkinter.CTkImage(light_image=OSVZ_Random2_curve, dark_image=OSVZ_Random2_curve_dark, size=OSVZ_Random2_curve.size)




label_img2 = customtkinter.CTkLabel(photo_info_fram, text="")
label_img3 = customtkinter.CTkLabel(photo_info_fram, text="")


#================================================ update parameters function =============================================
def update_parameters():
    if  not vz_raduis.get() or not svz_raduis.get() or not cr_thickness.get() or not intial_raduis.get() or not MST_factor.get() or not intial_division.get() or not migration_threshold.get() or not HV_exp.get() or not def_stiffness_case.get() or not poisson_ratio.get() or not shear_modulus.get() or not stiffness_ratio.get() or not max_density.get() or not refinement.get() or not degree.get() or not total_time.get() or not delt_t.get() or not stability_con.get() or not k_growth.get() or not growth_ratio.get() or not growth_exp.get() or not c_k.get() or (def_ORG_variation_case.get() == "   "):
        messagebox.showerror("","One or more field is empty!")
        return
  
    if  Error_1.get() or Error_2.get() or Error_3.get() or Error_4.get() or Error_5.get() or Error_6.get() or Error_7.get() or Error_8.get() or Error_9.get() or Error_10.get() or Error_11.get() or Error_12.get() or Error_13.get() or Error_14.get():
        messagebox.showerror("","One or more enrty values not correct!")
        return
 
    for line in fileinput.input("Parameters.prm", inplace=1):
    
        if "set Ventricular zone raduis" in line:
            line = "        set Ventricular zone raduis                       = "+vz_raduis.get()+" \n"
            
        if "set Subventricular zone raduis"in line:
            line = "        set Subventricular zone raduis                    = "+svz_raduis.get()+" \n"

        if "set Cortex thickness" in line:
            line = "        set Cortex thickness                              = "+cr_thickness.get()+" \n"

        if "set Initial radius" in line:
            line = "        set Initial radius                                = "+intial_raduis.get()+" \n"
            
        if "set Mitotic somal translocation factor" in line:
            line = "        set Mitotic somal translocation factor            = "+MST_factor.get()+" \n"

        if "set Cell dvision intial value" in line:
            line = "        set Cell dvision intial value                    = "+intial_division.get()+" \n"

        if "set IP cell migration speed" in line:
            line = "    set IP cell migration speed                          = "+def_IP_migration.get()+" \n"
            
        if "set ORG cell migration speed" in line:
            line = "    set ORG cell migration speed                         = "+def_OR_migration.get()+" \n"
            
        if "set NU cell migration speed" in line:
            line = "    set NU cell migration speed                          = "+def_NU_migration.get()+" \n"

        if "set RG diffusivity" in line:
            line ="    set RG diffusivity                                    = "+def_RG_diffusivity.get()+" \n"
            
        if "set ORG diffusivity" in line:
            line ="    set ORG diffusivity                                   = "+def_OR_diffusivity.get()+" \n"
            
        if "set IP diffusivity" in line:
            line ="    set IP diffusivity                                   = "+def_IP_diffusivity.get()+" \n"
            
        if "set NU diffusivity" in line:
            line ="    set NU diffusivity                                   = "+def_NU_diffusivity.get()+" \n"
            
        if "set Fourth phase" in line:
            line ="    set Fourth phase                                      = "+def_fourth_phase.get()+" \n"
            
        if "set Third phase" in line:
            line ="    set Third phase                                      = "+def_third_phase.get()+" \n"
            
        if "set Second phase" in line:
            line ="    set Second phase                                      = "+def_second_phase.get()+" \n"
            
        if "set First phase" in line:
            line ="    set First phase                                      = "+def_first_phase.get()+" \n"
            
        if "set Cell migration threshold" in line:
            line = "        set Cell migration threshold                     = "+migration_threshold.get()+" \n"

        if "set Heaviside function exponent" in line:
            line = "        set Heaviside function exponent                  = "+HV_exp.get()+" \n"

        if "set The state of the stiffness" in line:
            line = "        set The state of the stiffness                   = "+def_stiffness_case.get()+" \n"

        if "set Poisson's ratio" in line:
            line = "        set Poisson's ratio                              = "+poisson_ratio.get()+" \n"

        if "set The shear modulus of conrtex" in line:
            line = "        set The shear modulus of conrtex                 = "+shear_modulus.get()+" \n"

        if "set The ratio of stiffness" in line:
            line = "        set The ratio of stiffness                       = "+stiffness_ratio.get()+" \n"

        if "set The max cell density" in line:
            if (def_stiffness_case.get()=='Constant'):
                def_max_density.set(700)
            line = "        set The max cell density                         = "+def_max_density.get()+" \n"

        if "set Number global refinements" in line:
            line = "        set Number global refinements                     = "+refinement.get()+" \n"

        if "set Poly degree" in line:
            line = "        set Poly degree                                   = "+degree.get()+" \n"

        if "set Total time" in line:
            line = "        set Total time                                    = "+total_time.get()+" \n"

        if "set Time step size" in line:
            line = "        set Time step size                                = "+delt_t.get()+" \n"

        if "set Multiplier max iterations linear solver" in line:
            if (def_solver_type.get()=='Direct'):
                def_linear_it.set(100)
            line = "        set Multiplier max iterations linear solver       = "+def_linear_it.get()+" \n"

        if "set Max number newton iterations" in line:
            line = "        set Max number newton iterations                  = "+nonlinear_it.get()+" \n"

        if " set Tolerance residual deformation" in line:
            line = "        set Tolerance residual deformation                =" +tol_u.get()+" \n"

        if "set Tolerance residual diffusion" in line:
            line = "        set Tolerance residual diffusion                  = "+tol_c.get()+" \n"

        if "set Tolerance update" in line:
            line = "        set Tolerance update                              = "+update_u.get()+" \n"

        if "set Growth rate" in line:
            line = "        set Growth rate                                   = "+k_growth.get()+" \n"

        if "set Growth exponent" in line:
            line = "        set Growth exponent                               = "+growth_exp.get()+" \n"

        if "set  Growth ratio" in line:
            line = "        set  Growth ratio                                 = "+growth_ratio.get()+" \n"

        if "set Linear solver type" in line:
            line = "        set Linear solver type                            = "+def_solver_type.get()+" \n"
 
        if "set Stabilization constant" in line:
            line = "        set Stabilization constant                        = "+stability_con.get()+" \n"

        if "set c_k factor" in line:
            line = "        set c_k factor                                   = "+c_k.get()+" \n"
            
        if "set The distribution shape of Outer RGCs" in line:
            line = "        set The OSVZ regional variation                  = "+ def_ORG_variation_case.get()+"\n"
            
        if "set RG/RG_n P1" in line:
            line = "    set RG/RG_n P1    = " + def_RG_RG_n_ph1.get()+"\n"
            
        if "set RG/RG_n P2" in line:
            line = "    set RG/RG_n P2    = " + def_RG_RG_n_ph2.get()+"\n"
            
        if "set RG/RG_n P3" in line:
            line = "    set RG/RG_n P3    = " + def_RG_RG_n_ph3.get()+"\n"
            
        if "set RG/RG_n P4" in line:
            line = "    set RG/RG_n P4    = " + def_RG_RG_n_ph4.get()+"\n"
            
        if "set RG/RG_n P5" in line:
            line = "    set RG/RG_n P5    = " + def_RG_RG_n_ph5.get()+"\n"
            
        if "set IP/RG_n P1" in line:
            line = "    set IP/RG_n P1    = " + def_IP_RG_n_ph1.get()+"\n"
            
        if "set IP/RG_n P2" in line:
            line = "    set IP/RG_n P2    = " + def_IP_RG_n_ph2.get()+"\n"
            
        if "set IP/RG_n P3" in line:
            line = "    set IP/RG_n P3    = " + def_IP_RG_n_ph3.get()+"\n"
            
        if "set IP/RG_n P4" in line:
            line = "    set IP/RG_n P4    = " + def_IP_RG_n_ph4.get()+"\n"
            
        if "set IP/RG_n P5" in line:
            line = "    set IP/RG_n P5    = " + def_IP_RG_n_ph5.get()+"\n"
            
        if "set OR/RG_n P1" in line:
            line = "    set OR/RG_n P1    = " + def_OR_RG_n_ph1.get()+"\n"
            
        if "set OR/RG_n P2" in line:
            line = "    set OR/RG_n P2    = " + def_OR_RG_n_ph2.get()+"\n"
            
        if "set OR/RG_n P3" in line:
            line = "    set OR/RG_n P3    = " + def_OR_RG_n_ph3.get()+"\n"
            
        if "set OR/RG_n P4" in line:
            line = "    set OR/RG_n P4    = " + def_OR_RG_n_ph4.get()+"\n"
            
        if "set OR/RG_n P5" in line:
            line = "    set OR/RG_n P5    = " + def_OR_RG_n_ph5.get()+"\n"
            
        if "set NU/RG_n P1" in line:
            line = "    set NU/RG_n P1    = " + def_NU_RG_n_ph1.get()+"\n"
            
        if "set NU/RG_n P2" in line:
            line = "    set NU/RG_n P2    = " + def_NU_RG_n_ph2.get()+"\n"
            
        if "set NU/RG_n P3" in line:
            line = "    set NU/RG_n P3    = " + def_NU_RG_n_ph3.get()+"\n"
            
        if "set NU/RG_n P4" in line:
            line = "    set NU/RG_n P4    = " + def_NU_RG_n_ph4.get()+"\n"
            
        if "set NU/RG_n P5" in line:
            line = "    set NU/RG_n P5    = " + def_NU_RG_n_ph5.get()+"\n"
            
        if "set OR/OR_n P1" in line:
            line = "    set OR/OR_n P1    = " + def_OR_OR_n_ph1.get()+"\n"
            
        if "set OR/OR_n P2" in line:
            line = "    set OR/OR_n P2    = " + def_OR_OR_n_ph2.get()+"\n"
            
        if "set OR/OR_n P3" in line:
            line = "    set OR/OR_n P3    = " + def_OR_OR_n_ph3.get()+"\n"
            
        if "set OR/OR_n P4" in line:
            line = "    set OR/OR_n P4    = " + def_OR_OR_n_ph4.get()+"\n"
            
        if "set OR/OR_n P5" in line:
            line = "    set OR/OR_n P5    = " + def_OR_OR_n_ph5.get()+"\n"
            
        if "set IP/OR_n P1" in line:
            line = "    set IP/OR_n P1    = " + def_IP_OR_n_ph1.get()+"\n"
            
        if "set IP/OR_n P2" in line:
            line = "    set IP/OR_n P2    = " + def_IP_OR_n_ph2.get()+"\n"
            
        if "set IP/OR_n P3" in line:
            line = "    set IP/OR_n P3    = " + def_IP_OR_n_ph3.get()+"\n"
            
        if "set IP/OR_n P4" in line:
            line = "    set IP/OR_n P4    = " + def_IP_OR_n_ph4.get()+"\n"
            
        if "set IP/OR_n P5" in line:
            line = "    set IP/OR_n P5    = " + def_IP_OR_n_ph5.get()+"\n"
            
        if "set NU/OR_n P1" in line:
            line = "    set NU/OR_n P1    = " + def_NU_OR_n_ph1.get()+"\n"
            
        if "set NU/OR_n P2" in line:
            line = "    set NU/OR_n P2    = " + def_NU_OR_n_ph2.get()+"\n"
            
        if "set NU/OR_n P3" in line:
            line = "    set NU/OR_n P3    = " + def_NU_OR_n_ph3.get()+"\n"
            
        if "set NU/OR_n P4" in line:
            line = "    set NU/OR_n P4    = " + def_NU_OR_n_ph4.get()+"\n"
            
        if "set NU/OR_n P5" in line:
            line = "    set NU/OR_n P5    = " + def_NU_OR_n_ph5.get()+"\n"
            
        if "set IP/IP_n P1" in line:
            line = "    set IP/IP_n P1    = " + def_IP_IP_n_ph1.get()+"\n"
            
        if "set IP/IP_n P2" in line:
            line = "    set IP/IP_n P2    = " + def_IP_IP_n_ph2.get()+"\n"
            
        if "set IP/IP_n P3" in line:
            line = "    set IP/IP_n P3    = " + def_IP_IP_n_ph3.get()+"\n"
            
        if "set IP/IP_n P4" in line:
            line = "    set IP/IP_n P4    = " + def_IP_IP_n_ph4.get()+"\n"
            
        if "set IP/IP_n P5" in line:
            line = "    set IP/IP_n P5    = " + def_IP_IP_n_ph5.get()+"\n"
            
        if "set NU/IP_n P1" in line:
            line = "    set NU/IP_n P1    = " + def_NU_IP_n_ph1.get()+"\n"
            
        if "set NU/IP_n P2" in line:
            line = "    set NU/IP_n P2    = " + def_NU_IP_n_ph2.get()+"\n"
            
        if "set NU/IP_n P3" in line:
            line = "    set NU/IP_n P3    = " + def_NU_IP_n_ph3.get()+"\n"
            
        if "set NU/IP_n P4" in line:
            line = "    set NU/IP_n P4    = " + def_NU_IP_n_ph4.get()+"\n"
            
        if "set NU/IP_n P5" in line:
            line = "    set NU/IP_n P5    = " + def_NU_IP_n_ph5.get()+"\n"
            
        sys.stdout.write(line)

    make_run = subprocess.Popen(['./make_run.py', def_case.get()])
    root.destroy()
    
#=============================== set_default_values function ================================================================
def messageWindow():
    win2 = Toplevel(root)
    x_position2 = 565+root.winfo_x()
    y_position2 = 325+root.winfo_y()
    win2.geometry(f"300x80+{x_position2}+{y_position2}")
    win2.title('Choose')
    message2 = "Which case do you want to set?"
    L2 = Label(win2, text=message2)
    L2.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    B3 = Button(win2, text='2D', command=lambda:[ set_2D_default(), win2.destroy()])
    B3.place(relx=0.3, rely=0.7, anchor=tk.CENTER)
    B4 = Button(win2, text='3D', command=lambda:[ set_3D_default(), win2.destroy()])
    B4.place(relx=0.7, rely=0.7, anchor=tk.CENTER)
    

def set_default_values():
    win = Toplevel(root)
    x_position = 565+root.winfo_x()
    y_position = 325+root.winfo_y()
    win.geometry(f"350x80+{x_position}+{y_position}")
    win.title('Confirmation')
    message = "Are you sure that you want to set all values to  defaults?"
    L = Label(win, text=message)
    L.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    B1 = Button(win, text='Yes', command=lambda:[win.destroy(), messageWindow()])
    B1.place(relx=0.3, rely=0.7, anchor=tk.CENTER)
    B2 = Button(win, text='No', command = lambda:[win.destroy()])
    B2.place(relx=0.7, rely=0.7, anchor=tk.CENTER)
        
def set_2D_default():
    def_vz_raduis.set(0.25)
    def_svz_raduis.set(0.4)
    def_cr_thickness.set(0.1)
    def_intial_raduis.set(2)
    def_MST_factor.set(0.02)
    def_intial_division.set(1000)
    def_migration_threshold.set(500)
    def_HV_exp.set(0.008)
    def_shear_modulus.set(2.07)
    def_stiffness_ratio.set(3)
    def_max_density.set(700)
    def_stiffness_case.set('Varying')
    def_poisson_ratio.set(0.38)
    def_case.set('2')
    def_refinement.set(3)
    def_degree.set(2)
    def_total_time.set(1000)
    def_delt_t.set(0.1)
    def_stability_con.set(0.0335)
    def_c_k.set(0.33334)
    def_nonlinear_it.set(8)
    def_tol_c.set(1.0e-8)
    def_tol_u.set(1.0e-8)
    def_update_u.set(1.0e-4)
    def_solver_type.set('Direct')
    def_k_growth.set(4.7e-4)
    def_growth_exp.set(1.65)
    def_growth_ratio.set(1.5)
    def_ORG_variation_case.set(OSVZ_varying_options[0])
    def_RG_RG_n_ph1.set(2)
    def_RG_RG_n_ph2.set(1)
    def_RG_RG_n_ph3.set(1)
    def_RG_RG_n_ph4.set(1)
    def_RG_RG_n_ph5.set(1)
    def_IP_RG_n_ph1.set(0)
    def_IP_RG_n_ph2.set(1)
    def_IP_RG_n_ph3.set(0)
    def_IP_RG_n_ph4.set(0)
    def_IP_RG_n_ph5.set(0)
    def_OR_RG_n_ph1.set(0)
    def_OR_RG_n_ph2.set(0)
    def_OR_RG_n_ph3.set(1)
    def_OR_RG_n_ph4.set(0)
    def_OR_RG_n_ph5.set(0)
    def_OR_RG_n_ph5.set(0)
    def_NU_RG_n_ph1.set(0)
    def_NU_RG_n_ph2.set(0)
    def_NU_RG_n_ph3.set(0)
    def_NU_RG_n_ph4.set(0)
    def_NU_RG_n_ph5.set(0)
    def_IP_OR_n_ph1.set(0)
    def_IP_OR_n_ph2.set(0)
    def_IP_OR_n_ph3.set(0)
    def_IP_OR_n_ph4.set(1)
    def_IP_OR_n_ph5.set(0)
    def_OR_OR_n_ph1.set(0)
    def_OR_OR_n_ph2.set(0)
    def_OR_OR_n_ph3.set(1)
    def_OR_OR_n_ph4.set(1)
    def_OR_OR_n_ph5.set(1)
    def_NU_OR_n_ph1.set(0)
    def_NU_OR_n_ph2.set(0)
    def_NU_OR_n_ph3.set(0)
    def_NU_OR_n_ph4.set(0)
    def_NU_OR_n_ph5.set(0)
    def_IP_IP_n_ph1.set(0)
    def_IP_IP_n_ph2.set(0)
    def_IP_IP_n_ph3.set(1)
    def_IP_IP_n_ph4.set(1)
    def_IP_IP_n_ph5.set(0)
    def_NU_IP_n_ph1.set(0)
    def_NU_IP_n_ph2.set(0)
    def_NU_IP_n_ph3.set(2)
    def_NU_IP_n_ph4.set(2)
    def_NU_IP_n_ph5.set(4)
    def_first_phase.set(21)
    def_second_phase.set(49)
    def_third_phase.set(84)
    def_fourth_phase.set(105)
    def_IP_migration.set(0.25)
    def_OR_migration.set(10)
    def_NU_migration.set(5)
    def_RG_diffusivity.set(0.1)
    def_OR_diffusivity.set(0.1)
    def_IP_diffusivity.set(0.1)
    def_NU_diffusivity.set(0.1)


    
def set_3D_default():
    def_vz_raduis.set(0.25)
    def_svz_raduis.set(0.4)
    def_cr_thickness.set(0.1)
    def_intial_raduis.set(2)
    def_MST_factor.set(0.02)
    def_intial_division.set(10000)
    def_migration_threshold.set(500)
    def_HV_exp.set(0.008)
    def_shear_modulus.set(2.07)
    def_stiffness_ratio.set(3)
    def_max_density.set(700)
    def_stiffness_case.set('Varying')
    def_poisson_ratio.set(0.38)
    def_case.set('3')
    def_refinement.set(2)
    def_degree.set(2)
    def_total_time.set(1000)
    def_delt_t.set(0.1)
    def_stability_con.set(0.0335)
    def_c_k.set(0.33334)
    def_nonlinear_it.set(8)
    def_tol_c.set(1.0e-8)
    def_tol_u.set(1.0e-8)
    def_update_u.set(1.0e-4)
    def_solver_type.set('Direct')
    def_k_growth.set(4.7e-5)
    def_growth_exp.set(1.65)
    def_growth_ratio.set(2)
    def_ORG_variation_case.set(OSVZ_varying_options[0])
    def_RG_RG_n_ph1.set(2)
    def_RG_RG_n_ph2.set(1)
    def_RG_RG_n_ph3.set(1)
    def_RG_RG_n_ph4.set(1)
    def_RG_RG_n_ph5.set(1)
    def_IP_RG_n_ph1.set(0)
    def_IP_RG_n_ph2.set(1)
    def_IP_RG_n_ph3.set(0)
    def_IP_RG_n_ph4.set(0)
    def_IP_RG_n_ph5.set(0)
    def_OR_RG_n_ph1.set(0)
    def_OR_RG_n_ph2.set(0)
    def_OR_RG_n_ph3.set(1)
    def_OR_RG_n_ph4.set(0)
    def_OR_RG_n_ph5.set(0)
    def_OR_RG_n_ph5.set(0)
    def_NU_RG_n_ph1.set(0)
    def_NU_RG_n_ph2.set(0)
    def_NU_RG_n_ph3.set(0)
    def_NU_RG_n_ph4.set(0)
    def_NU_RG_n_ph5.set(0)
    def_IP_OR_n_ph1.set(0)
    def_IP_OR_n_ph2.set(0)
    def_IP_OR_n_ph3.set(0)
    def_IP_OR_n_ph4.set(1)
    def_IP_OR_n_ph5.set(0)
    def_OR_OR_n_ph1.set(0)
    def_OR_OR_n_ph2.set(0)
    def_OR_OR_n_ph3.set(1)
    def_OR_OR_n_ph4.set(1)
    def_OR_OR_n_ph5.set(1)
    def_NU_OR_n_ph1.set(0)
    def_NU_OR_n_ph2.set(0)
    def_NU_OR_n_ph3.set(0)
    def_NU_OR_n_ph4.set(0)
    def_NU_OR_n_ph5.set(0)
    def_IP_IP_n_ph1.set(0)
    def_IP_IP_n_ph2.set(0)
    def_IP_IP_n_ph3.set(1)
    def_IP_IP_n_ph4.set(1)
    def_IP_IP_n_ph5.set(0)
    def_NU_IP_n_ph1.set(0)
    def_NU_IP_n_ph2.set(0)
    def_NU_IP_n_ph3.set(2)
    def_NU_IP_n_ph4.set(2)
    def_NU_IP_n_ph5.set(4)
    def_first_phase.set(15)
    def_second_phase.set(22)
    def_third_phase.set(45)
    def_fourth_phase.set(75)
    def_IP_migration.set(0.25)
    def_OR_migration.set(10)
    def_NU_migration.set(5)
    def_RG_diffusivity.set(0.1)
    def_OR_diffusivity.set(0.1)
    def_IP_diffusivity.set(0.1)
    def_NU_diffusivity.set(0.1)


#======================================== Other functions ======================================

def enable_entry_it():
   linear_it.config(state= "")
   
def disable_entry_it():
   linear_it.config(state= "disabled")
   def_linear_it.set("")
   
def enable_entry_cmax():
   max_density.config(state= "")
   
def disable_entry_cmax():
   max_density.config(state= "disabled")
   def_max_density.set("")
   
def About_author():
    label_img3.configure(image=non_image_2)
    label_img3.place(relx=0.99, rely=0.99, anchor=tk.CENTER)
    label_img2.configure(image=author_image)
    label_img2.place(relx=0.8, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Mohammad Saeed Zarzor \n\n-PhD candidate in the field of Biomechanics.\n-Scientific employee in Institute of Applied Mechanics, \nFriedrich-Alexander-University Erlangen-Nürnberg. \n-Master of Science in Computational Engineering from FAU University. \n-Bachelor of Mechanical Engineering from Damascus University. ")
    info_label.place(relx=0.02, rely=0.07, anchor='nw')
    web_label.configure(text="contact details ")
    web_label.place(relx=0.02, rely=0.7, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://www.ltm.tf.fau.eu/person/zarzor-mohammad-saeed-m-sc/"))
    
def About_programm():
    label_img3.configure(image=non_image_2)
    label_img3.place(relx=0.99, rely=0.99, anchor=tk.CENTER)
    label_img2.configure(image=logo_image)
    label_img2.place(relx=0.85, rely=0.45, anchor=tk.CENTER)
    info_label.configure(text="This work is part of the BRAINIACS Project at the Institute of  Applied \nMechanics, Friedrich-Alexander-University Erlangen-Nürnberg, under \nthe supervision of Dr. Silvia Budday and in cooperation with  Prof. Dr. \nmed Ingmar Blümcke from Neuropathological Institute, University Ho-\nspitals Erlangen. We gratefully acknowledge the funding by the Deut-\nsche Forschungsgemeinschaft (DFG, German Research Foundation). \nThis work based on the following paper: ")
    info_label.place(relx=0.02, rely=0.07, anchor='nw')
    web_label.configure(text="Exploring the role of the outer subventricular zone during cortical fold-\ning through a physics-based model")
    web_label.place(relx=0.02, rely=0.7, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://www.biorxiv.org/content/10.1101/2022.09.25.509401v1.abstract"))
    
def Copy_right():
    label_img3.configure(image=non_image_2)
    label_img3.place(relx=0.99, rely=0.99, anchor=tk.CENTER)
    web_label.configure(text=" CC-BY 4.0 International license.", text_color=('red'))
    web_label.place(relx=0.02, rely=0.3, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://creativecommons.org/licenses/by/4.0/"))
    label_img2.configure(image=non_image)
    label_img2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="The copyright holder for this preprint is the author/funder, who has granted bioRxiv a license to display \nthe preprint in perpetuity. It is made available under a Copyright:")
    info_label.place(relx=0.02, rely=0.1, anchor='nw')
    


    
def callback(url):
    webbrowser.open_new(url)
    
#========================================== reset values ==========================================
def messageWindow_2(window):
    win3 = Toplevel(root)
    x_position3 = 555+root.winfo_x()
    y_position3 = 300+root.winfo_y()
    win3.geometry(f"350x100+{x_position3}+{y_position3}")
    win3.title('Error')
    message3 = "These values have already been set if you want to \nmodify them, please choose reset."
    L3 = Label(win3, text=message3)
    L3.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
    B5 = Button(win3, text='reset', command=lambda:[ win3.destroy(), reset_values(window)])
    B5.place(relx=0.3, rely=0.7, anchor=tk.CENTER)
    B6 = Button(win3, text='close', command=lambda:[ win3.destroy(), reset_values(window)])
    B6.place(relx=0.7, rely=0.7, anchor=tk.CENTER)
    
    
def reset_values(window):
    global counter_1
    global counter_2
    global counter_3
    global counter_4
    if window == 1:
        def_RG_RG_n_ph1.set(0)
        def_RG_RG_n_ph2.set(0)
        def_RG_RG_n_ph3.set(0)
        def_RG_RG_n_ph4.set(0)
        def_RG_RG_n_ph5.set(0)
        def_IP_RG_n_ph1.set(0)
        def_IP_RG_n_ph2.set(0)
        def_IP_RG_n_ph3.set(0)
        def_IP_RG_n_ph4.set(0)
        def_IP_RG_n_ph5.set(0)
        def_OR_RG_n_ph1.set(0)
        def_OR_RG_n_ph2.set(0)
        def_OR_RG_n_ph3.set(0)
        def_OR_RG_n_ph4.set(0)
        def_OR_RG_n_ph5.set(0)
        def_OR_RG_n_ph5.set(0)
        def_NU_RG_n_ph1.set(0)
        def_NU_RG_n_ph2.set(0)
        def_NU_RG_n_ph4.set(0)
        def_NU_RG_n_ph5.set(0)
        def_IP_OR_n_ph1.set(0)
        def_IP_OR_n_ph2.set(0)
        def_IP_OR_n_ph3.set(0)
        def_IP_OR_n_ph4.set(0)
        def_IP_OR_n_ph5.set(0)
        def_OR_OR_n_ph1.set(0)
        def_OR_OR_n_ph2.set(0)
        def_OR_OR_n_ph3.set(0)
        def_OR_OR_n_ph4.set(0)
        def_OR_OR_n_ph5.set(0)
        def_NU_OR_n_ph1.set(0)
        def_NU_OR_n_ph2.set(0)
        def_NU_OR_n_ph3.set(0)
        def_NU_OR_n_ph4.set(0)
        def_NU_OR_n_ph5.set(0)
        def_IP_IP_n_ph1.set(0)
        def_IP_IP_n_ph2.set(0)
        def_IP_IP_n_ph3.set(0)
        def_IP_IP_n_ph4.set(0)
        def_IP_IP_n_ph5.set(0)
        def_NU_IP_n_ph1.set(0)
        def_NU_IP_n_ph2.set(0)
        def_NU_IP_n_ph3.set(0)
        def_NU_IP_n_ph4.set(0)
        def_NU_IP_n_ph5.set(0)
        counter_1 = 0
        open_division_window()
    if window == 2:
        def_first_phase.set("")
        def_second_phase.set("")
        def_third_phase.set("")
        def_fourth_phase.set("")
        counter_2 = 0
        open_division_time_window()
    if window == 3:
        def_IP_migration.set("")
        def_NU_migration.set("")
        def_OR_migration.set("")
        counter_3 = 0
        open_migration_window()
    if window == 4:
        def_RG_diffusivity.set("")
        def_IP_diffusivity.set("")
        def_OR_diffusivity.set("")
        def_NU_diffusivity.set("")
        counter_4 = 0
        open_diffusion_window()
        

#======================================== divison window =========================================
def open_division_window():

    global counter_1

    def up_Ph1_RG_label(*args):
        RG_last[1] =  RG_last[0] * int(d_RG_RG_n_ph1.get())
        if (RG_last[1] > 0):
            RG_ph1_im.configure(image = RG)
            RG_RG_n_Ph2.config(state= "normal")
            OR_RG_n_Ph2.config(state= "normal")
            IP_RG_n_Ph2.config(state= "normal")
            NU_RG_n_Ph2.config(state= "normal")
            if( RG_last[1] > 1):
                Ph1_RG_label.configure(text = RG_last[1])

        else:
            Ph1_RG_label.configure(text = "")
            RG_ph1_im.configure(image = nonR)
            RG_RG_n_Ph2.config(state= "disabled")
            OR_RG_n_Ph2.config(state= "disabled")
            IP_RG_n_Ph2.config(state= "disabled")
            NU_RG_n_Ph2.config(state= "disabled")
            
    def up_Ph1_IP_label(*args):
        IP_last[1] = RG_last[0] * int(d_IP_RG_n_ph1.get())
        if (IP_last[1] > 0):
            IP_ph1_im.configure(image = IP)
            IP_IP_n_Ph2.config(state= "normal")
            NU_IP_n_Ph2.config(state= "normal")
            if(IP_last[1] > 1):
                Ph1_IP_label.configure(text =  IP_last[1])
        else:
            Ph1_IP_label.configure(text = "")
            IP_ph1_im.configure(image = nonR)
            IP_IP_n_Ph2.config(state= "disabled")
            NU_IP_n_Ph2.config(state= "disabled")

            
    def up_Ph1_OR_label(*args):
        OR_last[1] =  RG_last[0] * int(d_OR_RG_n_ph1.get())
        if (OR_last[1] > 0):
            OR_ph1_im.configure(image = OR)
            OR_OR_n_Ph2.config(state= "normal")
            IP_OR_n_Ph2.config(state= "normal")
            NU_OR_n_Ph2.config(state= "normal")
            if(OR_last[1] > 1):
                Ph1_OR_label.configure(text = OR_last[1])

        else:
            Ph1_OR_label.configure(text = "")
            OR_ph1_im.configure(image = nonR)
            OR_OR_n_Ph2.config(state= "disabled")
            IP_OR_n_Ph2.config(state= "disabled")
            NU_OR_n_Ph2.config(state= "disabled")
            
    def up_Ph1_NU_label(*args):
        NU_last[1] = RG_last[0]* int(d_NU_RG_n_ph1.get())
        if (NU_last[1] > 0):
            NU_ph1_im.configure(image = NU)
            if(NU_last[1] > 1):
                Ph1_NU_label.configure(text =  NU_last[1] )
        else:
            NU_ph1_im.configure(image = nonR)
            Ph1_NU_label.configure(text = "")
            
    def up_Ph2_RG_label(*args):
        RG_last[2] =  RG_last[1] * int(d_RG_RG_n_ph2.get())
        if ( RG_last[2] > 0):
            RG_ph2_im.configure(image = RG)
            RG_RG_n_Ph3.config(state= "normal")
            OR_RG_n_Ph3.config(state= "normal")
            IP_RG_n_Ph3.config(state= "normal")
            NU_RG_n_Ph3.config(state= "normal")
            if(  RG_last[2] > 1):
                Ph2_RG_label.configure(text =  RG_last[2] )
        else:
            Ph2_RG_label.configure(text = "")
            RG_ph2_im.configure(image = nonR)
            RG_RG_n_Ph3.config(state= "disabled")
            OR_RG_n_Ph3.config(state= "disabled")
            IP_RG_n_Ph3.config(state= "disabled")
            NU_RG_n_Ph3.config(state= "disabled")

    def up_Ph2_NU_label(*args):
        NU_last[2] = RG_last[1] * int(d_NU_RG_n_ph2.get()) + OR_last[1] * int(d_NU_OR_n_ph2.get()) + IP_last[1] * int(d_NU_IP_n_ph2.get()) + NU_last[1]
        if ( NU_last[2] >0 ):
            NU_ph2_im.configure(image = NU)
            if( NU_last[2] > 1):
                Ph2_NU_label.configure(text = NU_last[2])
        else:
            Ph2_NU_label.configure(text = "")
            NU_ph2_im.configure(image = nonR)

    def up_Ph2_OR_label(*args):
        OR_last[2] = OR_last[1] * int(d_OR_OR_n_ph2.get()) + RG_last[1] * int(d_OR_RG_n_ph2.get())
        if (  OR_last[2] >0 ):
            OR_ph2_im.configure(image = OR)
            OR_OR_n_Ph3.config(state= "normal")
            IP_OR_n_Ph3.config(state= "normal")
            NU_OR_n_Ph3.config(state= "normal")
            if(  OR_last[2] > 1):
                Ph2_OR_label.configure(text = OR_last[2] )
        else:
            Ph2_OR_label.configure(text = "")
            OR_ph2_im.configure(image = nonR)
            OR_OR_n_Ph3.config(state= "disabled")
            IP_OR_n_Ph3.config(state= "disabled")
            NU_OR_n_Ph3.config(state= "disabled")

    def up_Ph2_IP_label(*args):
        IP_last[2] = IP_last[1] * int(d_IP_IP_n_ph2.get()) + RG_last[1] * int(d_IP_RG_n_ph2.get()) + OR_last[1] * int(d_IP_OR_n_ph2.get())
        if (IP_last[2] >0 ):
            IP_ph2_im.configure(image = IP)
            IP_IP_n_Ph3.config(state= "normal")
            NU_IP_n_Ph3.config(state= "normal")
            if(IP_last[2] > 1):
                Ph2_IP_label.configure(text =  IP_last[2])
        else:
            Ph2_IP_label.configure(text = "")
            IP_ph2_im.configure(image = nonR)
            IP_IP_n_Ph3.config(state= "disabled")
            NU_IP_n_Ph3.config(state= "disabled")
        

    def up_Ph3_RG_label(*args):
        RG_last[3] = RG_last[2] * int(d_RG_RG_n_ph3.get())
        if (RG_last[3] > 0):
            RG_ph3_im.configure(image = RG)
            RG_RG_n_Ph4.config(state= "normal")
            OR_RG_n_Ph4.config(state= "normal")
            IP_RG_n_Ph4.config(state= "normal")
            NU_RG_n_Ph4.config(state= "normal")
            if( RG_last[3] > 1):
                Ph3_RG_label.configure(text = RG_last[3])
        else:
            Ph3_RG_label.configure(text = "")
            RG_ph3_im.configure(image = nonR)
            RG_RG_n_Ph4.config(state= "disabled")
            OR_RG_n_Ph4.config(state= "disabled")
            IP_RG_n_Ph4.config(state= "disabled")
            NU_RG_n_Ph4.config(state= "disabled")

    def up_Ph3_NU_label(*args):
        NU_last[3] =  RG_last[2] * int(d_NU_RG_n_ph3.get()) + OR_last[2] * int(d_NU_OR_n_ph3.get()) + IP_last[2] * int(d_NU_IP_n_ph3.get()) + NU_last[2]
        if (  NU_last[3] >0 ):
            NU_ph3_im.configure(image = NU)
            if(  NU_last[3] > 1):
                Ph3_NU_label.configure(text = NU_last[3])
        else:
            Ph3_NU_label.configure(text = "")
            NU_ph3_im.configure(image = nonR)
            
    def up_Ph3_OR_label(*args):
        OR_last[3] = OR_last[2] * int(d_OR_OR_n_ph3.get()) + RG_last[2] * int(d_OR_RG_n_ph3.get())
        if (  OR_last[3] >0 ):
            OR_ph3_im.configure(image = OR)
            OR_OR_n_Ph4.config(state= "normal")
            IP_OR_n_Ph4.config(state= "normal")
            NU_OR_n_Ph4.config(state= "normal")
            if(  OR_last[3] > 1):
                Ph3_OR_label.configure(text = OR_last[3] )
        else:
            Ph3_OR_label.configure(text = "")
            OR_ph3_im.configure(image = nonR)
            OR_OR_n_Ph4.config(state= "disabled")
            IP_OR_n_Ph4.config(state= "disabled")
            NU_OR_n_Ph4.config(state= "disabled")

    def up_Ph3_IP_label(*args):
        IP_last[3] = IP_last[2] * int(d_IP_IP_n_ph3.get()) + RG_last[2] * int(d_IP_RG_n_ph3.get()) + OR_last[2] * int(d_IP_OR_n_ph3.get())
        if (IP_last[3] >0 ):
            IP_ph3_im.configure(image = IP)
            IP_IP_n_Ph4.config(state= "normal")
            NU_IP_n_Ph4.config(state= "normal")
            if(IP_last[3] > 1):
                Ph3_IP_label.configure(text =  IP_last[3])
        else:
            Ph3_IP_label.configure(text = "")
            IP_ph3_im.configure(image = nonR)
            IP_IP_n_Ph4.config(state= "disabled")
            NU_IP_n_Ph4.config(state= "disabled")

    def up_Ph4_RG_label(*args):
        RG_last[4] = RG_last[3] * int(d_RG_RG_n_ph4.get())
        if (RG_last[4] > 0):
            RG_ph4_im.configure(image = RG)
            RG_RG_n_Ph5.config(state= "normal")
            OR_RG_n_Ph5.config(state= "normal")
            IP_RG_n_Ph5.config(state= "normal")
            NU_RG_n_Ph5.config(state= "normal")
            if( RG_last[4] > 1):
                Ph4_RG_label.configure(text = RG_last[4])
        else:
            Ph4_RG_label.configure(text = "")
            RG_ph4_im.configure(image = nonR)
            RG_RG_n_Ph5.config(state= "disabled")
            OR_RG_n_Ph5.config(state= "disabled")
            IP_RG_n_Ph5.config(state= "disabled")
            NU_RG_n_Ph5.config(state= "disabled")

    def up_Ph4_NU_label(*args):
        NU_last[4] =  RG_last[3] * int(d_NU_RG_n_ph4.get()) + OR_last[3] * int(d_NU_OR_n_ph4.get()) + IP_last[3] * int(d_NU_IP_n_ph4.get()) + NU_last[3]
        if (  NU_last[4] >0 ):
            NU_ph4_im.configure(image = NU)
            if(  NU_last[4] > 1):
                Ph4_NU_label.configure(text = NU_last[4])
        else:
            Ph4_NU_label.configure(text = "")
            NU_ph4_im.configure(image = nonR)
            
    def up_Ph4_OR_label(*args):
        OR_last[4] = OR_last[3] * int(d_OR_OR_n_ph4.get()) + RG_last[3] * int(d_OR_RG_n_ph4.get())
        if (  OR_last[4] >0 ):
            OR_ph4_im.configure(image = OR)
            OR_OR_n_Ph5.config(state= "normal")
            IP_OR_n_Ph5.config(state= "normal")
            NU_OR_n_Ph5.config(state= "normal")
            if(  OR_last[4] > 1):
                Ph4_OR_label.configure(text = OR_last[4] )
        else:
            Ph4_OR_label.configure(text = "")
            OR_ph4_im.configure(image = nonR)
            OR_OR_n_Ph5.config(state= "disabled")
            IP_OR_n_Ph5.config(state= "disabled")
            NU_OR_n_Ph5.config(state= "disabled")

    def up_Ph4_IP_label(*args):
        IP_last[4] = IP_last[3] * int(d_IP_IP_n_ph4.get()) + RG_last[3] * int(d_IP_RG_n_ph4.get()) + OR_last[3] * int(d_IP_OR_n_ph4.get())
        if (IP_last[4] >0 ):
            IP_ph4_im.configure(image = IP)
            IP_IP_n_Ph5.config(state= "normal")
            NU_IP_n_Ph5.config(state= "normal")
            if(IP_last[4] > 1):
                Ph4_IP_label.configure(text =  IP_last[4])
        else:
            Ph4_IP_label.configure(text = "")
            IP_ph4_im.configure(image = nonR)
            IP_IP_n_Ph5.config(state= "disabled")
            NU_IP_n_Ph5.config(state= "disabled")

    def up_Ph5_RG_label(*args):
        RG_last[5] = RG_last[4] * int(d_RG_RG_n_ph5.get())
        if (RG_last[5] > 0):
            RG_ph5_im.configure(image = RG)
            if( RG_last[5] > 1):
                Ph5_RG_label.configure(text = RG_last[5])
        else:
            Ph5_RG_label.configure(text = "")
            RG_ph5_im.configure(image = nonR)

    def up_Ph5_NU_label(*args):
        NU_last[5] =  RG_last[4] * int(d_NU_RG_n_ph5.get()) + OR_last[4] * int(d_NU_OR_n_ph5.get()) + IP_last[4] * int(d_NU_IP_n_ph5.get()) + NU_last[4]
        if (  NU_last[5] >0 ):
            NU_ph5_im.configure(image = NU)
            if(  NU_last[5] > 1):
                Ph5_NU_label.configure(text = NU_last[5])
        else:
            Ph5_NU_label.configure(text = "")
            NU_ph5_im.configure(image = nonR)
            
    def up_Ph5_OR_label(*args):
        OR_last[5] = OR_last[4] * int(d_OR_OR_n_ph5.get()) + RG_last[4] * int(d_OR_RG_n_ph5.get())
        if (  OR_last[5] >0 ):
            OR_ph5_im.configure(image = OR)
            if(  OR_last[5] > 1):
                Ph5_OR_label.configure(text = OR_last[5] )
        else:
            Ph5_OR_label.configure(text = "")
            OR_ph5_im.configure(image = nonR)

    def up_Ph5_IP_label(*args):
        IP_last[5] = IP_last[4] * int(d_IP_IP_n_ph5.get()) + RG_last[4] * int(d_IP_RG_n_ph5.get()) + OR_last[4] * int(d_IP_OR_n_ph5.get())
        if (IP_last[5] >0 ):
            IP_ph5_im.configure(image = IP)
            if(IP_last[5] > 1):
                Ph5_IP_label.configure(text =  IP_last[5])
        else:
            Ph5_IP_label.configure(text = "")
            IP_ph5_im.configure(image = nonR)

    def Set_values():
        def_RG_RG_n_ph1.set(d_RG_RG_n_ph1.get())
        def_RG_RG_n_ph2.set(d_RG_RG_n_ph2.get())
        def_RG_RG_n_ph3.set(d_RG_RG_n_ph3.get())
        def_RG_RG_n_ph4.set(d_RG_RG_n_ph4.get())
        def_RG_RG_n_ph5.set(d_RG_RG_n_ph5.get())
        def_IP_RG_n_ph1.set(d_IP_RG_n_ph1.get())
        def_IP_RG_n_ph2.set(d_IP_RG_n_ph2.get())
        def_IP_RG_n_ph3.set(d_IP_RG_n_ph3.get())
        def_IP_RG_n_ph4.set(d_IP_RG_n_ph4.get())
        def_IP_RG_n_ph5.set(d_IP_RG_n_ph5.get())
        def_OR_RG_n_ph1.set(d_OR_RG_n_ph1.get())
        def_OR_RG_n_ph2.set(d_OR_RG_n_ph2.get())
        def_OR_RG_n_ph3.set(d_OR_RG_n_ph3.get())
        def_OR_RG_n_ph4.set(d_OR_RG_n_ph4.get())
        def_OR_RG_n_ph5.set(d_OR_RG_n_ph5.get())
        def_OR_RG_n_ph5.set(d_OR_RG_n_ph5.get())
        def_NU_RG_n_ph1.set(d_NU_RG_n_ph1.get())
        def_NU_RG_n_ph2.set(d_NU_RG_n_ph2.get())
        def_NU_RG_n_ph4.set(d_NU_RG_n_ph4.get())
        def_NU_RG_n_ph5.set(d_NU_RG_n_ph5.get())
        def_IP_OR_n_ph1.set(d_IP_OR_n_ph1.get())
        def_IP_OR_n_ph2.set(d_IP_OR_n_ph2.get())
        def_IP_OR_n_ph3.set(d_IP_OR_n_ph3.get())
        def_IP_OR_n_ph4.set(d_IP_OR_n_ph4.get())
        def_IP_OR_n_ph5.set(d_IP_OR_n_ph5.get())
        def_OR_OR_n_ph1.set(d_OR_OR_n_ph1.get())
        def_OR_OR_n_ph2.set(d_OR_OR_n_ph2.get())
        def_OR_OR_n_ph3.set(d_OR_OR_n_ph3.get())
        def_OR_OR_n_ph4.set(d_OR_OR_n_ph4.get())
        def_OR_OR_n_ph5.set(d_OR_OR_n_ph5.get())
        def_NU_OR_n_ph1.set(d_NU_OR_n_ph1.get())
        def_NU_OR_n_ph2.set(d_NU_OR_n_ph2.get())
        def_NU_OR_n_ph3.set(d_NU_OR_n_ph3.get())
        def_NU_OR_n_ph4.set(d_NU_OR_n_ph4.get())
        def_NU_OR_n_ph5.set(d_NU_OR_n_ph5.get())
        def_IP_IP_n_ph1.set(d_IP_IP_n_ph1.get())
        def_IP_IP_n_ph2.set(d_IP_IP_n_ph2.get())
        def_IP_IP_n_ph3.set(d_IP_IP_n_ph3.get())
        def_IP_IP_n_ph4.set(d_IP_IP_n_ph4.get())
        def_IP_IP_n_ph5.set(d_IP_IP_n_ph5.get())
        def_NU_IP_n_ph1.set(d_NU_IP_n_ph1.get())
        def_NU_IP_n_ph2.set(d_NU_IP_n_ph2.get())
        def_NU_IP_n_ph3.set(d_NU_IP_n_ph3.get())
        def_NU_IP_n_ph4.set(d_NU_IP_n_ph4.get())
        def_NU_IP_n_ph5.set(d_NU_IP_n_ph5.get())
        divison_root.destroy()

    if counter_1 < 1:
        
        RG_last = np.array([1,0,0,0,0,0])
        OR_last = np.array([0,0,0,0,0,0])
        IP_last = np.array([0,0,0,0,0,0])
        NU_last = np.array([0,0,0,0,0,0])

        global cell_divsion_image
        divison_root = Toplevel(root)
        x_position = 650+root.winfo_x()
        y_position = 50+root.winfo_y()
        divison_root.geometry(f"750x600+{x_position}+{y_position}")
        divison_root.title("Adjust cells division rate values")
        cell_divsion_dark = Image.open('Images/cell_divsion_dark.png')
        cell_divsion_light = Image.open('Images/cell_divsion.png')
        RG_insert = Image.open('Images/RG.png')
        OR_insert = Image.open('Images/OR.png')
        IP_insert = Image.open('Images/IP.png')
        NU_insert = Image.open('Images/NU.png')
        non_insert = Image.open('Images/nonR.png')
        
        d_RG_RG_n_ph1 = tk.StringVar(divison_root, "")
        d_RG_RG_n_ph2 = tk.StringVar(divison_root, "")
        d_RG_RG_n_ph3 = tk.StringVar(divison_root, "")
        d_RG_RG_n_ph4 = tk.StringVar(divison_root, "")
        d_RG_RG_n_ph5 = tk.StringVar(divison_root, "")
        d_IP_RG_n_ph1 = tk.StringVar(divison_root, "")
        d_IP_RG_n_ph2 = tk.StringVar(divison_root, "")
        d_IP_RG_n_ph3 = tk.StringVar(divison_root, "")
        d_IP_RG_n_ph4 = tk.StringVar(divison_root, "")
        d_IP_RG_n_ph5 = tk.StringVar(divison_root, "")
        d_OR_RG_n_ph1 = tk.StringVar(divison_root, "")
        d_OR_RG_n_ph2 = tk.StringVar(divison_root, "")
        d_OR_RG_n_ph3 = tk.StringVar(divison_root, "")
        d_OR_RG_n_ph4 = tk.StringVar(divison_root, "")
        d_OR_RG_n_ph5 = tk.StringVar(divison_root, "")
        d_NU_RG_n_ph1 = tk.StringVar(divison_root, "")
        d_NU_RG_n_ph2 = tk.StringVar(divison_root, "")
        d_NU_RG_n_ph3 = tk.StringVar(divison_root, "")
        d_NU_RG_n_ph4 = tk.StringVar(divison_root, "")
        d_NU_RG_n_ph5 = tk.StringVar(divison_root, "")
        d_IP_OR_n_ph1 = tk.StringVar(divison_root, "")
        d_IP_OR_n_ph2 = tk.StringVar(divison_root, "")
        d_IP_OR_n_ph3 = tk.StringVar(divison_root, "")
        d_IP_OR_n_ph4 = tk.StringVar(divison_root, "")
        d_IP_OR_n_ph5 = tk.StringVar(divison_root, "")
        d_OR_OR_n_ph1 = tk.StringVar(divison_root, "")
        d_OR_OR_n_ph2 = tk.StringVar(divison_root, "")
        d_OR_OR_n_ph3 = tk.StringVar(divison_root, "")
        d_OR_OR_n_ph4 = tk.StringVar(divison_root, "")
        d_OR_OR_n_ph5 = tk.StringVar(divison_root, "")
        d_NU_OR_n_ph1 = tk.StringVar(divison_root, "")
        d_NU_OR_n_ph2 = tk.StringVar(divison_root, "")
        d_NU_OR_n_ph3 = tk.StringVar(divison_root, "")
        d_NU_OR_n_ph4 = tk.StringVar(divison_root, "")
        d_NU_OR_n_ph5 = tk.StringVar(divison_root, "")
        d_IP_IP_n_ph1 = tk.StringVar(divison_root, "")
        d_IP_IP_n_ph2 = tk.StringVar(divison_root, "")
        d_IP_IP_n_ph3 = tk.StringVar(divison_root, "")
        d_IP_IP_n_ph4 = tk.StringVar(divison_root, "")
        d_IP_IP_n_ph5 = tk.StringVar(divison_root, "")
        d_NU_IP_n_ph1 = tk.StringVar(divison_root, "")
        d_NU_IP_n_ph2 = tk.StringVar(divison_root, "")
        d_NU_IP_n_ph3 = tk.StringVar(divison_root, "")
        d_NU_IP_n_ph4 = tk.StringVar(divison_root, "")
        d_NU_IP_n_ph5 = tk.StringVar(divison_root, "")
        
        
        w,h = cell_divsion_light.size
        cell_divsion_image = customtkinter.CTkImage(light_image=cell_divsion_light, dark_image=cell_divsion_dark, size=cell_divsion_light.size)
        nonR = customtkinter.CTkImage(light_image=non_insert, dark_image=non_insert, size=non_insert.size)
        RG = customtkinter.CTkImage(light_image=RG_insert, dark_image=RG_insert, size=RG_insert.size)
        OR = customtkinter.CTkImage(light_image=OR_insert, dark_image=OR_insert, size=OR_insert.size)
        IP = customtkinter.CTkImage(light_image=IP_insert, dark_image=IP_insert, size=IP_insert.size)
        NU = customtkinter.CTkImage(light_image=NU_insert, dark_image=NU_insert, size=NU_insert.size)
        label_img_division = customtkinter.CTkLabel(divison_root, image = cell_divsion_image, text="")
        label_img_division.place(relx=0.73, rely=0.57, anchor=tk.CENTER)
        label_img_division = customtkinter.CTkLabel(divison_root, image = RG, text="")
        label_img_division.place(relx=0.7, rely=0.95, anchor=tk.CENTER)

        
        frame_Ph1 = customtkinter.CTkFrame(master=divison_root, width=300, height=80,corner_radius=10)
        frame_Ph1.place(x=360, y=9, anchor='nw')
        frame_Ph2 = customtkinter.CTkFrame(master=divison_root, width=300, height=140,corner_radius=10)
        frame_Ph2.place(x=30, y=9, anchor='nw')
        frame_Ph3 = customtkinter.CTkFrame(master=divison_root, width=300, height=140,corner_radius=10)
        frame_Ph3.place(x=30, y=158, anchor='nw')
        frame_Ph4 = customtkinter.CTkFrame(master=divison_root, width=300, height=140,corner_radius=10)
        frame_Ph4.place(x=30, y=307, anchor='nw')
        frame_Ph5 = customtkinter.CTkFrame(master=divison_root, width=300, height=140,corner_radius=10)
        frame_Ph5.place(x=30, y=456, anchor='nw')

        
        
        
        Ph1_RG_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph1_RG_label.place(relx=0.83, rely=0.85, anchor=tk.CENTER)
        RG_ph1_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        RG_ph1_im.place(relx=0.88, rely=0.85, anchor=tk.CENTER)
        Ph1_OR_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph1_OR_label.place(relx=0.74, rely=0.85, anchor=tk.CENTER)
        OR_ph1_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        OR_ph1_im.place(relx=0.78, rely=0.85, anchor=tk.CENTER)
        Ph1_IP_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph1_IP_label.place(relx=0.63, rely=0.85, anchor=tk.CENTER)
        IP_ph1_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        IP_ph1_im.place(relx=0.68, rely=0.85, anchor=tk.CENTER)
        Ph1_NU_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph1_NU_label.place(relx=0.5, rely=0.85, anchor=tk.CENTER)
        NU_ph1_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        NU_ph1_im.place(relx=0.55, rely=0.85, anchor=tk.CENTER)
        Ph2_RG_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph2_RG_label.place(relx=0.83, rely=0.74, anchor=tk.CENTER)
        RG_ph2_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        RG_ph2_im.place(relx=0.88, rely=0.74, anchor=tk.CENTER)
        Ph2_OR_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph2_OR_label.place(relx=0.74, rely=0.74, anchor=tk.CENTER)
        OR_ph2_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        OR_ph2_im.place(relx=0.78, rely=0.74, anchor=tk.CENTER)
        Ph2_IP_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph2_IP_label.place(relx=0.63, rely=0.74, anchor=tk.CENTER)
        IP_ph2_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        IP_ph2_im.place(relx=0.68, rely=0.74, anchor=tk.CENTER)
        Ph2_NU_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph2_NU_label.place(relx=0.5, rely=0.74, anchor=tk.CENTER)
        NU_ph2_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        NU_ph2_im.place(relx=0.55, rely=0.74, anchor=tk.CENTER)
        Ph3_RG_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph3_RG_label.place(relx=0.83, rely=0.62, anchor=tk.CENTER)
        RG_ph3_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        RG_ph3_im.place(relx=0.88, rely=0.62, anchor=tk.CENTER)
        Ph3_OR_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph3_OR_label.place(relx=0.74, rely=0.62, anchor=tk.CENTER)
        OR_ph3_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        OR_ph3_im.place(relx=0.78, rely=0.62, anchor=tk.CENTER)
        Ph3_IP_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph3_IP_label.place(relx=0.63, rely=0.62, anchor=tk.CENTER)
        IP_ph3_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        IP_ph3_im.place(relx=0.68, rely=0.62, anchor=tk.CENTER)
        Ph3_NU_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph3_NU_label.place(relx=0.5, rely=0.62, anchor=tk.CENTER)
        NU_ph3_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        NU_ph3_im.place(relx=0.55, rely=0.62, anchor=tk.CENTER)
        Ph4_RG_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph4_RG_label.place(relx=0.83, rely=0.51, anchor=tk.CENTER)
        RG_ph4_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        RG_ph4_im.place(relx=0.88, rely=0.51, anchor=tk.CENTER)
        Ph4_OR_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph4_OR_label.place(relx=0.74, rely=0.51, anchor=tk.CENTER)
        OR_ph4_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        OR_ph4_im.place(relx=0.78, rely=0.51, anchor=tk.CENTER)
        Ph4_IP_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph4_IP_label.place(relx=0.63, rely=0.51, anchor=tk.CENTER)
        IP_ph4_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        IP_ph4_im.place(relx=0.68, rely=0.51, anchor=tk.CENTER)
        Ph4_NU_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph4_NU_label.place(relx=0.5, rely=0.51, anchor=tk.CENTER)
        NU_ph4_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        NU_ph4_im.place(relx=0.55, rely=0.51, anchor=tk.CENTER)
        Ph5_RG_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph5_RG_label.place(relx=0.83, rely=0.4, anchor=tk.CENTER)
        RG_ph5_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        RG_ph5_im.place(relx=0.88, rely=0.4, anchor=tk.CENTER)
        Ph5_OR_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph5_OR_label.place(relx=0.74, rely=0.4, anchor=tk.CENTER)
        OR_ph5_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        OR_ph5_im.place(relx=0.78, rely=0.4, anchor=tk.CENTER)
        Ph5_IP_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph5_IP_label.place(relx=0.63, rely=0.4, anchor=tk.CENTER)
        IP_ph5_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        IP_ph5_im.place(relx=0.68, rely=0.4, anchor=tk.CENTER)
        Ph5_NU_label = customtkinter.CTkLabel(master=divison_root, text="", font=("Roboto", 14))
        Ph5_NU_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        NU_ph5_im = customtkinter.CTkLabel(master=divison_root, image=nonR,  text="")
        NU_ph5_im.place(relx=0.55, rely=0.4, anchor=tk.CENTER)
        
        First_Phase_label = customtkinter.CTkLabel(master=frame_Ph1, text="First Phase", font=("Roboto", 16, "bold"))
        First_Phase_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        RG_n_Ph1 = customtkinter.CTkLabel(master=frame_Ph1, text="RG", font=("Roboto", 14))
        RG_n_Ph1.place(relx=0.075, rely=0.8, anchor=tk.CENTER)
        RG_Ph1 = customtkinter.CTkLabel(master=frame_Ph1, text="RG", font=("Roboto", 14))
        RG_Ph1.place(relx=0.25, rely=0.5, anchor=tk.CENTER)
        OR_Ph1 = customtkinter.CTkLabel(master=frame_Ph1, text="OR", font=("Roboto", 14))
        OR_Ph1.place(relx=0.45, rely=0.5, anchor=tk.CENTER)
        IP_Ph1 = customtkinter.CTkLabel(master=frame_Ph1, text="IP", font=("Roboto", 14))
        IP_Ph1.place(relx=0.65, rely=0.5, anchor=tk.CENTER)
        NU_Ph1 = customtkinter.CTkLabel(master=frame_Ph1, text="NU", font=("Roboto", 14))
        NU_Ph1.place(relx=0.85, rely=0.5, anchor=tk.CENTER)
        RG_RG_n_Ph1 = tk.Spinbox(frame_Ph1, from_= 0, to = 4,width=4, increment=1,textvariable= d_RG_RG_n_ph1,font=("Aral",10))
        RG_RG_n_Ph1.place(relx=0.25, rely=0.8, anchor=tk.CENTER)
        d_RG_RG_n_ph1.trace('w', up_Ph1_RG_label)
        OR_RG_n_Ph1 = tk.Spinbox(frame_Ph1, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_RG_n_ph1,font=("Aral",10))
        OR_RG_n_Ph1.place(relx=0.45, rely=0.8, anchor=tk.CENTER)
        d_OR_RG_n_ph1.trace('w', up_Ph1_OR_label)
        IP_RG_n_Ph1 = tk.Spinbox(frame_Ph1, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_RG_n_ph1,font=("Aral",10))
        IP_RG_n_Ph1.place(relx=0.65, rely=0.8, anchor=tk.CENTER)
        d_IP_RG_n_ph1.trace('w', up_Ph1_IP_label)
        NU_RG_n_Ph1 = tk.Spinbox(frame_Ph1, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_RG_n_ph1,font=("Aral",10))
        NU_RG_n_Ph1.place(relx=0.85, rely=0.8, anchor=tk.CENTER)
        d_NU_RG_n_ph1.trace('w', up_Ph1_NU_label)
        
        Second_Phase_label = customtkinter.CTkLabel(master=frame_Ph2, text="Second Phase", font=("Roboto", 16, "bold"))
        Second_Phase_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        RG_n_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="RG", font=("Roboto", 14))
        RG_n_Ph2.place(relx=0.075, rely=0.45, anchor=tk.CENTER)
        OR_n_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="OR", font=("Roboto", 14))
        OR_n_Ph2.place(relx=0.075, rely=0.65, anchor=tk.CENTER)
        IP_n_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="IP", font=("Roboto", 14))
        IP_n_Ph2.place(relx=0.075, rely=0.85, anchor=tk.CENTER)
        RG_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="RG", font=("Roboto", 14))
        RG_Ph2.place(relx=0.25, rely=0.275, anchor=tk.CENTER)
        OR_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="OR", font=("Roboto", 14))
        OR_Ph2.place(relx=0.45, rely=0.275, anchor=tk.CENTER)
        IP_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="IP", font=("Roboto", 14))
        IP_Ph2.place(relx=0.65, rely=0.275, anchor=tk.CENTER)
        NU_Ph2 = customtkinter.CTkLabel(master=frame_Ph2, text="NU", font=("Roboto", 14))
        NU_Ph2.place(relx=0.85, rely=0.275, anchor=tk.CENTER)
        RG_RG_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_RG_RG_n_ph2,font=("Aral",10), state = "disabled")
        RG_RG_n_Ph2.place(relx=0.25, rely=0.45, anchor=tk.CENTER)
        d_RG_RG_n_ph2.trace('w', up_Ph2_RG_label)
        OR_RG_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_RG_n_ph2,font=("Aral",10), state = "disabled")
        OR_RG_n_Ph2.place(relx=0.45, rely=0.45, anchor=tk.CENTER)
        d_OR_RG_n_ph2.trace('w', up_Ph2_OR_label)
        IP_RG_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_RG_n_ph2,font=("Aral",10), state = "disabled")
        IP_RG_n_Ph2.place(relx=0.65, rely=0.45, anchor=tk.CENTER)
        d_IP_RG_n_ph2.trace('w', up_Ph2_IP_label)
        NU_RG_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_RG_n_ph2,font=("Aral",10), state = "disabled")
        NU_RG_n_Ph2.place(relx=0.85, rely=0.45, anchor=tk.CENTER)
        d_NU_RG_n_ph2.trace('w', up_Ph2_NU_label)
        OR_OR_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_OR_n_ph2,font=("Aral",10), state = "disabled")
        OR_OR_n_Ph2.place(relx=0.45, rely=0.65, anchor=tk.CENTER)
        d_OR_OR_n_ph2.trace('w', up_Ph2_OR_label)
        IP_OR_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_OR_n_ph2,font=("Aral",10), state = "disabled")
        IP_OR_n_Ph2.place(relx=0.65, rely=0.65, anchor=tk.CENTER)
        d_IP_OR_n_ph2.trace('w', up_Ph2_IP_label)
        NU_OR_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_OR_n_ph2,font=("Aral",10), state = "disabled")
        NU_OR_n_Ph2.place(relx=0.85, rely=0.65, anchor=tk.CENTER)
        d_NU_OR_n_ph2.trace('w', up_Ph2_NU_label)
        IP_IP_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_IP_n_ph2,font=("Aral",10), state = "disabled")
        IP_IP_n_Ph2.place(relx=0.65, rely=0.85, anchor=tk.CENTER)
        d_IP_IP_n_ph2.trace('w', up_Ph2_IP_label)
        NU_IP_n_Ph2 = tk.Spinbox(frame_Ph2, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_IP_n_ph2,font=("Aral",10), state = "disabled")
        NU_IP_n_Ph2.place(relx=0.85, rely=0.85, anchor=tk.CENTER)
        d_NU_IP_n_ph2.trace('w', up_Ph2_NU_label)
        
        Third_Phase_label = customtkinter.CTkLabel(master=frame_Ph3, text="Third Phase", font=("Roboto", 16, "bold"))
        Third_Phase_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        RG_n_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="RG", font=("Roboto", 14))
        RG_n_Ph3.place(relx=0.075, rely=0.45, anchor=tk.CENTER)
        OR_n_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="OR", font=("Roboto", 14))
        OR_n_Ph3.place(relx=0.075, rely=0.65, anchor=tk.CENTER)
        IP_n_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="IP", font=("Roboto", 14))
        IP_n_Ph3.place(relx=0.075, rely=0.85, anchor=tk.CENTER)
        RG_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="RG", font=("Roboto", 14))
        RG_Ph3.place(relx=0.25, rely=0.275, anchor=tk.CENTER)
        OR_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="OR", font=("Roboto", 14))
        OR_Ph3.place(relx=0.45, rely=0.275, anchor=tk.CENTER)
        IP_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="IP", font=("Roboto", 14))
        IP_Ph3.place(relx=0.65, rely=0.275, anchor=tk.CENTER)
        NU_Ph3 = customtkinter.CTkLabel(master=frame_Ph3, text="NU", font=("Roboto", 14))
        NU_Ph3.place(relx=0.85, rely=0.275, anchor=tk.CENTER)
        RG_RG_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_RG_RG_n_ph3,font=("Aral",10), state = "disabled")
        RG_RG_n_Ph3.place(relx=0.25, rely=0.45, anchor=tk.CENTER)
        d_RG_RG_n_ph3.trace('w', up_Ph3_RG_label)
        OR_RG_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_RG_n_ph3,font=("Aral",10), state = "disabled")
        OR_RG_n_Ph3.place(relx=0.45, rely=0.45, anchor=tk.CENTER)
        d_OR_RG_n_ph3.trace('w', up_Ph3_OR_label)
        IP_RG_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_RG_n_ph3,font=("Aral",10), state = "disabled")
        IP_RG_n_Ph3.place(relx=0.65, rely=0.45, anchor=tk.CENTER)
        d_IP_RG_n_ph3.trace('w', up_Ph3_IP_label)
        NU_RG_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_RG_n_ph3,font=("Aral",10), state = "disabled")
        NU_RG_n_Ph3.place(relx=0.85, rely=0.45, anchor=tk.CENTER)
        d_NU_RG_n_ph3.trace('w', up_Ph3_NU_label)
        OR_OR_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_OR_n_ph3,font=("Aral",10), state = "disabled")
        OR_OR_n_Ph3.place(relx=0.45, rely=0.65, anchor=tk.CENTER)
        d_OR_OR_n_ph3.trace('w', up_Ph3_OR_label)
        IP_OR_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_OR_n_ph3,font=("Aral",10), state = "disabled")
        IP_OR_n_Ph3.place(relx=0.65, rely=0.65, anchor=tk.CENTER)
        d_IP_OR_n_ph3.trace('w', up_Ph3_IP_label)
        NU_OR_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_OR_n_ph3,font=("Aral",10), state = "disabled")
        NU_OR_n_Ph3.place(relx=0.85, rely=0.65, anchor=tk.CENTER)
        d_NU_OR_n_ph3.trace('w', up_Ph3_NU_label)
        IP_IP_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_IP_n_ph3,font=("Aral",10), state = "disabled")
        IP_IP_n_Ph3.place(relx=0.65, rely=0.85, anchor=tk.CENTER)
        d_IP_IP_n_ph3.trace('w', up_Ph3_IP_label)
        NU_IP_n_Ph3 = tk.Spinbox(frame_Ph3, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_IP_n_ph3,font=("Aral",10), state = "disabled")
        NU_IP_n_Ph3.place(relx=0.85, rely=0.85, anchor=tk.CENTER)
        d_NU_IP_n_ph3.trace('w', up_Ph3_NU_label)
        
        Fourth_Phase_label = customtkinter.CTkLabel(master=frame_Ph4, text="Fourth Phase", font=("Roboto", 16, "bold"))
        Fourth_Phase_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        RG_n_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="RG", font=("Roboto", 14))
        RG_n_Ph4.place(relx=0.075, rely=0.45, anchor=tk.CENTER)
        OR_n_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="OR", font=("Roboto", 14))
        OR_n_Ph4.place(relx=0.075, rely=0.65, anchor=tk.CENTER)
        IP_n_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="IP", font=("Roboto", 14))
        IP_n_Ph4.place(relx=0.075, rely=0.85, anchor=tk.CENTER)
        RG_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="RG", font=("Roboto", 14))
        RG_Ph4.place(relx=0.25, rely=0.275, anchor=tk.CENTER)
        OR_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="OR", font=("Roboto", 14))
        OR_Ph4.place(relx=0.45, rely=0.275, anchor=tk.CENTER)
        IP_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="IP", font=("Roboto", 14))
        IP_Ph4.place(relx=0.65, rely=0.275, anchor=tk.CENTER)
        NU_Ph4 = customtkinter.CTkLabel(master=frame_Ph4, text="NU", font=("Roboto", 14))
        NU_Ph4.place(relx=0.85, rely=0.275, anchor=tk.CENTER)
        RG_RG_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_RG_RG_n_ph4,font=("Aral",10), state = "disabled")
        RG_RG_n_Ph4.place(relx=0.25, rely=0.45, anchor=tk.CENTER)
        d_RG_RG_n_ph4.trace('w', up_Ph4_RG_label)
        OR_RG_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_RG_n_ph4,font=("Aral",10), state = "disabled")
        OR_RG_n_Ph4.place(relx=0.45, rely=0.45, anchor=tk.CENTER)
        d_OR_RG_n_ph4.trace('w', up_Ph4_OR_label)
        IP_RG_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_RG_n_ph4,font=("Aral",10), state = "disabled")
        IP_RG_n_Ph4.place(relx=0.65, rely=0.45, anchor=tk.CENTER)
        d_IP_RG_n_ph4.trace('w', up_Ph4_IP_label)
        NU_RG_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_RG_n_ph4,font=("Aral",10), state = "disabled")
        NU_RG_n_Ph4.place(relx=0.85, rely=0.45, anchor=tk.CENTER)
        d_NU_RG_n_ph4.trace('w', up_Ph4_NU_label)
        OR_OR_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_OR_n_ph4,font=("Aral",10), state = "disabled")
        OR_OR_n_Ph4.place(relx=0.45, rely=0.65, anchor=tk.CENTER)
        d_OR_OR_n_ph4.trace('w', up_Ph4_OR_label)
        IP_OR_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_OR_n_ph4,font=("Aral",10), state = "disabled")
        IP_OR_n_Ph4.place(relx=0.65, rely=0.65, anchor=tk.CENTER)
        d_IP_OR_n_ph4.trace('w', up_Ph4_IP_label)
        NU_OR_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_OR_n_ph4,font=("Aral",10), state = "disabled")
        NU_OR_n_Ph4.place(relx=0.85, rely=0.65, anchor=tk.CENTER)
        d_NU_OR_n_ph4.trace('w', up_Ph4_NU_label)
        IP_IP_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_IP_n_ph4,font=("Aral",10), state = "disabled")
        IP_IP_n_Ph4.place(relx=0.65, rely=0.85, anchor=tk.CENTER)
        d_IP_IP_n_ph4.trace('w', up_Ph4_IP_label)
        NU_IP_n_Ph4 = tk.Spinbox(frame_Ph4, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_IP_n_ph4,font=("Aral",10), state = "disabled")
        NU_IP_n_Ph4.place(relx=0.85, rely=0.85, anchor=tk.CENTER)
        d_NU_IP_n_ph4.trace('w', up_Ph4_NU_label)

        Fifth_Phase_label = customtkinter.CTkLabel(master=frame_Ph5, text="Fifth Phase", font=("Roboto", 16, "bold"))
        Fifth_Phase_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        RG_n_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="RG", font=("Roboto", 14))
        RG_n_Ph5.place(relx=0.075, rely=0.45, anchor=tk.CENTER)
        OR_n_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="OR", font=("Roboto", 14))
        OR_n_Ph5.place(relx=0.075, rely=0.65, anchor=tk.CENTER)
        IP_n_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="IP", font=("Roboto", 14))
        IP_n_Ph5.place(relx=0.075, rely=0.85, anchor=tk.CENTER)
        RG_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="RG", font=("Roboto", 14))
        RG_Ph5.place(relx=0.25, rely=0.275, anchor=tk.CENTER)
        OR_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="OR", font=("Roboto", 14))
        OR_Ph5.place(relx=0.45, rely=0.275, anchor=tk.CENTER)
        IP_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="IP", font=("Roboto", 14))
        IP_Ph5.place(relx=0.65, rely=0.275, anchor=tk.CENTER)
        NU_Ph5 = customtkinter.CTkLabel(master=frame_Ph5, text="NU", font=("Roboto", 14))
        NU_Ph5.place(relx=0.85, rely=0.275, anchor=tk.CENTER)
        RG_RG_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_RG_RG_n_ph5,font=("Aral",10), state = "disabled")
        RG_RG_n_Ph5.place(relx=0.25, rely=0.45, anchor=tk.CENTER)
        d_RG_RG_n_ph5.trace('w', up_Ph5_RG_label)
        OR_RG_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_RG_n_ph5,font=("Aral",10), state = "disabled")
        OR_RG_n_Ph5.place(relx=0.45, rely=0.45, anchor=tk.CENTER)
        d_OR_RG_n_ph5.trace('w', up_Ph5_OR_label)
        IP_RG_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_RG_n_ph5,font=("Aral",10), state = "disabled")
        IP_RG_n_Ph5.place(relx=0.65, rely=0.45, anchor=tk.CENTER)
        d_IP_RG_n_ph5.trace('w', up_Ph5_IP_label)
        NU_RG_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_RG_n_ph5,font=("Aral",10), state = "disabled")
        NU_RG_n_Ph5.place(relx=0.85, rely=0.45, anchor=tk.CENTER)
        d_NU_RG_n_ph5.trace('w', up_Ph5_NU_label)
        OR_OR_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_OR_OR_n_ph5,font=("Aral",10), state = "disabled")
        OR_OR_n_Ph5.place(relx=0.45, rely=0.65, anchor=tk.CENTER)
        d_OR_OR_n_ph5.trace('w', up_Ph5_OR_label)
        IP_OR_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_OR_n_ph5,font=("Aral",10), state = "disabled")
        IP_OR_n_Ph5.place(relx=0.65, rely=0.65, anchor=tk.CENTER)
        d_IP_OR_n_ph5.trace('w', up_Ph5_IP_label)
        NU_OR_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_OR_n_ph5,font=("Aral",10), state = "disabled")
        NU_OR_n_Ph5.place(relx=0.85, rely=0.65, anchor=tk.CENTER)
        d_NU_OR_n_ph5.trace('w', up_Ph5_NU_label)
        IP_IP_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_IP_IP_n_ph5,font=("Aral",10), state = "disabled")
        IP_IP_n_Ph5.place(relx=0.65, rely=0.85, anchor=tk.CENTER)
        d_IP_IP_n_ph5.trace('w', up_Ph5_IP_label)
        NU_IP_n_Ph5 = tk.Spinbox(frame_Ph5, from_= 0, to = 4,width=4, increment=1,textvariable=d_NU_IP_n_ph5,font=("Aral",10), state = "disabled")
        NU_IP_n_Ph5.place(relx=0.85, rely=0.85, anchor=tk.CENTER)
        d_NU_IP_n_ph5.trace('w', up_Ph5_NU_label)
        
        update_button =  customtkinter.CTkButton(master=divison_root, text="Set \nvalues", width = 70 , height = 80  ,hover_color="darkgreen"  ,fg_color="green", command=Set_values)
        update_button.place(x=670, y=9, anchor='nw')
        counter_1 +=1
        
    else:
        messageWindow_2(1)

#====================================== open_division time window =============================
def open_division_time_window():
    global counter_2
    def first_trace(*args):
        first_entry.configure(text = '%.1f'%((0.3*first_value.get())+4)+" GW")
        second_value.set(first_value.get())
        
    def second_trace(*args):
        second_entry.configure(text = '%.1f'%((0.3*second_value.get())+4)+" GW")
        if  second_value.get()<first_value.get():
            messagebox.showerror("","The Second phase has to be after the First phase!")
            second_slider.set(first_value.get())
            second_value.set(first_value.get())
        third_value.set(second_value.get())

        
    def third_trace(*args):
        third_entry.configure(text = '%.1f'%((0.3*third_value.get())+4)+" GW")
        if  third_value.get()<second_value.get():
            messagebox.showerror("","The Third phase has to be after the Second phase!")
            third_slider.set(second_value.get())
            third_value.set(second_value.get())
        fourth_value.set(third_value.get())
        
    def fourth_trace(*args):
        fourth_entry.configure(text = '%.1f'%((0.3*fourth_value.get())+4)+" GW")
        if  fourth_value.get()<third_value.get():
            messagebox.showerror("","The Fourth phase has to be after the Third phase!")
            fourth_slider.set(third_value.get())
            fourth_value.set(third_value.get())
            
    def Set_values():
        def_first_phase.set(first_value.get())
        def_second_phase.set(second_value.get())
        def_third_phase.set(third_value.get())
        def_fourth_phase.set(fourth_value.get())
        divison_time_root.destroy()
    
    if counter_2 < 1:
        
        divison_time_root = Toplevel(root)
        x_position = 500+root.winfo_x()
        y_position = 150+root.winfo_y()
        divison_time_root.geometry(f"500x300+{x_position}+{y_position}")
        divison_time_root.title("Adjust phases timeline")
        
        Gw_light = Image.open('Images/GW.png')
        Gw_dark = Image.open('Images/GW_dark.png')
        first_value = tk.IntVar(divison_time_root, 0)
        second_value = tk.IntVar(divison_time_root, 0)
        third_value = tk.IntVar(divison_time_root, 0)
        fourth_value = tk.IntVar(divison_time_root, 0)

        GW_image = customtkinter.CTkImage(light_image= Gw_light, dark_image=Gw_dark, size=Gw_light.size)
        GW_label = customtkinter.CTkLabel(divison_time_root, image = GW_image, text="")
        GW_label.place(relx=0.815, rely=0.05, anchor='ne')
        _label = customtkinter.CTkLabel(master=divison_time_root, text="The beginning of ", font=("Roboto", 14, "bold")).place(relx=0.05, rely=0.15, anchor='nw')
        end_first_label = customtkinter.CTkLabel(master=divison_time_root, text="the second phase:", font=("Roboto", 14)).place(relx=0.05, rely=0.275, anchor='nw')
        end_second_label = customtkinter.CTkLabel(master=divison_time_root, text="the third phase:", font=("Roboto", 14)).place(relx=0.05, rely=0.425, anchor='nw')
        end_third_label = customtkinter.CTkLabel(master=divison_time_root, text="the fourth pahse:", font=("Roboto", 14)).place(relx=0.05, rely=0.575, anchor='nw')
        end_fourth_label = customtkinter.CTkLabel(master=divison_time_root, text="the fifth phase:", font=("Roboto", 14)).place(relx=0.05, rely=0.725, anchor='nw')
        first_slider = customtkinter.CTkSlider(master=divison_time_root, from_=0, to=147, width = 208 ,variable =first_value , number_of_steps = 147)
        first_slider.place(relx=0.8, rely=0.3, anchor='ne')
        second_slider = customtkinter.CTkSlider(master=divison_time_root, from_=0, to=147, width = 208  ,variable =second_value)
        second_slider.place(relx=0.8, rely=0.45, anchor='ne')
        third_slider = customtkinter.CTkSlider(master=divison_time_root, from_=0, to=147, width = 208  ,variable =third_value)
        third_slider.place(relx=0.8, rely=0.6, anchor='ne')
        fourth_slider = customtkinter.CTkSlider(master=divison_time_root, from_=0, to=147, width = 208  ,variable =fourth_value)
        fourth_slider.place(relx=0.8, rely=0.75, anchor='ne')
        first_entry = customtkinter.CTkLabel(master=divison_time_root, text= "4 GW", font=("Roboto", 14))
        first_entry.place(relx=0.85, rely=0.275, anchor='nw')
        second_entry = customtkinter.CTkLabel(master=divison_time_root, text="4 GW", font=("Roboto", 14))
        second_entry.place(relx=0.85, rely=0.425, anchor='nw')
        third_entry = customtkinter.CTkLabel(master=divison_time_root , text="4 GW", font=("Roboto", 14))
        third_entry.place(relx=0.85, rely=0.575, anchor='nw')
        fourth_entry = customtkinter.CTkLabel(master=divison_time_root,  text="4 GW", font=("Roboto", 14))
        fourth_entry.place(relx=0.85, rely=0.725, anchor='nw')
        first_value.trace('w', first_trace)
        second_value.trace('w', second_trace)
        third_value.trace('w', third_trace)
        fourth_value.trace('w', fourth_trace)
        
        update_button =  customtkinter.CTkButton(master=divison_time_root, text="Set values", width = 400  , height = 30  ,hover_color="darkgreen"  ,fg_color="green", command=Set_values)
        update_button.place(relx=0.5, rely=0.92, anchor=tk.CENTER)
        counter_2 +=1
        
    else:
        messageWindow_2(2)
        
#==================================== migration_window ========================================
def open_migration_window():
    global counter_3
    def Set_values():
        def_IP_migration.set(IP_migration.get())
        def_OR_migration.set(OR_migration.get())
        def_NU_migration.set(NU_migration.get())
        info_dis_2()
        migration_root.destroy()
        
    if counter_3 < 1:
        migration_root = Toplevel(root)
        x_position = 500+root.winfo_x()
        y_position = 200+root.winfo_y()
        migration_root.geometry(f"450x250+{x_position}+{y_position}")
        migration_root.title("Adjust cells migration speed")
        migration_root.bind('<FocusIn>', migration_speed_info)
        IP_migration = tk.StringVar(migration_root, "")
        OR_migration = tk.StringVar(migration_root, "")
        NU_migration = tk.StringVar(migration_root, "")
        OR_insert = Image.open('Images/OR.png')
        IP_insert = Image.open('Images/IP.png')
        NU_insert = Image.open('Images/NU.png')
        IP_image = customtkinter.CTkImage(light_image=IP_insert, dark_image=IP_insert, size=IP_insert.size)
        IP_label = customtkinter.CTkLabel(master=migration_root, image=IP_image,  text="").place(relx=0.075, rely=0.15, anchor=tk.CENTER)
        OR_image = customtkinter.CTkImage(light_image=OR_insert, dark_image=OR_insert, size=OR_insert.size)
        OR_label = customtkinter.CTkLabel(master=migration_root, image=OR_image,  text="").place(relx=0.075, rely=0.4, anchor=tk.CENTER)
        NU_image = customtkinter.CTkImage(light_image=NU_insert, dark_image=NU_insert, size=NU_insert.size)
        NU_label = customtkinter.CTkLabel(master=migration_root, image=NU_image,  text="").place(relx=0.075, rely=0.65, anchor=tk.CENTER)
        IP_migration_label = customtkinter.CTkLabel(master=migration_root,  text="Intermediate progenitor cell's translocation  speed:", font=("Roboto", 14)).place(relx=0.125, rely=0.1, anchor='nw')
        OR_migration_label = customtkinter.CTkLabel(master=migration_root,  text="Outer radial glial cell's translocation speed:", font=("Roboto", 14)).place(relx=0.125, rely=0.35, anchor='nw')
        NU_migration_label = customtkinter.CTkLabel(master=migration_root,  text="Neurons cell's migration speed:", font=("Roboto", 14)).place(relx=0.125, rely=0.6, anchor='nw')
        IP_migration_Entry = ttk.Entry(master=migration_root,font =("Aral",10) ,textvariable = IP_migration , width=5 ,style="IP_migration_style.TEntry")
        IP_migration_Entry.place(relx=0.875, rely=0.125, anchor='nw')
        OR_migration_Entry = ttk.Entry(master=migration_root,font =("Aral",10) ,textvariable = OR_migration , width=5 ,style="OR_migration_style.TEntry")
        OR_migration_Entry.place(relx=0.875, rely=0.375, anchor='nw')
        NU_migration_Entry = ttk.Entry(master=migration_root,font =("Aral",10) ,textvariable = NU_migration , width=5 ,style="NU_migration_style.TEntry")
        NU_migration_Entry.place(relx=0.875, rely=0.625, anchor='nw')
        update_button =  customtkinter.CTkButton(master=migration_root, text="Set values", width = 400  , height = 30  ,hover_color="darkgreen"  ,fg_color="green", command=Set_values)
        update_button.place(relx=0.5, rely=0.85, anchor=tk.CENTER)
        
        counter_3 +=1
        
    else:
        messageWindow_2(3)

        
#================================== diffusion window ====================================
def open_diffusion_window():
    global counter_4
    def Set_values():
        def_RG_diffusivity.set(RG_diffusivity.get())
        def_IP_diffusivity.set(IP_diffusivity.get())
        def_OR_diffusivity.set(OR_diffusivity.get())
        def_NU_diffusivity.set(NU_diffusivity.get())
        info_dis_2()
        diffusivity_root.destroy()
        
    if counter_4 < 1:
        diffusivity_root = Toplevel(root)
        x_position = 500+root.winfo_x()
        y_position = 200+root.winfo_y()
        diffusivity_root.geometry(f"450x300+{x_position}+{y_position}")
        diffusivity_root.title("Adjust cells diffusivity values")
        diffusivity_root.bind('<FocusIn>', diffusivity_info)
        RG_diffusivity = tk.StringVar(diffusivity_root, "")
        IP_diffusivity = tk.StringVar(diffusivity_root, "")
        OR_diffusivity = tk.StringVar(diffusivity_root, "")
        NU_diffusivity = tk.StringVar(diffusivity_root, "")
        RG_insert = Image.open('Images/RG.png')
        OR_insert = Image.open('Images/OR.png')
        IP_insert = Image.open('Images/IP.png')
        NU_insert = Image.open('Images/NU.png')
        RG_image = customtkinter.CTkImage(light_image=RG_insert, dark_image=RG_insert, size=RG_insert.size)
        RG_label = customtkinter.CTkLabel(master=diffusivity_root, image=RG_image,  text="").place(relx=0.075, rely=0.125, anchor=tk.CENTER)
        IP_image = customtkinter.CTkImage(light_image=IP_insert, dark_image=IP_insert, size=IP_insert.size)
        IP_label = customtkinter.CTkLabel(master=diffusivity_root, image=IP_image,  text="").place(relx=0.075, rely=0.325, anchor=tk.CENTER)
        OR_image = customtkinter.CTkImage(light_image=OR_insert, dark_image=OR_insert, size=OR_insert.size)
        OR_label = customtkinter.CTkLabel(master=diffusivity_root, image=OR_image,  text="").place(relx=0.075, rely=0.525, anchor=tk.CENTER)
        NU_image = customtkinter.CTkImage(light_image=NU_insert, dark_image=NU_insert, size=NU_insert.size)
        NU_label = customtkinter.CTkLabel(master=diffusivity_root, image=NU_image,  text="").place(relx=0.075, rely=0.725, anchor=tk.CENTER)
        RG_diffusivity_label = customtkinter.CTkLabel(master=diffusivity_root,  text="Radial glial cell's diffusivity value:", font=("Roboto", 14)).place(relx=0.125, rely=0.1, anchor='nw')
        IP_diffusivity_label = customtkinter.CTkLabel(master=diffusivity_root,  text="Intermediate progenitor cell's diffusivity value:", font=("Roboto", 14)).place(relx=0.125, rely=0.3, anchor='nw')
        OR_diffusivity_label = customtkinter.CTkLabel(master=diffusivity_root,  text="Outer radial glial cell's diffusivity value:", font=("Roboto", 14)).place(relx=0.125, rely=0.5, anchor='nw')
        NU_diffusivity_label = customtkinter.CTkLabel(master=diffusivity_root,  text="Neurons cell's diffusivity value:", font=("Roboto", 14)).place(relx=0.125, rely=0.7, anchor='nw')
        RG_diffusivity_Entry = ttk.Entry(master=diffusivity_root,font =("Aral",10) ,textvariable = RG_diffusivity , width=5 ,style="IP_diffusivity_style.TEntry")
        RG_diffusivity_Entry.place(relx=0.875, rely=0.125, anchor='nw')
        IP_diffusivity_Entry = ttk.Entry(master=diffusivity_root,font =("Aral",10) ,textvariable = IP_diffusivity , width=5 ,style="IP_diffusivity_style.TEntry")
        IP_diffusivity_Entry.place(relx=0.875, rely=0.325, anchor='nw')
        OR_diffusivity_Entry = ttk.Entry(master=diffusivity_root,font =("Aral",10) ,textvariable = OR_diffusivity , width=5 ,style="OR_diffusivity_style.TEntry")
        OR_diffusivity_Entry.place(relx=0.875, rely=0.525, anchor='nw')
        NU_diffusivity_Entry = ttk.Entry(master=diffusivity_root,font =("Aral",10) ,textvariable = NU_diffusivity , width=5 ,style="NU_diffusivity_style.TEntry")
        NU_diffusivity_Entry.place(relx=0.875, rely=0.725, anchor='nw')
        update_button =  customtkinter.CTkButton(master=diffusivity_root, text="Set values", width = 400  , height = 30  ,hover_color="darkgreen"  ,fg_color="green", command=Set_values)
        update_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        counter_4 +=1
        
    else:
        messageWindow_2(4)

    
#=================================== info functions ==============================================
def VZ_info(event):
    label_img2.configure(image=gemotry_image_vz)
    label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Ventricular zone raduis as a ratio to initial radius \nshould take a value between 0.2 and 0.4.")
    info_label.place(relx=0.4, rely=0.1, anchor='nw')
    
        
def SVZ_info(event):
    label_img2.configure(image=gemotry_image_isvz)
    label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Inner subventricular zone raduis as a ratio to initial radius \nshould take a value between ventricular zone raduis and 0.5.")
    info_label.place(relx=0.4, rely=0.1, anchor='nw')

        
def cr_thickness_info(event):
    label_img2.configure(image=gemotry_image_tc)
    label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Initial cortex thickness as a ratio to initial radius \nshould take a value between 0.01 and 0.35.")
    info_label.place(relx=0.4, rely=0.1, anchor='nw')



def intial_raduis_info(event):
    label_img2.configure(image=gemotry_image_R)
    label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Initial fetal brain radius at gestational week 5 in [mm].")
    info_label.place(relx=0.4, rely=0.1, anchor='nw')

def MST_factor_info(event):
    label_img2.configure(image=gemotry_image_mst)
    label_img2.place(relx=0.29, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Mitotic somal translocation factor of ORGCs \nshould take a value smaller than 0.1.")
    info_label.place(relx=0.5, rely=0.1, anchor='nw')



def ORG_variation_case_info(event):
    if (def_ORG_variation_case.get() == OSVZ_varying_options[0]):
        label_img2.configure(image=OSVZ_constant_image)
        label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
        info_label.configure(text="The variable controllers the regional variation of ORGCs. \nConstant mean no regional variation.")
        info_label.place(relx=0.4, rely=0.1, anchor='nw')
    if (def_ORG_variation_case.get() == OSVZ_varying_options[1]):
        label_img2.configure(image=OSVZ_Linear_gradient_image)
        label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
        info_label.configure(text="The variable controllers the regional variation of ORGCs. \nThe ORGCs division rate increases linearly between \nthe angle 0 and 90.")
        info_label.place(relx=0.4, rely=0.1, anchor='nw')
        label_img3.configure(image=OSVZ_Linear_gradient_curve_image)
        label_img3.place(relx=0.55, rely=0.7, anchor=tk.CENTER)

    if (def_ORG_variation_case.get() == OSVZ_varying_options[2]):
        label_img2.configure(image=OSVZ_Quadratic_gradient_image)
        label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
        info_label.configure(text="The variable controllers the regional variation of ORGCs. \nThe ORGCs division rate increases quadratically \nbetween the angle 0 and 90.")
        info_label.place(relx=0.4, rely=0.1, anchor='nw')
        label_img3.configure(image=OSVZ_Quadratic_gradient_curve_image)
        label_img3.place(relx=0.55, rely=0.7, anchor=tk.CENTER)
        
        
    if (def_ORG_variation_case.get() == OSVZ_varying_options[3]):
        label_img2.configure(image=OSVZ_Random1_image)
        label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
        info_label.configure(text="The variable controllers the regional variation of ORGCs. \nThe random variation of ORGCs division rate as shown in Figure.")
        info_label.place(relx=0.4, rely=0.1, anchor='nw')
        label_img3.configure(image=OSVZ_Random1_curve_image)
        label_img3.place(relx=0.55, rely=0.7, anchor=tk.CENTER)
        
    if (def_ORG_variation_case.get() == OSVZ_varying_options[4]):
        label_img2.configure(image=OSVZ_Random2_image)
        label_img2.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
        info_label.configure(text="The variable controllers the regional variation of ORGCs. \nThe random variation of ORGCs division rate as shown in Figure.")
        info_label.place(relx=0.4, rely=0.1, anchor='nw')
        label_img3.configure(image=OSVZ_Random2_curve_image)
        label_img3.place(relx=0.55, rely=0.7, anchor=tk.CENTER)


def intial_division_info(event):
    info_label.configure(text="Cell density intial value in ventricular zone in [1/(mm^2)].")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')


        
def migration_speed_info(event):
    label_img2.configure(image=diffusion_image_v)
    label_img2.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Cell migration speed in [mm/wk]. \nThe cells migrate along RGC fibers \ni.e. radial direction.")
    info_label.place(relx=0.6, rely=0.1, anchor='nw')


def diffusivity_info(event):
    label_img2.configure(image=diffusion_image_d)
    label_img2.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Diffusivity in cortex in [mm^2/wk]. \nIn this model, we consider isotropic diffusion \nwhich  means the diffusion is equal in all \ndirections.")
    info_label.place(relx=0.6, rely=0.1, anchor='nw')


        
def migration_threshold_info(event):
    label_img2.configure(image=diffusion_image_c0)
    label_img2.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Cell migration threshold in [1/mm^3].")
    info_label.place(relx=0.6, rely=0.1, anchor='nw')


def HV_exp_info(event):
    label_img2.configure(image=diffusion_image_gamma)
    label_img2.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Heaviside function exponent. \nFor a smooth solution, should take a value \nsmaller than 0.01.")
    info_label.place(relx=0.6, rely=0.1, anchor='nw')


        
def shear_modulus_info(event):
    label_img2.configure(image=strain_image_mu)
    label_img2.place(relx=0.35, rely=0.4, anchor=tk.CENTER)
    info_label.configure(text="The shear modulus of the cortical layer in [KPa]. The recommended value according to literature is 2.07 KPa.")
    info_label.place(relx=0.05, rely=0.8, anchor='nw')

    
def stiffness_ratio_info(event):
    label_img2.configure(image=strain_image_ratio)
    label_img2.place(relx=0.35, rely=0.4, anchor=tk.CENTER)
    info_label.configure(text="The ratio of stiffness between cortex and subcortex. The recommended values 3 and 5.")
    info_label.place(relx=0.05, rely=0.8, anchor='nw')

    
def poisson_ratio_info(event):
    label_img2.configure(image=strain_image_nu)
    label_img2.place(relx=0.35, rely=0.4, anchor=tk.CENTER)
    info_label.configure(text="The value of the Poisson's ratio, have to take a vlaue between 0.0 and 0.5.")
    info_label.place(relx=0.05, rely=0.75, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.85, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://en.wikipedia.org/wiki/Poisson%27s_ratio"))


    
def max_density_info(event):
    label_img2.configure(image=varying_image_cmax)
    label_img2.place(relx=0.35, rely=0.4, anchor=tk.CENTER)
    info_label.configure(text="The max cell density, after this value the stiffness becomes constant and equals to value set in \n(shear modulus of cortex). Here the c_min was set to 200, so the c_max have to be bigger than 200.")
    info_label.place(relx=0.05, rely=0.8, anchor='nw')

    
def stiffness_case_info(event):
    info_label.configure(text="The state of cortical stiffness is constant.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')

    
def stiffness_case_varying_info(event):
    label_img2.configure(image=varying_image)
    label_img2.place(relx=0.35, rely=0.4, anchor=tk.CENTER)
    info_label.configure(text="The state of cortical stiffness. Varying means exists a positive relation with cell density value.")
    info_label.place(relx=0.05, rely=0.8, anchor='nw')

    
def case_info_2d(event):
    label_img2.configure(image=two_d)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="")

    
def case_info_3d(event):
    label_img2.configure(image=three_d)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="")

    
def c_k_info(event):
    info_label.configure(text="c_k factor that satisfies CFL condition, this value matter only in case of Newton Raphson method not \nconverged. In this case, the solver repeat solving the not converged time-step with considering \na smaller time-step size according to the value of c_k.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.5, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition"))

    
    
def refinement_info(event):
    label_img2.configure(image=ref_2d)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="The number of mesh global \nrefinements. With increasing this \nvalue the mesh becomes finer \nand the solution smoother but \nwith a longer solving time. \nThe recommended value for the \n2D case is 3 and for the 3D \ncase is 2.")
    info_label.place(relx=0.7 , rely=0.1, anchor='nw')

    
def degree_info(event):
    info_label.configure(text="The shape function polynomial degree of the FE. The shape function is used to approximate the solution. \nwith increasing the degree value the shape functions become smoother and the solution should become \nmore accurate but that increases the solving time.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.4, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://en.wikipedia.org/wiki/Hp-FEM"))

    
def total_time_info(event):
    info_label.configure(text="Total run time. If you do not know the exact time, write 1000. Thus the solver \nwill automatically stop when it reaches to mechanical instability point.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')

    
def delt_t_info(event):
    info_label.configure(text="Time step size, should take a value smaller than 1.0.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')


def stability_con_info(event):
    info_label.configure(text="Stabilization constant Beta of advection-diffusion equation. \nIn this model, a numerical stabilization method is applied. \nThis value should be smaller than 0.1. The recommended value is 0.03334.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.4, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://www.dealii.org/current/doxygen/deal.II/step_31.html"))


def nonlinear_it_info(event):
    info_label.configure(text="Max number of nonlinear iterations allowed. \nHere the Newton-Raphson method is used to solve the nonlinear problem.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.3, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://en.wikipedia.org/wiki/Newton%27s_method"))

    
def tol_u_info(event):
    info_label.configure(text="Force residual error tolerance. The recommended value is 1.0e-8.\nThe smallest allowed value is 1.0e-4.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')

    
def tol_c_info(event):
    info_label.configure(text="Advection-diffusion residual error tolerance. The recommended value is 1.0e-8.\nThe smallest allowed value is 1.0e-4.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')


def update_u_info(event):
    info_label.configure(text="Displacement & cell-density update error tolerance. The recommended value is 1.0e-4.\nThe smallest allowed value is 1.0e-3.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')

    
def solver_type_info(event):
    info_label.configure(text="Type of solver used to solve the linear system. \nIn case of choosing CG solver you should enter the number of linear solver iterations.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')
    web_label.configure(text="For more details click here")
    web_label.place(relx=0.05, rely=0.3, anchor='nw')
    web_label.bind("<Button 1>", lambda e: callback("https://en.wikipedia.org/wiki/Conjugate_gradient_method"))

    
def linear_it_info(event):
    info_label.configure(text="The number of max iterations CG linear solver.")
    info_label.place(relx=0.05, rely=0.1, anchor='nw')

    
def k_growth_info(event):
    label_img2.configure(image=growth_image_ks)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Growth rate factor. This constant \ncoefficient controls the amount of \nisotropic growth in the subcortical \nlayer. To serve numerical stability \nrequirements, this factor has to \nbe smaller than 1.0e-3 for the 2D \ncase and 1.0e-4 for the 3D case.")
    info_label.place(relx=0.67, rely=0.1, anchor='nw')


def growth_ratio_info(event):
    label_img2.configure(image=growth_image_ratio)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Growth ratio. This ratio controls \nthe tangential and radial growth \namount in the cortical layer. With \nincreasing this value the growth \nvarying between tangential and \nradial growth increases. \nThe recommended value 1.5 and 3.")
    info_label.place(relx=0.67, rely=0.1, anchor='nw')

    
def growth_exp_info(event):
    label_img2.configure(image=growth_image_exp)
    label_img2.place(relx=0.35, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="Growth exponent.")
    info_label.place(relx=0.67, rely=0.1, anchor='nw')

def info_dis(event):
    label_img3.configure(image=non_image_2)
    label_img3.place(relx=0.99, rely=0.99, anchor=tk.CENTER)
    label_img2.configure(image=non_image)
    label_img2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="")
    info_label.place(relx=1, rely=1, anchor=tk.CENTER)
    web_label.configure(text="")
    web_label.place(relx=1, rely=1, anchor=tk.CENTER)
    
def info_dis_2():
    label_img3.configure(image=non_image_2)
    label_img3.place(relx=0.99, rely=0.99, anchor=tk.CENTER)
    label_img2.configure(image=non_image)
    label_img2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    info_label.configure(text="")
    info_label.place(relx=1, rely=1, anchor=tk.CENTER)
    web_label.configure(text="")
    web_label.place(relx=1, rely=1, anchor=tk.CENTER)

#============================================== Check values function =========================================
def check_vz_raduis(*args):
    if(def_vz_raduis.get() != ''):
        if (float(def_vz_raduis.get()) > 0.4) or (float(def_vz_raduis.get())< 0.2):
            style.configure("vz_style.TEntry",background="red2")
            Error_1.set(value= True)
        else:
            style.configure("vz_style.TEntry",background="systemTextBackgroundColor")
            Error_1.set(value = False)

def check_svz_raduis(*args):
    if(def_svz_raduis.get() != ''):
        if (float(def_svz_raduis.get()) > 0.5) or (float(def_svz_raduis.get())< float(vz_raduis.get())):
            style.configure("svz_style.TEntry",background="red2")
            Error_2.set(value= True)
        else:
            style.configure("svz_style.TEntry",background="systemTextBackgroundColor")
            Error_2.set(value= False)
        
def check_cr_thickness(*args):
    if(def_cr_thickness.get() != ''):
        if (float(def_cr_thickness.get()) < 0.01) or (float(def_cr_thickness.get())> 0.35):
            style.configure("cr_style.TEntry",background="red2")
            Error_3.set(value= True)
        else:
            style.configure("cr_style.TEntry",background="systemTextBackgroundColor")
            Error_3.set(value= False)
        
def check_MST_factor(*args):
    if(def_MST_factor.get() != ''):
        if (float(def_MST_factor.get()) > 0.1):
            style.configure("MST_style.TEntry",background="red2")
            Error_4.set(value= True)
        else:
            style.configure("MST_style.TEntry",background="systemTextBackgroundColor")
            Error_4.set(value= False)

def check_nonlinear_it(*args):
    if(def_nonlinear_it.get() != ''):
        if (float(def_nonlinear_it.get()) < 3):
            style.configure("nonlinear_style.TEntry",background="red2")
            Error_5.set(value= True)
        else:
            style.configure("nonlinear_style.TEntry",background="systemTextBackgroundColor")
            Error_5.set(value= False)
        
def check_tol_u(*args):
    if(def_tol_u.get() != ''):
        if (float(def_tol_u.get()) > 1.0e-4):
            style.configure("tol_u_style.TEntry",background="red2")
            Error_6.set(value= True)
        else:
            style.configure("tol_u_style.TEntry",background="systemTextBackgroundColor")
            Error_6.set(value= False)
        
def check_tol_c(*args):
    if(def_tol_c.get() != ''):
        if (float(def_tol_c.get()) > 1.0e-4):
            style.configure("tol_c_style.TEntry",background="red2")
            Error_7.set(value= True)
        else:
            style.configure("tol_c_style.TEntry",background="systemTextBackgroundColor")
            Error_7.set(value= False)
        
def check_tol_update(*args):
    if(def_update_u.get() != ''):
        if (float(def_update_u.get()) > 1.0e-3):
            style.configure("tol_update_style.TEntry",background="red2")
            Error_8.set(value= True)
        else:
            style.configure("tol_update_style.TEntry",background="systemTextBackgroundColor")
            Error_8.set(value= False)
        
def check_refinement(*args):
    if(def_refinement.get() != ''):
        try:
            int(def_refinement.get())
        except ValueError:
            style.configure("refinementf_style.TEntry",background="red2")
            Error_9.set(value= True)
        else:
            style.configure("refinementf_style.TEntry",background="systemTextBackgroundColor")
            Error_9.set(value= False)
        
def check_degree(*args):
    if(def_degree.get() != ''):
        try:
            integerdegree = int(def_degree.get())
        except ValueError:
            style.configure("degree_style.TEntry",background="red2")
            Error_10.set(value= True)
        else:
            if (integerdegree == 0):
                style.configure("degree_style.TEntry",background="red2")
                Error_10.set(value= True)
            else:
                style.configure("degree_style.TEntry",background="systemTextBackgroundColor")
                Error_10.set(value= False)
        

def check_delt_t(*args):
    if(def_delt_t.get() != ''):
        if (float(def_delt_t.get()) >= 1.0) or (float(def_delt_t.get()) == 0.0):
            style.configure("delt_t_style.TEntry",background="red2")
            Error_11.set(value= True)
        else:
            style.configure("delt_t_style.TEntry",background="systemTextBackgroundColor")
            Error_11.set(value= False)

def check_total_time(*args):
    if(def_total_time.get() != ''):
        try:
            integertime = int(def_total_time.get())
        except ValueError:
            style.configure("total_time_style.TEntry",background="red2")
            Error_12.set(value= True)
        else:
            if (integertime == 0):
                style.configure("total_time_style.TEntry",background="red2")
                Error_12.set(value= True)
            else:
                style.configure("total_time_style.TEntry",background="systemTextBackgroundColor")
                Error_12.set(value= False)
            
def check_poisson_ratio(*args):
    if(def_poisson_ratio.get() != ''):
        if (float(def_poisson_ratio.get()) >0.5) or (float(def_poisson_ratio.get())<0.0):
            style.configure("poisson_ratio_style.TEntry", background = "red2")
            Error_13.set(value= True)
        else:
            style.configure("poisson_ratio_style.TEntry",background="systemTextBackgroundColor")
            Error_13.set(value= False)
            
def cehck_max_density(*args):
    if(def_max_density.get() != ''):
        if (float(def_max_density.get()) <= 200):
            style.configure("max_density_style.TEntry", background = "red2")
            Error_14.set(value= True)
        else:
            style.configure("max_density_style.TEntry",background="systemTextBackgroundColor")
            Error_14.set(value= False)
            
def cehck_stability_con(*args):
    if(def_stability_con.get() != ''):
        if (float(def_stability_con.get()) > 0.1):
            style.configure("stability_con_style.TEntry", background = "red2")
            Error_14.set(value= True)
        else:
            style.configure("stability_con_style.TEntry",background="systemTextBackgroundColor")
            Error_14.set(value= False)
            
def check_c_k(*args):
    if(def_c_k.get() != ''):
        if (float(def_c_k.get()) > 1):
            style.configure("c_k_style.TEntry", background = "red2")
            Error_14.set(value= True)
        else:
            style.configure("c_k_style.TEntry",background="systemTextBackgroundColor")
            Error_14.set(value= False)
            
def check_k_growth(*args):
    if(def_k_growth.get() != ''):
        if (float(def_case.get())==2):
            if (float(def_k_growth.get()) > 1.0e-3):
                style.configure("k_growth_style.TEntry", background = "red2")
                Error_14.set(value= True)
            else:
                style.configure("k_growth_style.TEntry",background="systemTextBackgroundColor")
                Error_14.set(value= False)
        if (float(def_case.get())==3):
            if (float(def_k_growth.get()) > 1.0e-4):
                style.configure("k_growth_style.TEntry", background = "red2")
                Error_14.set(value= True)
            else:
                style.configure("k_growth_style.TEntry",background="systemTextBackgroundColor")
                Error_14.set(value= False)

#============================================== Gemotry frame ============================================
style = ttk.Style()

label_Gemotry = customtkinter.CTkLabel(master=frame_Gemotry, text="Geometry Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen"))
label_Gemotry.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

label_vz = customtkinter.CTkLabel(master=frame_Gemotry, text="Ventricular zone raduis:", font=("Roboto", 16)).place(relx=0.05, rely=0.225)

vz_raduis = ttk.Entry(master=frame_Gemotry,font =("Aral",10) ,textvariable = def_vz_raduis , width=15 ,style="vz_style.TEntry")
vz_raduis.place(relx=0.6, rely=0.225)
def_vz_raduis.trace('w', check_vz_raduis)
vz_raduis.bind('<FocusIn>', VZ_info)
vz_raduis.bind('<FocusOut>', info_dis)


label_svz = customtkinter.CTkLabel(master=frame_Gemotry, text="Subventricular zone raduis:", font=("Roboto", 16)).place(relx=0.05, rely=0.375)

svz_raduis = ttk.Entry(master=frame_Gemotry, font =("Aral",10), textvariable = def_svz_raduis ,width=15, style="svz_style.TEntry")
svz_raduis.place(relx=0.6, rely=0.375)
def_svz_raduis.trace('w', check_svz_raduis)
svz_raduis.bind('<FocusIn>', SVZ_info)
svz_raduis.bind('<FocusOut>', info_dis)

label_cth = customtkinter.CTkLabel(master=frame_Gemotry, text="Cortex thickness:", font=("Roboto", 16)).place(relx=0.05, rely=0.525)

cr_thickness = ttk.Entry(master=frame_Gemotry,font =("Aral",10) ,textvariable = def_cr_thickness ,width=15, style = "cr_style.TEntry")
cr_thickness.place(relx=0.6, rely=0.525)
def_cr_thickness.trace('w',check_cr_thickness)
cr_thickness.bind('<FocusIn>', cr_thickness_info)
cr_thickness.bind('<FocusOut>', info_dis)

label_raduis = customtkinter.CTkLabel(master=frame_Gemotry, text="Initial brain radius:", font=("Roboto", 16)).place(relx=0.05, rely=0.675)

intial_raduis = ttk.Entry(master=frame_Gemotry, font =("Aral",10), textvariable = def_intial_raduis ,width=15)
intial_raduis.place(relx=0.6, rely=0.675)
intial_raduis.bind('<FocusIn>', intial_raduis_info)
intial_raduis.bind('<FocusOut>', info_dis)

label_mst = customtkinter.CTkLabel(master=frame_Gemotry, text="Mitotic translocation factor:", font=("Roboto", 16)).place(relx=0.05, rely=0.825)

MST_factor = ttk.Entry(master=frame_Gemotry, font =("Aral",10),textvariable = def_MST_factor  ,width=15, style="MST_style.TEntry")
MST_factor.place(relx=0.6, rely=0.825)
def_MST_factor.trace('w',check_MST_factor)
MST_factor.bind('<FocusIn>', MST_factor_info)
MST_factor.bind('<FocusOut>', info_dis)

#=================================================== Diffusion frame ===========================================

label_diffusion = customtkinter.CTkLabel(master=frame_diffusion, text="Advection diffusion Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen")).place(relx=0.5, rely=0.08, anchor=tk.CENTER)

label_divsion_rate = customtkinter.CTkLabel(master=frame_diffusion, text="Cells division ratios:", font=("Roboto", 16)).place(relx=0.05, rely=0.18)

division_rate =  customtkinter.CTkButton(master=frame_diffusion, text="adjust values", width = 125 , height = 25  ,hover_color="gainsboro"  ,fg_color="white smoke", text_color= "black" , command=open_division_window)
division_rate.place(relx=0.6, rely=0.18)

label_divsion_time = customtkinter.CTkLabel(master=frame_diffusion, text="Phases timeline:", font=("Roboto", 16)).place(relx=0.05, rely=0.28)

division_time =  customtkinter.CTkButton(master=frame_diffusion, text="adjust values", width = 125 , height = 25  ,hover_color="gainsboro"  ,fg_color="white smoke",text_color= "black" ,command=open_division_time_window)
division_time.place(relx=0.6, rely=0.28)

label_ORG_variation = customtkinter.CTkLabel(master=frame_diffusion, text="Distribution of OSVZ proliferation:", font=("Roboto", 16)).place(relx=0.05, rely=0.38)

ORG_variation_case = ttk.Combobox(frame_diffusion , textvariable=def_ORG_variation_case ,values = OSVZ_varying_options, width = 11, state='readonly')
ORG_variation_case.place(relx=0.6, rely=0.38)
ORG_variation_case.bind('<FocusIn>', ORG_variation_case_info)
ORG_variation_case.bind('<FocusOut>', info_dis)

label_divsion = customtkinter.CTkLabel(master=frame_diffusion, text="Initial cell density value:", font=("Roboto", 16)).place(relx=0.05, rely=0.48)

intial_division = ttk.Entry(master=frame_diffusion, font =("Aral",10) ,textvariable = def_intial_division , width=15)
intial_division.place(relx=0.6, rely=0.48)
intial_division.bind('<FocusIn>', intial_division_info)
intial_division.bind('<FocusOut>', info_dis)

label_speed = customtkinter.CTkLabel(master=frame_diffusion, text="Cell migration speed:", font=("Roboto", 16)).place(relx=0.05, rely=0.58)

migration_speed = customtkinter.CTkButton(master=frame_diffusion, text="adjust values", width = 125 , height = 25  ,hover_color="gainsboro"  ,fg_color="white smoke", text_color= "black" ,command=open_migration_window)
migration_speed.place(relx=0.6, rely=0.58)


label_diff = customtkinter.CTkLabel(master=frame_diffusion, text="Diffusivity:", font=("Roboto", 16)).place(relx=0.05, rely=0.68)

diffusivity = customtkinter.CTkButton(master=frame_diffusion, text="adjust values", width = 125 , height = 25  ,hover_color="gainsboro"  ,fg_color="white smoke" , text_color= "black" ,command=open_diffusion_window)
diffusivity.place(relx=0.6, rely=0.68)

label_threshold = customtkinter.CTkLabel(master=frame_diffusion, text="Cell migration threshold:", font=("Roboto", 16)).place(relx=0.05, rely=0.78)

migration_threshold = ttk.Entry(master=frame_diffusion,font =("Aral",10), textvariable = def_migration_threshold  , width=15)
migration_threshold.place(relx=0.6, rely=0.78)
migration_threshold.bind('<FocusIn>', migration_threshold_info)
migration_threshold.bind('<FocusOut>', info_dis)

label_Hvexp = customtkinter.CTkLabel(master=frame_diffusion, text="Heaviside function exponent:", font=("Roboto", 16)).place(relx=0.05, rely=0.88)

HV_exp = ttk.Entry(master=frame_diffusion,font =("Aral",10) , textvariable = def_HV_exp ,width=15)
HV_exp.place(relx=0.6, rely=0.88)
HV_exp.bind('<FocusIn>', HV_exp_info)
HV_exp.bind('<FocusOut>', info_dis)

#=========================================== Stiffness frame ==============================================

label_stiffness = customtkinter.CTkLabel(master=frame_stiffness, text="Mechanical properties Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen")).place(relx=0.5, rely=0.1, anchor=tk.CENTER)

label_stiffness_case = customtkinter.CTkLabel(master=frame_stiffness, text="Cortical stiffness case:", font=("Roboto", 16)).place(relx=0.05, rely=0.25)

stiffness_varying_case = ttk.Radiobutton(master=frame_stiffness ,text= "Varying", variable = def_stiffness_case, value='Varying', command=enable_entry_cmax)
stiffness_constant_case = ttk.Radiobutton(master=frame_stiffness ,text= "Constant", variable = def_stiffness_case, value='Constant', command = disable_entry_cmax)
stiffness_varying_case.place(relx=0.6, rely=0.25)
stiffness_constant_case.place(relx=0.76, rely=0.25)
stiffness_varying_case.bind('<FocusIn>', stiffness_case_varying_info)
stiffness_constant_case.bind('<FocusIn>', stiffness_case_info)
stiffness_varying_case.bind('<FocusOut>', info_dis)
stiffness_constant_case.bind('<FocusOut>', info_dis)



label_shear_modulus = customtkinter.CTkLabel(master=frame_stiffness, text="Cortical shear modulus:", font=("Roboto", 16)).place(relx=0.05, rely=0.4)

shear_modulus = ttk.Entry(master=frame_stiffness ,font =("Aral",10) ,textvariable = def_shear_modulus  ,width=15)
shear_modulus.place(relx=0.6, rely=0.4)
shear_modulus.bind('<FocusIn>', shear_modulus_info)
shear_modulus.bind('<FocusOut>', info_dis)

label_stiffness_ratio = customtkinter.CTkLabel(master=frame_stiffness, text="Stiffness ratio:", font=("Roboto", 16)).place(relx=0.05, rely=0.55)

stiffness_ratio = ttk.Entry(master=frame_stiffness ,font =("Aral",10) ,textvariable = def_stiffness_ratio  ,width=15)
stiffness_ratio.place(relx=0.6, rely=0.55)
stiffness_ratio.bind('<FocusIn>', stiffness_ratio_info)
stiffness_ratio.bind('<FocusOut>', info_dis)

labe_poisson_ratio = customtkinter.CTkLabel(master=frame_stiffness, text="Poisson's ratio:", font=("Roboto", 16)).place(relx=0.05, rely=0.7)

poisson_ratio = ttk.Entry(master=frame_stiffness ,font =("Aral",10) ,textvariable = def_poisson_ratio  ,width=15, style="poisson_ratio_style.TEntry")
poisson_ratio.place(relx=0.6, rely=0.7)
def_poisson_ratio.trace('w', check_poisson_ratio)
poisson_ratio.bind('<FocusIn>', poisson_ratio_info)
poisson_ratio.bind('<FocusOut>', info_dis)

max_density_ratio = customtkinter.CTkLabel(master=frame_stiffness, text="Maximum cell density:", font=("Roboto", 16)).place(relx=0.05, rely=0.85)

max_density = ttk.Entry(master=frame_stiffness ,font =("Aral",10) ,textvariable = def_max_density ,width=15, style ="max_density_style.TEntry" ,state = "disabled")
max_density.place(relx=0.6, rely=0.85)
def_max_density.trace('w', cehck_max_density)
max_density.bind('<FocusIn>', max_density_info)
max_density.bind('<FocusOut>', info_dis)

#=================================================== mesh frame ===========================================

label_mesh = customtkinter.CTkLabel(master=frame_mesh, text="Discretization Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen")).place(relx=0.5, rely=0.08, anchor=tk.CENTER)

label_case = customtkinter.CTkLabel(master=frame_mesh, text="Geometry:", font=("Roboto", 16)).place(relx=0.05, rely=0.18)

d2_case = ttk.Radiobutton(master=frame_mesh ,text= "2D", variable = def_case, value='2')
d3_case = ttk.Radiobutton(master=frame_mesh ,text= "3D", variable = def_case, value='3')
d2_case.place(relx=0.6, rely=0.18)
d3_case.place(relx=0.75, rely=0.18)
d2_case.bind('<FocusIn>', case_info_2d)
d2_case.bind('<FocusOut>', info_dis)
d3_case.bind('<FocusIn>', case_info_3d)
d3_case.bind('<FocusOut>', info_dis)

label_refinement = customtkinter.CTkLabel(master=frame_mesh, text="Number global refinements:", font=("Roboto", 16)).place(relx=0.05, rely=0.295)

refinement = ttk.Entry(master=frame_mesh, font =("Aral",10) ,textvariable = def_refinement , width=15, style = "refinementf_style.TEntry")
refinement.place(relx=0.6, rely=0.295)
def_refinement.trace('w', check_refinement)
refinement.bind('<FocusIn>', refinement_info)
refinement.bind('<FocusOut>', info_dis)

label_degree = customtkinter.CTkLabel(master=frame_mesh, text="Polynomial degree:", font=("Roboto", 16)).place(relx=0.05, rely=0.41)

degree = ttk.Entry(master=frame_mesh,font =("Aral",10), textvariable = def_degree  , width=15, style = "degree_style.TEntry")
degree.place(relx=0.6, rely=0.41)
def_degree.trace('w', check_degree)
degree.bind('<FocusIn>', degree_info)
degree.bind('<FocusOut>', info_dis)

label_total_time = customtkinter.CTkLabel(master=frame_mesh, text="Total time:", font=("Roboto", 16)).place(relx=0.05, rely=0.525)

total_time = ttk.Entry(master=frame_mesh,font =("Aral",10), textvariable = def_total_time , width=15, style = "total_time_style.TEntry")
total_time.place(relx=0.6, rely=0.525)
def_total_time.trace('w', check_total_time)
total_time.bind('<FocusIn>', total_time_info)
total_time.bind('<FocusOut>', info_dis)

label_delt_t = customtkinter.CTkLabel(master=frame_mesh, text="Time step size:", font=("Roboto", 16)).place(relx=0.05, rely=0.64)

delt_t = ttk.Entry(master=frame_mesh,font =("Aral",10), textvariable = def_delt_t  , width=15, style = "delt_t_style.TEntry")
delt_t.place(relx=0.6, rely=0.64)
def_delt_t.trace('w', check_delt_t)
delt_t.bind('<FocusIn>', delt_t_info)
delt_t.bind('<FocusOut>', info_dis)

label_stability_con = customtkinter.CTkLabel(master=frame_mesh, text="Stabilization constant:", font=("Roboto", 16)).place(relx=0.05, rely=0.755)

stability_con = ttk.Entry(master=frame_mesh,font =("Aral",10) , textvariable = def_stability_con ,width=15, style = "stability_con_style.TEntry")
stability_con.place(relx=0.6, rely=0.755)
def_stability_con.trace('w', cehck_stability_con)
stability_con.bind('<FocusIn>', stability_con_info)
stability_con.bind('<FocusOut>', info_dis)

label_c_k= customtkinter.CTkLabel(master=frame_mesh, text="c_k factor:", font=("Roboto", 16)).place(relx=0.05, rely=0.87)

c_k = ttk.Entry(master=frame_mesh,font =("Aral",10) ,textvariable = def_c_k  , width=15, style="c_k_style.TEntry")
c_k.place(relx=0.6, rely=0.87)
def_c_k.trace('w', check_c_k)
c_k.bind('<FocusIn>', c_k_info)
c_k.bind('<FocusOut>', info_dis)

#============================================= Solver Frame ==================================================


label_solver = customtkinter.CTkLabel(master=frame_solver, text="Numerical solver Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen")).place(relx=0.5, rely=0.08, anchor=tk.CENTER)


label_newton = customtkinter.CTkLabel(master=frame_solver, text="Maximum Newton iterations:", font=("Roboto", 16)).place(relx=0.05, rely=0.21)

nonlinear_it = ttk.Entry(master=frame_solver,font =("Aral",10), textvariable = def_nonlinear_it  , width=15, style="nonlinear_style.TEntry")
nonlinear_it.place(relx=0.6, rely=0.21)
def_nonlinear_it.trace('w', check_nonlinear_it)
nonlinear_it.bind('<FocusIn>', nonlinear_it_info)
nonlinear_it.bind('<FocusOut>', info_dis)

label_tol_u = customtkinter.CTkLabel(master=frame_solver, text="Tolerance residual deformation:", font=("Roboto", 16)).place(relx=0.05, rely=0.34)

tol_u = ttk.Entry(master=frame_solver,font =("Aral",10), textvariable = def_tol_u  , width=15, style="tol_u_style.TEntry")
tol_u.place(relx=0.6, rely=0.34)
def_tol_u.trace('w', check_tol_u)
tol_u.bind('<FocusIn>', tol_u_info)
tol_u.bind('<FocusOut>', info_dis)

label_tol_c = customtkinter.CTkLabel(master=frame_solver, text="Tolerance residual cell density:", font=("Roboto", 16)).place(relx=0.05, rely=0.47)

tol_c = ttk.Entry(master=frame_solver,font =("Aral",10), textvariable = def_tol_c  , width=15, style="tol_c_style.TEntry")
tol_c.place(relx=0.6, rely=0.47)
def_tol_c.trace('w', check_tol_c)
tol_c.bind('<FocusIn>', tol_c_info)
tol_c.bind('<FocusOut>', info_dis)

label_update_u = customtkinter.CTkLabel(master=frame_solver, text="Tolerance update:", font=("Roboto", 16)).place(relx=0.05, rely=0.6)

update_u = ttk.Entry(master=frame_solver,font =("Aral",10), textvariable = def_update_u  , width=15, style="tol_update_style.TEntry")
update_u.place(relx=0.6, rely=0.6)
def_update_u.trace('w', check_tol_update)
update_u.bind('<FocusIn>', update_u_info)
update_u.bind('<FocusOut>', info_dis)

label_solver_type = customtkinter.CTkLabel(master=frame_solver, text="Linear solver type:", font=("Roboto", 16)).place(relx=0.05, rely=0.73)

solver_type_direct= ttk.Radiobutton(master=frame_solver ,text= "Direct", variable = def_solver_type, value='Direct', command=disable_entry_it)
solver_type_cg = ttk.Radiobutton(master=frame_solver ,text= "CG", variable = def_solver_type, value='CG', command=enable_entry_it)
solver_type_direct.place(relx=0.6, rely=0.73)
solver_type_cg.place(relx=0.75, rely=0.73)
solver_type_direct.bind('<FocusIn>', solver_type_info)
solver_type_direct.bind('<FocusOut>', info_dis)
solver_type_cg.bind('<FocusIn>', solver_type_info)
solver_type_cg.bind('<FocusOut>', info_dis)

label_linear_it = customtkinter.CTkLabel(master=frame_solver, text="Iterations linear solver:", font=("Roboto", 16)).place(relx=0.05, rely=0.86)

linear_it = ttk.Entry(master=frame_solver,font =("Aral",10), textvariable = def_linear_it  , width=15, state = "disabled")
linear_it.place(relx=0.6, rely=0.86)
linear_it.bind('<FocusIn>', linear_it_info)
linear_it.bind('<FocusOut>', info_dis)

#=============================================== Growth Frame =================================================
label_growth = customtkinter.CTkLabel(master=fram_growth, text="Growth Parameters", font=("Roboto", 20, "bold"), text_color=("darkgreen")).place(relx=0.5, rely=0.12, anchor=tk.CENTER)

label_k_growth = customtkinter.CTkLabel(master=fram_growth, text="Growth rate:", font=("Roboto", 16)).place(relx=0.05, rely=0.32)

k_growth = ttk.Entry(master=fram_growth,font =("Aral",10), textvariable = def_k_growth  , width=15, style="k_growth_style.TEntry")
k_growth.place(relx=0.6, rely=0.32)
def_k_growth.trace('w', check_k_growth)
k_growth.bind('<FocusIn>', k_growth_info)
k_growth.bind('<FocusOut>', info_dis)

label_growth_ratio = customtkinter.CTkLabel(master=fram_growth, text="Growth ratio:", font=("Roboto", 16)).place(relx=0.05, rely=0.52)

growth_ratio = ttk.Entry(master=fram_growth,font =("Aral",10), textvariable = def_growth_ratio , width=15)
growth_ratio.place(relx=0.6, rely=0.52)
growth_ratio.bind('<FocusIn>', growth_ratio_info)
growth_ratio.bind('<FocusOut>', info_dis)

label_growth_exp = customtkinter.CTkLabel(master=fram_growth, text="Growth exponent:", font=("Roboto", 16)).place(relx=0.05, rely=0.72)

growth_exp = ttk.Entry(master=fram_growth,font =("Aral",10), textvariable = def_growth_exp  , width=15)
growth_exp.place(relx=0.6, rely=0.72)
growth_exp.bind('<FocusIn>', growth_exp_info)
growth_exp.bind('<FocusOut>', info_dis)

#========================================= Buttons==============================================================

run = customtkinter.CTkButton(master=root, text="Run", width = 140 , hover_color="darkgreen"  ,fg_color="green", command=update_parameters )
run.place(x=1335, y=700, anchor=tk.CENTER)

default = customtkinter.CTkButton(master=root, text="Default values",width = 140 ,command=set_default_values )
default.place(x=1335, y=662.5,anchor=tk.CENTER)

About_pro = customtkinter.CTkButton(master=root, text="About programm",width = 140 ,command=About_programm)
About_pro.place(x=1335, y=550,anchor=tk.CENTER)

About_au = customtkinter.CTkButton(master=root, text="About author",width = 140 ,command=About_author)
About_au.place(x=1335, y=587.5,anchor=tk.CENTER)

copyrig = customtkinter.CTkButton(master=root, text="Copyright",width = 140 ,command=Copy_right)
copyrig.place(x=1335, y=625,anchor=tk.CENTER)

info_label = customtkinter.CTkLabel(master=photo_info_fram, text="", font=("Roboto", 14), justify=LEFT)
web_label = customtkinter.CTkLabel(master=photo_info_fram, text="", text_color=('blue'), font=("Roboto", 14), justify=LEFT)


root.mainloop()
