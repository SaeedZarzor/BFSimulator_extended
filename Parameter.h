#ifndef PARAMETER_H
#define PARAMETER_H

#include <deal.II/base/parameter_handler.h>

/*! A namespace to group some functions to declare
 * and parse Parameters into instances of
 * ParameterHandler - READ Tutorial-19
 */



using namespace dealii;

namespace Parameter
{

struct GeneralParameters
{
	
	GeneralParameters(const std::string &input_filename);
	std::string output_file_name;
    std::string solver_type;
    std::string stiffness_case;
    std::vector<double> zones_raduis;
    std::vector<double> migration_speed;
    std::vector<double> diffusivity;
    std::vector<double> phase_days;
    std::vector<std::vector<int> > phase_ratio;
	double tolerance_residual_u;
    double tolerance_residual_c;
	unsigned int global_refinements;
    double cortex_thickness;
    double ventricular_raduis;
	double subventricular_raduis;
    double outer_subventricular_raduis;
	double Poisson;
	double total_time;
    double time_step;
	unsigned int degree;
    double scale;
    unsigned int max_number_newton_iterations;
    unsigned int multiplier_max_iterations_linear_solver;
    double growth_rate;
    double growth_ratio;
    double initial_radius;
    double cell_dvision_rate_v;
	double cell_dvision_rate_ovz;
	double dvision_value;
    double cell_migration_threshold;
    double exponent;
    double growth_exponent;
    double tol_u;
    double damention_ratio;
    double shear_modulud_cortex;
	double stiffness_ratio;
	double max_cell_density;
	double MST_factor;
    double Betta;
	double c_k;

	
	
	void declare_parameters(ParameterHandler &prm);
	void parse_parameters(ParameterHandler &prm);
};

GeneralParameters::GeneralParameters(const std::string &input_filename):
zones_raduis(4,0), migration_speed(4, 0), diffusivity(4, 0), phase_days(4,0), phase_ratio(6, std::vector<int>(5, 0))
{
	ParameterHandler prm;
	declare_parameters(prm);
	prm.parse_input(input_filename);
	parse_parameters(prm);
}

void GeneralParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection ("General");
	{
		prm.declare_entry ("Output file name", "output",
							Patterns::Anything(),
							"The name of the output file to be generated");
        
		prm.declare_entry ("Poly degree","1",
						   Patterns::Integer(),
				"The polynomial degree of the FE");
        
		prm.declare_entry ("Number global refinements","1",
						   Patterns::Integer(),
				 "The number of mesh global refinements");
        
        prm.declare_entry ("Grid scale","1",
						   Patterns::Double(),
				 "Grid scale");
        
        prm.declare_entry ("Initial radius","0.5",
						   Patterns::Double(),
					"Initial radius [mm]");

        prm.declare_entry ("Cortex thickness","0.1",
						   Patterns::Double(),
					"Initial cortex thickness as a ratio to initial radius (value between 0 and 1)");
        
        prm.declare_entry ("Ventricular zone raduis","0.25",
						   Patterns::Double(),
					"Ventricular zone raduis as a ratio to initial radius (value between 0.2 and 0.6)");

		prm.declare_entry ("Subventricular zone raduis","0.35",
						   Patterns::Double(),
					"Subventricular zone raduis as a ratio to initial radius (value between ventricular zone raduis and 0.5)");
        
        prm.declare_entry ("Outer subventricular zone raduis","0.5",
                           Patterns::Double(),
                    "Outer subventricular zone raduis as a ratio to initial radius (value between Subventricular ventricular zone raduis and 0.6)");

	 	prm.declare_entry ("Mitotic somal translocation factor","0.05",
						   Patterns::Double(),
					"Mitotic somal translocation factor (value <= 0.1)");

		prm.declare_entry ("Total time","1",
						   Patterns::Double(),
						   "Total time");
        
        prm.declare_entry ("Multiplier max iterations linear solver","10",
						   Patterns::Integer(),
						   "Multiplier max iterations linear solver");
        
        prm.declare_entry ("Max number newton iterations","10",
						   Patterns::Integer(),
						   "Max number newton iterations");
        prm.declare_entry ("Time step size","0.1",
						   Patterns::Double(),
						   "Time step size");
        
		prm.declare_entry ("Tolerance residual deformation","1e-5",
						   Patterns::Double(),
						   "The tolerance wrt the normalised residual deformation");
        
        prm.declare_entry ("Tolerance residual diffusion","1e-6",
						   Patterns::Double(),
						   "The tolerance wrt the normalised residual diffusion");
        
        prm.declare_entry ("Tolerance update","1e-6",
						   Patterns::Double(),
						   "The tolerance wrt the normalised residual norm");
        
        prm.declare_entry("The state of the stiffness","Varying",
                          Patterns::Anything(),
                          "The state of the stiffness Constant or Varying");
        
		prm.declare_entry ("Poisson's ratio","0.38",
						   Patterns::Double(),
						   "The value of the Poisson's ratio");
        
        prm.declare_entry ("The shear modulus of conrtex","2.07",
						   Patterns::Double(),
						   "The shear modulus of conrtex layer [KPa]");
        
		prm.declare_entry ("The ratio of stiffness","3",
						   Patterns::Double(),
						   "The ratio of stiffness between cortex and subcortex  mu_cmax/mu_smax");
        
		prm.declare_entry ("The max cell density","700",
						   Patterns::Double(),
						   "The max cell density where the stiffness still constant c_max");
        
		prm.declare_entry ("Growth rate","4.7e-4",
						   Patterns::Double(),
					"Growth rate k_s mm^2");
        
        prm.declare_entry ("Growth exponent","1.65",
						   Patterns::Double(),
					"Growth exponent alpha ");
        
        prm.declare_entry ("Growth ratio","1.5",
						   Patterns::Double(),
					"Growth ratio bitta_k");

        prm.declare_entry ("Cell dvision intial value","5",
						   Patterns::Double(),
					"Cell dvision intial value [1/(mm^2)]");
        
        prm.declare_entry ("IP cell migration speed","0.25",
 						   Patterns::Double(),
					"Intermediate progenitor cell migration speed  [mm/wk]");
        
        prm.declare_entry ("ORG cell migration speed","5",
                            Patterns::Double(),
                    "Outer radial glia cell migration speed  [mm/wk]");
        
        prm.declare_entry ("NU cell migration speed","2.5",
                            Patterns::Double(),
                    "Neurons cell migration speed  [mm/wk]");
        
        prm.declare_entry ("RG diffusivity","0.1",
 						   Patterns::Double(),
					"Radial glia cells diffusivity  d^cc [mm^2/wk]");
        
        prm.declare_entry ("IP diffusivity","0.1",
                            Patterns::Double(),
                    "Intermediate progenitor cells diffusivity  d^cc [mm^2/wk]");
        
        prm.declare_entry ("ORG diffusivity","0.1",
                            Patterns::Double(),
                    "Outer radial glia diffusivity  d^cc [mm^2/wk]");
        
        prm.declare_entry ("NU diffusivity","0.1",
                            Patterns::Double(),
                    "Neurons diffusivity  d^cc [mm^2/wk]");
        
        prm.declare_entry ("Cell migration threshold","400",
 						   Patterns::Double(),
					"Cell migration threshold  c_0 [1/mm^3]");
        
        prm.declare_entry ("Heaviside function exponent","0.008",
 						   Patterns::Double(),
					"Heaviside function exponent  gamma");
        
        prm.declare_entry("First phase", "30",
                          Patterns::Double(),
                          "On which gestational day the first division phase end?");
        
        prm.declare_entry("Second phase", "60",
                          Patterns::Double(),
                          "On which gestational day the second division phase end?");
        
        prm.declare_entry("Third phase", "84",
                          Patterns::Double(),
                          "On which gestational day the third division phase end?");
        
        prm.declare_entry("Fourth Phase", "120",
                          Patterns::Double(),
                          "On which gestational day the fourth division phase end?");
        
        prm.declare_entry ("Linear solver type", "CG",
					      Patterns::Anything(),
						"Linear solver type (CG or Direct)");
        
        prm.declare_entry ("Stabilization constant", "0.017",
                           Patterns::Double(),
                           "Stabilization constant Betta");
        
        prm.declare_entry ("c_k factor", "0.25",
                           Patterns::Double(),
                           "c_k factor (CFL condition)");
        
        prm.declare_entry ("RG/RG_n P1", "1", Patterns::Integer(0,4));
        prm.declare_entry ("RG/RG_n P2", "0", Patterns::Integer(0,4));
        prm.declare_entry ("RG/RG_n P3", "0", Patterns::Integer(0,4));
        prm.declare_entry ("RG/RG_n P4", "0", Patterns::Integer(0,4));
        prm.declare_entry ("RG/RG_n P5", "0", Patterns::Integer(0,4));
        
        prm.declare_entry ("IP/Rg_n P1", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/Rg_n P2", "1", Patterns::Integer(0,4));
        prm.declare_entry ("IP/Rg_n P3", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/Rg_n P4", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/Rg_n P5", "0", Patterns::Integer(0,4));

        prm.declare_entry ("IP/OR_n P1", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/OR_n P2", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/OR_n P3", "1", Patterns::Integer(0,4));
        prm.declare_entry ("IP/OR_n P4", "1", Patterns::Integer(0,4));
        prm.declare_entry ("IP/OR_n P5", "0", Patterns::Integer(0,4));
        
        prm.declare_entry ("IP/IP_n P1", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/IP_n P2", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/IP_n P3", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/IP_n P4", "0", Patterns::Integer(0,4));
        prm.declare_entry ("IP/IP_n P1", "1", Patterns::Integer(0,4));

        prm.declare_entry ("OR/RG_n P1", "0", Patterns::Integer(0,4));
        prm.declare_entry ("OR/RG_n P2", "0", Patterns::Integer(0,4));
        prm.declare_entry ("OR/RG_n P3", "1", Patterns::Integer(0,4));
        prm.declare_entry ("OR/RG_n P4", "0", Patterns::Integer(0,4));
        prm.declare_entry ("OR/RG_n P5", "0", Patterns::Integer(0,4));

        prm.declare_entry ("NU/IP_n P1", "0", Patterns::Integer(0,4));
        prm.declare_entry ("NU/IP_n P2", "0", Patterns::Integer(0,4));
        prm.declare_entry ("NU/IP_n P3", "2", Patterns::Integer(0,4));
        prm.declare_entry ("NU/IP_n P4", "2", Patterns::Integer(0,4));
        prm.declare_entry ("NU/IP_n P5", "4", Patterns::Integer(0,4));
        
	}
	prm.leave_subsection ();
}

void GeneralParameters::parse_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("General");
    {
        stiffness_case = prm.get("The state of the stiffness");
        tolerance_residual_u=prm.get_double("Tolerance residual deformation");
        tolerance_residual_c=prm.get_double("Tolerance residual diffusion");
        global_refinements =prm.get_integer("Number global refinements");
        Poisson=prm.get_double("Poisson's ratio");
        shear_modulud_cortex=prm.get_double("The shear modulus of conrtex");
        stiffness_ratio=prm.get_double("The ratio of stiffness");
        max_cell_density=prm.get_double("The max cell density");
        total_time = prm.get_double("Total time");
        time_step = prm.get_double("Time step size");
        output_file_name = prm.get("Output file name");
        degree = prm.get_integer("Poly degree");
        scale = prm.get_double("Grid scale");
        max_number_newton_iterations = prm.get_integer("Max number newton iterations");
        multiplier_max_iterations_linear_solver = prm.get_integer("Multiplier max iterations linear solver");
        growth_rate = prm.get_double("Growth rate");
        growth_ratio = prm.get_double("Growth ratio");
        initial_radius = prm.get_double("Initial radius");
        cortex_thickness = prm.get_double("Cortex thickness");
        zones_raduis[0] = prm.get_double("Ventricular zone raduis");
        zones_raduis[1] = prm.get_double("Subventricular zone raduis");
        zones_raduis[2] = prm.get_double("Outer subventricular zone raduis");
        MST_factor = prm.get_double("Mitotic somal translocation factor");
        cell_dvision_rate_v = prm.get_double("Cell dvision rate of RGCs");
        cell_dvision_rate_ovz = prm.get_double("Cell dvision rate of Outer RGCs");
        dvision_value = prm.get_double("Cell dvision intial value");
        migration_speed[1] = prm.get_double("IP Cell migration speed");
        migration_speed[2] = prm.get_double("ORG Cell migration speed");
        migration_speed[3] = prm.get_double("NU Cell migration speed");
        diffusivity[0] = prm.get_double("RG diffusivity");
        diffusivity[1] = prm.get_double("IP diffusivity");
        diffusivity[2] = prm.get_double("ORG diffusivity");
        diffusivity[3] = prm.get_double("NU diffusivity");
        cell_migration_threshold = prm.get_double("Cell migration threshold");
        exponent = prm.get_double("Heaviside function exponent");
        growth_exponent = prm.get_double("Growth exponent");
        solver_type = prm.get("Linear solver type");
        tol_u = prm.get_double("Tolerance update");
        damention_ratio = prm.get_double("Damention ratio");
        Betta = prm.get_double("Stabilization constant");
        c_k = prm.get_double("c_k factor");
        phase_days[0] = prm.get_double("First Phase");
        phase_days[1] = prm.get_double("Second Phase");
        phase_days[2] = prm.get_double("Third Phase");
        phase_days[3] = prm.get_double("Fourth Phase");
        phase_ratio[0][0] = prm.get_integer("RG/RG_n P1");
        phase_ratio[0][1] = prm.get_integer("RG/RG_n P2");
        phase_ratio[0][2] = prm.get_integer("RG/RG_n P3");
        phase_ratio[0][3] = prm.get_integer("RG/RG_n P4");
        phase_ratio[0][4] = prm.get_integer("RG/RG_n P5");
        phase_ratio[1][0] = prm.get_integer("IP/RG_n P1");
        phase_ratio[1][1] = prm.get_integer("IP/RG_n P2");
        phase_ratio[1][2] = prm.get_integer("IP/RG_n P3");
        phase_ratio[1][3] = prm.get_integer("IP/RG_n P4");
        phase_ratio[1][4] = prm.get_integer("IP/RG_n P5");
        phase_ratio[2][0] = prm.get_integer("IP/OR_n P1");
        phase_ratio[2][1] = prm.get_integer("IP/OR_n P2");
        phase_ratio[2][2] = prm.get_integer("IP/OR_n P3");
        phase_ratio[2][3] = prm.get_integer("IP/OR_n P4");
        phase_ratio[2][4] = prm.get_integer("IP/OR_n P5");
        phase_ratio[3][0] = prm.get_integer("IP/IP_n P1");
        phase_ratio[3][1] = prm.get_integer("IP/IP_n P2");
        phase_ratio[3][2] = prm.get_integer("IP/IP_n P3");
        phase_ratio[3][3] = prm.get_integer("IP/IP_n P4");
        phase_ratio[3][4] = prm.get_integer("IP/IP_n P5");
        phase_ratio[4][0] = prm.get_integer("OR/RG_n P1");
        phase_ratio[4][1] = prm.get_integer("OR/RG_n P2");
        phase_ratio[4][2] = prm.get_integer("OR/RG_n P3");
        phase_ratio[4][3] = prm.get_integer("OR/RG_n P4");
        phase_ratio[4][4] = prm.get_integer("OR/RG_n P5");
        phase_ratio[5][0] = prm.get_integer("NU/IP_n P1");
        phase_ratio[5][1] = prm.get_integer("NU/IP_n P2");
        phase_ratio[5][2] = prm.get_integer("NU/IP_n P3");
        phase_ratio[5][3] = prm.get_integer("NU/IP_n P4");
        phase_ratio[5][4] = prm.get_integer("NU/IP_n P5");

    }
    
    prm.leave_subsection();
    
    Assert((zones_raduis[0] >= 0.2)||(zones_raduis[0] <= zones_raduis[1]),
           ExcMessage("The Ventricular zone raduis must be biger than 0.2 and smaller than Subventricular zone raduis"));
    
    Assert((zones_raduis[1] >= zones_raduis[0] )||(zones_raduis[1] <= zones_raduis[2]),
           ExcMessage("The Subventricular zone raduis must be biger than Ventricular zone raduis and smaller than Outer subventricular zone raduis"));
    
    Assert((zones_raduis[2] >= zones_raduis[1] )||(zones_raduis[2] <= (1-cortex_thickness)),
           ExcMessage("The Outer subventricular zone raduis must be biger than Subventricular zone raduis and smaller than (1 - cortex thickness)"));
    
    for (unsigned int i = 0; i < 3; ++i)
      zones_raduis[i] *= initial_radius;
    
      zones_raduis[3] = (1-cortex_thickness) * initial_radius;
    
}


}//END namespace
#endif
