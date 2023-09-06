#ifndef POINTHISTORY_H
#define POINTHISTORY_H


#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/physics/elasticity/standard_tensors.h>		
#include "NeoHookeanMaterial.h"
#include "NonStandardTensors.h"
#include "CellDensity.h"
#include "Parameter.h"
#include "Growth.h"
#include <iostream>


using namespace dealii;


template <int dim>
class PointHistory
{
public:
	PointHistory():
	     sources(4, 0), dcc_r(4, 0), old_velocity_values(4, Tensor<1, dim>()), old_old_velocity_values(4, Tensor<1, dim>()), 
	     F(Physics::Elasticity::StandardTensors< dim >::I),
       	     inv_F_g(Physics::Elasticity::StandardTensors< dim >::I),
	     J(1.0) {}

	~PointHistory(){
	       delete material;
	       material = NULL;

               delete growth;
               growth= NULL;

               delete density;
               density= NULL;
             
                       }


	   void setup_lqp (const Parameter::GeneralParameters &parameter, const Point<dim> & position, const double &delta_t)
	    {
             
          p = position;
//          a = parameter.source_deceleration_factor;
           
//           std::vector<std::vector<int> > phase_ratio{{1,0,0,0,0},  // RG/RG_n
//                                                         {0,1,0,0,0},  // IP/Rg_n
//                                                         {0,0,1,1,0},  // IP/OR_n
//                                                         {0,0,0,0,1},  // IP/IP_n
//                                                         {0,0,1,0,0},  // OR/RG_n
//                                                         {0,0,2,2,4}}; // NU/IP_n


              material = new NeoHookeanMaterial<dim>(parameter.stiffness_case, parameter.shear_modulud_cortex, parameter.Poisson, parameter.stiffness_ratio, parameter.max_cell_density, parameter.zones_raduis[3]);

              growth   = new Growth<dim>(parameter.growth_rate, parameter.growth_ratio, parameter.growth_exponent, parameter.zones_raduis[3]);

              density  = new CellDensity<dim>(parameter.cell_migration_threshold, parameter.exponent, parameter.MST_factor, parameter.migration_speed, parameter.diffusivity, parameter.zones_raduis,parameter.phase_days, parameter.phase_ratio);

                
                 update_values (Tensor<2, dim>(), std::vector<Tensor<1, dim> >(4, Tensor<1, dim>()), std::vector<double>(4, 0) , Tensor<2, dim>(), Tensor<2, dim>(), std::vector<double>(4, 0), std::vector<double>(4, 0), 0 , delta_t,  false);
	    }

          void update_values (const Tensor<2, dim> &Grad_u, const std::vector<Tensor<1, dim> > &Grad_c, const std::vector<double> &c, const Tensor<2, dim> &Grad_u_n ,
                                const Tensor<2, dim> &Grad_u_n_1,const std::vector<double> &c_n, const std::vector<double>  &c_n_1,
                                const double &t, const double &d_t, bool update)
    {

           F = (Physics::Elasticity::StandardTensors< dim >::I + Grad_u);
	       Tensor<2, dim> F_n   = (Physics::Elasticity::StandardTensors< dim >::I + Grad_u_n);
	       Tensor<2, dim> F_n_1 = (Physics::Elasticity::StandardTensors< dim >::I + Grad_u_n_1);

           J     =  determinant(F);

           J_n   =  determinant(F_n);
           J_n_1 =  determinant(F_n_1);
              
           cell_density = c;
           old_cell_density = c_n;
	   old_old_cell_density = c_n_1;


           velocity = density -> compute_speed(F,c[NU],p, NU);

	   for (unsigned int i=0; i < old_velocity_values.size(); ++i){
		   dcc_r[i] = density-> get_dcc_r(p, i);
		   old_velocity_values[i] = density-> compute_speed (F_n, c_n[i], p, i);
		   old_old_velocity_values[i] = density-> compute_speed (F_n_1, c_n_1[i], p, i);
	  }

                     if(update){ 
                         growth-> update_growth(p, c);
                         Tensor<2,dim> F_g= growth->get_growth_tensor();
        		
                         inv_F_g=invert(F_g);
                        }
   
                      F_e = F * inv_F_g;
         
            density-> update_flux(F, Grad_c, c ,p, t);
            density-> compute_denisty_source(t, d_t,c_n, sources);
            material-> update_material_data(F_e, p, c_n[NU]);


            tau            = J * (material-> get_Cauchy_stress());
            elastic_tensor = J * (material-> get_tangent_tensor());
            dP_e_dF_e      = material-> get_tangent_tensor_elastic();
            dP_e_dc        = (update ? material-> get_stress_dervitive_wrt_cell():Tensor<2,dim>());
            G              = (update ? growth-> get_G():Tensor<2,dim>());
              
         }


        Tensor<2 ,dim> get_dtau_dc() const {

                Tensor<2 ,dim> dtau_dc;
                double         J_g         = 1/determinant(inv_F_g);
                          
                Tensor<2 ,dim> tr_inv_F_g  = transpose(inv_F_g);
                Tensor<2, dim> trm_t       = transpose(F_e * G * invert(F));  
		Tensor<2, dim> src         = F_e * G * inv_F_g;

		src = NonStandardTensors::fourth_second_orders_contrac<dim>(dP_e_dF_e, src);
  
                dtau_dc  = scalar_product(tr_inv_F_g, G) * tau;
                dtau_dc -= tau * trm_t;
                dtau_dc -= J_g * src * transpose(F_e);
		dtau_dc += J_g * dP_e_dc * transpose(F_e);
		
                return dtau_dc;
              }


           
           SymmetricTensor<4, dim> get_elastic_tensor() const {return elastic_tensor;}
           SymmetricTensor<2, dim> get_tau() const {return tau;}
    
           Tensor<3 ,dim> get_Jdq_dF_Ft(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (J*(density->get_flux_deformation_derivative(cell_type)));
             }

           Tensor<2, dim> get_inv_F() const {return invert(F);}
    
           Tensor<2 ,dim> get_dq_dgrad_c(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (J*(density-> get_diffusion_tensor(cell_type)));
             }
    
           Tensor<1 ,dim> get_Jq_migration(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (J*(density-> get_flux_migration(cell_type)));
             }
    
           Tensor<1 ,dim> get_Jq_diffusion(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (J*(density-> get_flux_diffusion(cell_type)));
             }
 
           Tensor<1, dim> get_Jdq_dc(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (J*(density->get_flux_derivative(cell_type)));
            }

           Tensor<1, dim> get_velocity() const {return velocity;}
    
           Tensor<1, dim> get_grad_c_spatial(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return (density->get_grad_c_s(cell_type));
            }

           double get_dcc_r(const int cell_type) const {
            Assert(cell_type < 4, ExcInternalError());
            return dcc_r[cell_type];
           }

           Tensor<1, dim> get_old_velocity_values(const int cell_type) const {
            Assert(cell_type < 4, ExcInternalError());
            return old_velocity_values[cell_type];
           }

           Tensor<1, dim> get_old_old_velocity_values(const int cell_type) const {
            Assert(cell_type < 4, ExcInternalError());
            return old_old_velocity_values[cell_type];
           }

    
           double  get_c(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return cell_density[cell_type];
             }

           double  get_c_old(const int cell_type) const {
                Assert(cell_type < 4, ExcInternalError());
                return old_cell_density[cell_type];
              }
    
           double  get_F_c(const int cell_type) const {
               Assert(cell_type < 4, ExcInternalError());
               return sources[cell_type];
             }

    
           double  get_J() const {return J;}
           double  get_J_old() const {return J_n;}
           double  get_J_old_old() const {return J_n_1;}
           double  get_growth_factor_t() const {return (growth-> get_growth_factor_tangent());}
           double  get_growth_factor_r() const {return (growth-> get_growth_factor_radius());}
           double  get_growth_norm() const {  
   
                 if(dim == 3)
                   return ((growth-> get_growth_tensor().norm())/std::sqrt(3));

                 else if(dim == 2)
                   return ((growth-> get_growth_tensor().norm())/std::sqrt(2));
                  }

          
          double get_elastic_modulus() const {return (material-> get_elastic_modulus());}

	private:
	NeoHookeanMaterial<dim> *material;
    Growth<dim>             *growth;
    CellDensity<dim>        *density;
    
    std::vector<double> cell_density;
    std::vector<double> old_cell_density;
    std::vector<double> old_old_cell_density;
    std::vector<double> sources;
    std::vector<double> dcc_r;


    Point<dim>  p;
    Tensor<1, dim> velocity;
    std::vector<Tensor<1, dim> > old_velocity_values;
    std::vector<Tensor<1, dim> > old_old_velocity_values;


    Tensor<2, dim> F;
    Tensor<2 ,dim> inv_F_g;
    Tensor<2, dim> F_e;
    Tensor<2 ,dim> dP_e_dc;
    Tensor<2, dim> G ;
	
    SymmetricTensor<2 ,dim> tau;
    SymmetricTensor<4, dim> elastic_tensor;
    Tensor<4, dim> dP_e_dF_e;



    double J;
    double J_n;
    double J_n_1;
    int a;

    
    enum
    {
      RG = 0,
      IP = 1,
      OR = 2,
      NU = 3
    };
    
};


#endif
