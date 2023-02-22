#ifndef CELLDENSITY_H
#define CELLDENSITY_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>
#include <cmath>

using namespace dealii;

template<int dim>
class CellDensity
{
     public:
       
	CellDensity(const double &cell_migration_threshold, const double &exponent,  const double &MST_factor,
                const std::vector<double> &cell_migration_speed, const std::vector<double> &diffusivity ,const std::vector<double> &Z_raduis,
                const std::vector<double> P_weeks, const std::vector<std::vector<int> > P_ratios):
        c_0(cell_migration_threshold),gamma(exponent), c_mst(MST_factor),
        r_osvz_t(Z_raduis[ISVZ]), v(cell_migration_speed), d_cc(diffusivity),
        zones_radius(Z_raduis), phase_days(P_weeks), phase_ratio(P_ratios),
        grad_c_s(4, Tensor<1, dim>()), first_flux_terms(4, Tensor<1, dim>()),
        second_flux_terms(4, Tensor<1, dim>()), flux_derivatives(4, Tensor<1, dim>()),
        diffusion_tensor (4, Tensor<2 ,dim>()), flux_deformation_derivatives(4, Tensor<3, dim>())
        {}

        ~CellDensity(){}


        void update_flux(const Tensor<2, dim> &F, const std::vector<Tensor<1, dim> > &Grad_c, const std::vector<double> &c, const Point<dim> &p, const double time)
          {
                
               std::vector<Tensor<1 ,dim>> speed(4, Tensor<1 ,dim>());
               std::vector<double> cof(4, 0);
               Tensor<2 ,dim> I = Physics::Elasticity::StandardTensors< dim >::I;
            
               r_osvz_t = zones_radius[ISVZ]+ (c_mst) * ((time >= 10)? (time-10):((time > 20)? 10:0));
               r_osvz_t = ((r_osvz_t> zones_radius[OSVZ])?  zones_radius[OSVZ]:r_osvz_t);
            
               Tensor<1 ,dim> N = direction_vector(p);
               Tensor<1 ,dim> n = F*N;
                    
                for (unsigned int s = 0; s < first_flux_terms.size(); ++s){
                    speed[s] = compute_speed(F,c[s], p, s);
                    cof[s] = heaviside_function((c[s]-c_0), gamma) + c[s] * heaviside_function_derivative((c[s]-c_0), gamma);
                    first_flux_terms[s] = -c[s] * speed[s];
                    flux_derivatives[s] = -cof[s] * (speed[s]/heaviside_function((c[s]-c_0),gamma));        // dq_dc
                    grad_c_s[s] = Grad_c[s] * invert(F);
                    diffusion_tensor[s] = get_dcc_r(p, s) * I; //dq_dgrad_c
                    second_flux_terms[s] = diffusion_tensor[s] * grad_c_s[s]; // q_2
                }

               {
                   std::vector<double> cof_2(4, 0);
                   
                   double a = 1/std::pow((n.norm()),2);
                   Tensor<2, dim> nxn = outer_product(n,n);
                   Tensor<2, dim> direction_derivative = I - (a * nxn);
                   
                   for(int d = 0; d < flux_deformation_derivatives.size(); ++d){
                       cof_2[d] = -c[d] * heaviside_function((c[d]-c_0), gamma) * (get_v_r(p, d)/n.norm());
                       flux_deformation_derivatives[d] = cof_2[d] * outer_product(direction_derivative, n);
                       flux_deformation_derivatives[d] -= outer_product(grad_c_s[d], diffusion_tensor[d]);
                   }
             
                } // compute dq_dF_Ft
           }
      
    void compute_denisty_source(const double t, const double d_t,  const int a ,const std::vector<double> Old_values, std::vector<double> &sources)
    {
        int ph = (t < phase_days[0] ? 0 : (t < phase_days[1] ? 1 : (t < phase_days[2] ? 2 : (t < phase_days[3] ? 3 : 4))));
            sources[RG] = std::pow(phase_ratio[0][ph],d_t)  * Old_values[RG];
            sources[IP] = std::pow(phase_ratio[1][ph],d_t)  * Old_values[RG] + std::pow(phase_ratio[2][ph],d_t)  * Old_values[OR] - std::pow(phase_ratio[3][ph],d_t) * Old_values[IP];
            sources[OR] = std::pow(phase_ratio[4][ph],d_t)  * Old_values[RG];
            sources[NU] = std::pow(phase_ratio[5][ph],d_t)  * Old_values[IP];
        
      
               }

    
  Tensor<1, dim> compute_speed(const Tensor<2, dim> &F ,const double &c ,const Point<dim> &p, const int cell_type)
    {
        
        double v_r  = get_v_r(p, cell_type);

        Tensor<1 ,dim> N = direction_vector(p);
        Tensor<1 ,dim> n = F * N;
        Tensor<1 ,dim> speed = v_r * (n/n.norm());
        return (heaviside_function((c-c_0),gamma) * speed);
    }
        
        double  get_v_r(const Point<dim> &p, const int cell_type)  {
            double r =0;
            if (dim ==2)
                r = p.distance(Point<dim>(0.0,0.0));
            else if (dim ==3)
                r = p.distance(Point<dim>(0.0,0.0, 0.0));
            double exp = (cell_type == NU)? 10:20;
            double R = (cell_type == NU)? zones_radius[CR]:r_osvz_t;
            return (v[cell_type]*(1-heaviside_function((r-R),exp)));
        }
    
        double  get_dcc_r(const Point<dim> &p, const int cell_type)  {
            double value = 0;
            double r =0;
            if (dim ==2)
                r = p.distance(Point<dim>(0.0,0.0));
            else if (dim ==3)
                r = p.distance(Point<dim>(0.0,0.0, 0.0));
            
            if (cell_type == NU)
                value = d_cc[cell_type] * (heaviside_function((r-zones_radius[CR]),10));

            else if (cell_type == RG)
                value = d_cc[cell_type] * (1-heaviside_function((r-zones_radius[VZ]),20));

	    else
                value = d_cc[cell_type] * (1-heaviside_function((r-r_osvz_t),20));
            return value;
        }
        

        Tensor<1, dim> get_grad_c_s(const int cell_type) const {return grad_c_s[cell_type];}
        Tensor<1, dim> get_flux_migration(const int cell_type) const {return first_flux_terms[cell_type];}
        Tensor<1, dim> get_flux_diffusion(const int cell_type) const{return second_flux_terms[cell_type];}
        Tensor<1 ,dim> get_flux_derivative(const int cell_type) const {return flux_derivatives[cell_type];}
        Tensor<2 ,dim> get_diffusion_tensor(const int cell_type) const {return diffusion_tensor[cell_type];}
        Tensor<3 ,dim> get_flux_deformation_derivative(const int cell_type) const {return flux_deformation_derivatives[cell_type];}

     protected:
	
        double c_0;
        double gamma;
        double c_mst;
        double r_osvz_t;
    
        enum
        {
          RG = 0,
          IP = 1,
          OR = 2,
          NU = 3
        };
    
        enum
        {
          VZ = 0,
          ISVZ = 1,
          OSVZ = 2,
          CR = 3
        };
    
        std::vector<double> v;   // v_rg v_ip v_or v_nu
        std::vector<double> d_cc;
        std::vector<double> zones_radius;            // vz isvz osvz c
        std::vector<double> phase_days;
        std::vector<std::vector<int> > phase_ratio; // r_rg r_ip r_or r_nu

        std::vector<Tensor<1 ,dim> > grad_c_s;            // ip or nu
        std::vector<Tensor<1, dim> > first_flux_terms;    // flux_rg flux_ip flux_or flux_nu
        std::vector<Tensor<1 ,dim> > second_flux_terms;
        std::vector<Tensor<1 ,dim> > flux_derivatives;    // dq / dc
        std::vector<Tensor<2 ,dim> > diffusion_tensor;                 // d^cc Tensor
        std::vector<Tensor<3, dim> > flux_deformation_derivatives;     // dq / dF * F^t


        Tensor<1 ,dim> direction_vector(const Point<dim> &p)
          {
               return Tensor<1, dim>(p/p.norm());
              }       // n= x/|x|


       double heaviside_function(const double &x, const double c)  // H(c-c_0; gamma)
        {
             
             double a = std::exp(c*x);
             double b = 1.+std::exp(c*x);

            return a/b;
        
          }

      double heaviside_function_derivative(const double &x, const double c)  // dH_dc
        {
            
              double a = c * std::exp(c*x);
              double b = std::pow((1.+std::exp(c*x)),2);
                return a/b;
           
             }

};

#endif
