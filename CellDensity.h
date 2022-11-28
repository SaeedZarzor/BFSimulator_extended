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
       
	CellDensity(const double &diffusivity, const double &cell_migration_threshold, const double &exponent,  const double &MST_factor,
                const std::vector<double> &cell_migration_speed, const std::vector<double> &Z_raduis,
                const std::vector<std::vector<double> > P_ratios):
        d_cc(diffusivity), c_0(cell_migration_threshold),gamma(exponent),
        c_mst(MST_factor), r_osvz_t(Z_raduis[ISVZ]), v(cell_migration_speed),
        zones_radius(Z_raduis), phase_ratio(P_ratios),
        grad_c_s(3, Tensor<1, dim>()), flux_terms(4, Tensor<1, dim>()),
        flux_derivatives(4, Tensor<1, dim>()), diffusion_tensor (Tensor<2 ,dim>()),
        flux_deformation_derivatives(4, Tensor<3, dim>())
        {}

        ~CellDensity(){}


        void update_flux(const Tensor<2, dim> &F, const std::vector<Tensor<1, dim> > &Grad_c, const std::vector<double> &c, const Point<dim> &p, const double time)
          {
                
               std::vector<Tensor<1 ,dim>> speed(4, Tensor<1 ,dim>());
               std::vector<double> cof(4, 0);
               Tensor<2 ,dim> I = Physics::Elasticity::StandardTensors< dim >::I;
            
               r_osvz_t = zones_radius[ISVZ]+ c_mst * ((time > 0)? time:0);
               r_osvz_t = ((r_osvz_t> zones_radius[OSVZ])?  zones_radius[OSVZ]:r_osvz_t);
            
               Tensor<1 ,dim> N = direction_vector(p);
               Tensor<1 ,dim> n = F*N;
                    
                for (unsigned int s=0; s< flux_terms.size(); ++s){
                    speed[s] = compute_speed(F,c[s], p, s);
                    cof[s] = heaviside_function((c[s]-c_0), gamma) + c[s] * heaviside_function_derivative((c[s]-c_0), gamma);
                    flux_terms[s] = -c[s] * speed[s];
                    flux_derivatives[s] = -cof[s] * (speed[s]/heaviside_function((c[s]-c_0),gamma));        // dq_dc
                }
            
                for(unsigned int i=0; i< Grad_c.size(); ++i)
                     grad_c_s[i] = Grad_c[i] * invert(F);
               

               diffusion_tensor = get_dcc_r(p) * I; //dq_dgrad_c
               second_neurons_flux_term = diffusion_tensor* grad_c_s[NU-1]; // q_nu_2

               {
                   std::vector<double> cof_2(4, 0);
                   
                   double a = 1/std::pow((n.norm()),2);
                   Tensor<2, dim> nxn = outer_product(n,n);
                   Tensor<2, dim> direction_derivative = I - (a * nxn);
                   
                   for(int d=0; d< flux_deformation_derivatives.size(); ++d){
                       cof_2[d] = -c[d] * heaviside_function((c[d]-c_0), gamma) * (get_v_r(p, d)/n.norm());
                       flux_deformation_derivatives[d] = cof_2[d] * outer_product(direction_derivative, n);
                   }
             
                   flux_deformation_derivatives[NU] = flux_deformation_derivatives[NU]  - outer_product(grad_c_s[NU-1], diffusion_tensor);
                } // compute dq_dF_Ft
           }
      
    void compute_denisty_source(const double t, const double d_t ,const std::vector<double> Old_values, std::vector<double> &sources)
         {
            const int a= 4;
            int ph = (t < 5 ? 0 : (t < 10 ? 1 : (t < 20 ? 2 : (t < 25 ? 3 : 4))));
            sources[RG] = phase_ratio[0][ph] * d_t * Old_values[RG];
            sources[IP] = phase_ratio[1][ph] * d_t * Old_values[RG] + phase_ratio[2][ph] * d_t * Old_values[OR] - phase_ratio[3][ph] * (d_t/a) * Old_values[IP];
            sources[OR] = phase_ratio[4][ph] * d_t * Old_values[OR];
            sources[NU] = phase_ratio[5][ph] * (d_t/a) * Old_values[IP];
        
      
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
            double r = p.distance(Point<dim>(0.0,0.0));
            double exp = (cell_type == NU)? 10:50;
            double R = (cell_type == NU)? zones_radius[CR]:r_osvz_t;
            return (v[cell_type]*(1-heaviside_function((r-R),exp)));
        }
    
        double  get_dcc_r(const Point<dim> &p)  {
            double r = p.distance(Point<dim>(0.0,0.0));
            return (d_cc*(heaviside_function((r-zones_radius[CR]),10)));
        }
        
        Tensor<1, dim> get_flux_ip() const {return flux_terms[IP];}
        Tensor<1, dim> get_flux_or() const {return flux_terms[OR];}
        Tensor<1, dim> get_grad_c_ip_s() const {return grad_c_s[IP-1];}
        Tensor<1, dim> get_grad_c_or_s() const {return grad_c_s[OR-1];}
        Tensor<1, dim> get_grad_c_nu_s() const {return grad_c_s[NU-1];}
        Tensor<1, dim> get_flux_nu_migration() const {return flux_terms[NU];}
        Tensor<1 ,dim> get_flux_ip_derivative() const {return flux_derivatives[IP];}
        Tensor<1 ,dim> get_flux_or_derivative() const {return flux_derivatives[OR];}
        Tensor<1 ,dim> get_flux_nu_derivative() const {return flux_derivatives[NU];}
        Tensor<2 ,dim> get_diffusion_tensor() const {return diffusion_tensor;}
        Tensor<3 ,dim> get_flux_ip_deformation_derivative() const {return flux_deformation_derivatives[IP];}
        Tensor<3 ,dim> get_flux_or_deformation_derivative() const {return flux_deformation_derivatives[OR];}
        Tensor<3 ,dim> get_flux_nu_deformation_derivative() const {return flux_deformation_derivatives[NU];}



     protected:
	
        double d_cc;
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
        std::vector<double> zones_radius;            // vz isvz osvz c
        std::vector<std::vector<double> > phase_ratio; // r_rg r_ip r_or r_nu

        std::vector<Tensor<1 ,dim> > grad_c_s;            // ip or nu
        std::vector<Tensor<1, dim> > flux_terms;          // flux_rg flux_ip flux_or flux_nu
        std::vector<Tensor<1 ,dim> > flux_derivatives;    // dq / dc
        Tensor<1, dim> second_neurons_flux_term;
        Tensor<2 ,dim> diffusion_tensor;                  // d^cc Tensor
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
