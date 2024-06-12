 /* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2015 by the deal.II authors and
 *                              & Jean-Paul Pelteret and Andrew McBride
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Jean-Paul Pelteret, University of Cape Town,
 *          Andrew McBride, University of Erlangen-Nuremberg, 2010
 */


#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/tensor_function.h>


#include <iostream>
#include <math.h>
#include <fstream>
#include <cmath>

#include "PointHistory.h"
#include "Parameter.h"


namespace Brain_growth
{
  using namespace dealii;


    
  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t)
    {}

    virtual ~Time()
    {}

    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

   void set_delta_t(double x) 
    {
       delta_t = x;
    }
  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    double delta_t;
  };


  template <int dim>
  class InitialValueC : public Function<dim>
  {

      public:
       InitialValueC(const double d_r, const double d_v): Function<dim>(dim+4)
        {dvision_raduis = d_r; 
	 dvision_value = d_v;}

	virtual void vector_value(const Point<dim> &p, Vector<double>& value) const override
	 {
	     Assert(value.size() == dim + 4, ExcDimensionMismatch(value.size(), dim + 4));

		
         double r =0;
         if (dim ==2) {
             r = p.distance(Point<dim>(0.0,0.0));}
         else  {
             r = p.distance(Point<dim>(0.0,0.0, 0.0));}
         

	double H = (std::exp((r-dvision_raduis)*20))/(1.+std::exp((r-dvision_raduis)*20));

             value(dim) = dvision_value*(1-H);
		}
	

      private:


	 double dvision_raduis;
	 double dvision_value;
   };

template <int dim>
class Rotate3d
  {
     public:
       Rotate3d (const double angle,
                 const unsigned int axis)
         :
         angle(angle),
         axis(axis)
       {}
 
       Point<dim> operator() (const Point<dim> &p) const
       {
         if (axis==0)
           return Point<dim> (p(0),
                            std::cos(angle)*p(1) - std::sin(angle) * p(2),
                            std::sin(angle)*p(1) + std::cos(angle) * p(2));
         else if (axis==1)
           return Point<dim> (std::cos(angle)*p(0) + std::sin(angle) * p(2),
                            p(1),
                            -std::sin(angle)*p(0) + std::cos(angle) * p(2));
         else
           return Point<dim> (std::cos(angle)*p(0) - std::sin(angle) * p(1),
                            std::sin(angle)*p(0) + std::cos(angle) * p(1),
                            p(2));
       }
     private:
       const double angle;
       const unsigned int axis;
     };


  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);

    virtual
    ~Solid();

    void
    run();

  private:

  
    struct PerTaskData_K;
    struct ScratchData_K;

    struct PerTaskData_RHS;
    struct ScratchData_RHS;

    struct PerTaskData_SQPH;
    struct ScratchData_SQPH;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    struct PerTaskData_SC;
    struct ScratchData_SC;

    void make_grid();

    void system_setup();

    void determine_component_extractors();
      
    void assemble_system_tangent();

    void assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     ScratchData_K &scratch,
                                     PerTaskData_K &data);

    void copy_local_to_global_K(const PerTaskData_K &data);

    void assemble_system_rhs();

    void assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_RHS &scratch,
                                 PerTaskData_RHS &data);

    void copy_local_to_global_rhs(const PerTaskData_RHS &data);


    void make_constraints(const int &it_nr);

 
    void setup_qph();

    void setup_qph_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    ScratchData_SQPH &scratch,
                                    PerTaskData_SQPH &data);
	
    void copy_local_to_global_SQPH(const PerTaskData_SQPH &/*data*/) {}

    void update_qph_incremental(const BlockVector<double> &solution_delta, bool update_growth);

    void update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    ScratchData_UQPH &scratch,
                                    PerTaskData_UQPH &data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH &/*data*/) {}
 
    void copy_local_to_global_sc(const PerTaskData_SC &data);

    void solve_nonlinear_timestep(BlockVector<double> &solution_delta, bool &CONVERGED);

    void assemble_sc();

    void assemble_sc_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                         ScratchData_SC &scratch,
                         PerTaskData_SC &data);

    std::pair<unsigned int, double> solve_linear_system(BlockVector<double> &newton_update);

    BlockVector<double> get_total_solution(const BlockVector<double> &solution_delta) const;
      
    std::pair<double, double> get_cell_density_range_end_time_step() const;
      

    void compute_delta_t();
      
    double compute_viscosity( PointHistory<dim> *lqph,
                            const double cell_diameter, const int cell_type) const;

    void output_results();

    void sptial_density_grads_projection(BlockVector<double> &sptial_density_grads);

    void growth_factor_projection(BlockVector<double> &growth_factor);

    void compute_growth_norm_projection(BlockVector<double> &growth_norm);

    void compute_Stiffness_projection(BlockVector<double> &Stiffness);

    void compute_stress_projection_and_average( BlockVector<double> &stress_projected, Vector<double> &stress_averaged);
      
//    void source_terms_projection(BlockVector<double> &source_terms);
      
    void velocity_projection(BlockVector<double> &velocity_values);
      
    void diffusion_projection(BlockVector<double> &diffusion_values);

    
    Parameter::GeneralParameters     parameters;
    double                           vol_reference;
    double                           vol_current;
 

    Triangulation<dim>               triangulation;

    Time                             time;
    TimerOutput                      timer;


    std::vector<PointHistory<dim> >  quadrature_point_history;

//std::vector<double>              stretch_max{1.0,1.0,1.0}; // [stretch_max , old_stretch_max, old_old_stretch_max]
    const unsigned int               degree;
    const FESystem<dim>              fe;
    double                           global_Omega_diameter;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;
    const FEValuesExtractors::Scalar c_rg_fe;
    const FEValuesExtractors::Scalar c_ip_fe;
    const FEValuesExtractors::Scalar c_or_fe;
    const FEValuesExtractors::Scalar c_n_fe;

    static const unsigned int        n_blocks = 5;
    static const unsigned int        n_components = dim + 4;
    static const unsigned int        first_u_component = 0;
    static const unsigned int        c_rg_component = dim;
    static const unsigned int        c_ip_component = dim + 1;
    static const unsigned int        c_or_component = dim + 2;
    static const unsigned int        c_n_component = dim + 3;


    enum
    {
      u_dof = 0,
      c_rg_dof = 1,
      c_ip_dof = 2,
      c_or_dof = 3,
      c_nu_dof = 4
    };
      
      enum
      {
        RG = 0,
        IP = 1,
        OR = 2,
        NU = 3
      };
      

    std::vector<types::global_dof_index>        dofs_per_block;
    std::vector<types::global_dof_index>        element_indices_u;
    std::vector<types::global_dof_index>        element_indices_c_rg;
    std::vector<types::global_dof_index>        element_indices_c_ip;
    std::vector<types::global_dof_index>        element_indices_c_or;
    std::vector<types::global_dof_index>        element_indices_c_nu;

      
    const QGauss<dim>                qf_cell;
    const QGauss<dim - 1>            qf_face;
    const unsigned int               n_q_points;
    const unsigned int               n_q_points_f;


    AffineConstraints<double>        constraints;
    BlockSparsityPattern             sparsity_pattern;
    BlockSparseMatrix<double>        tangent_matrix;
    BlockSparseMatrix<double>        matrix_sc;
    BlockVector<double>              system_rhs;
    BlockVector<double>              solution_n;
    BlockVector<double>              old_solution;
    BlockVector<double>              old_old_solution;

    struct Errors
    {
      Errors()
        :
        norm(1.0), u(1.0), c_rg(1.0), c_ip(1.0), c_or(1.0), c_n(1.0)
      {}

      void reset()
      {
        norm = 1.0;
        u = 1.0;
        c_rg = 1.0;
        c_ip = 1.0;
        c_or = 1.0;
        c_n = 1.0;
      }
      void normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
        if (rhs.c_rg != 0.0)
          c_rg /= rhs.c_rg;
        if (rhs.c_ip != 0.0)
            c_ip /= rhs.c_ip;
        if (rhs.c_or != 0.0)
            c_or /= rhs.c_or;
        if (rhs.c_n != 0.0)
            c_n /= rhs.c_n;
      }

      double norm, u, c_rg, c_ip, c_or, c_n;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update,
           error_update_0, error_update_norm;


    void get_error_residual(Errors &error_residual);

    void get_error_update(const BlockVector<double> &newton_update,
                     Errors &error_update);


    static void print_conv_header();

    void print_conv_footer();
    
    //class Postprocessor;
  };


  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    :
    parameters(input_file),
    triangulation(Triangulation<dim>::maximum_smoothing),
    time(parameters.total_time, parameters.time_step),
    timer(std::cout,
          TimerOutput::summary,
          TimerOutput::wall_times),
    degree(parameters.degree),
    fe(FE_Q<dim>(parameters.degree), dim, // displacement
      FE_Q<dim>(parameters.degree), 1,    // RG_cell_density
      FE_Q<dim>(parameters.degree), 1,    // IP_cell_density
      FE_Q<dim>(parameters.degree), 1,    // OR_cell_density
      FE_Q<dim>(parameters.degree), 1),   // N_cell_density
    global_Omega_diameter(std::numeric_limits<double>::quiet_NaN()),
    dof_handler_ref(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(first_u_component),
    c_rg_fe(c_rg_component),
    c_ip_fe(c_ip_component),
    c_or_fe(c_or_component),
    c_n_fe(c_n_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.degree + 1),
    qf_face(parameters.degree +1),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())
  {

        Assert(dim == 2 || dim == 3,
                    ExcMessage("This problem only works in 2 or 3 space dimensions."));
               determine_component_extractors();
  }


  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler_ref.clear();
  }



  template <int dim>
  void Solid<dim>::run()
  {

     std::ofstream timeing;
    timeing.open( "timeing.csv");
    timeing << "time step," << "current time," << "min cell density," << "max cell density \n";
    make_grid();
    system_setup();
    {
      AffineConstraints<double>  constraints;
      constraints.close();

       
      VectorTools::project (dof_handler_ref,
                            constraints,
                            QGauss<dim>(degree+2),
                            InitialValueC<dim>((parameters.zones_raduis[0]), parameters.dvision_value),
                            solution_n);
                   constraints.distribute(solution_n);
    }
    output_results();
    time.increment();
    bool CONVERGED;
    old_old_solution = 0.0;
    old_solution = solution_n;
    

   BlockVector<double> solution_delta(dofs_per_block);
    while (time.current() < time.end())
      {
	 if(time.get_timestep() > 1 ){
	    old_old_solution = old_solution;
            old_solution = solution_n;
	    }
			
        solution_delta = 0.0;
          for(unsigned int k = 0; k < 3; ++k) {
//              if(time.current() >= parameters.phase_days[3])
//                  time.set_delta_t(parameters.time_step*0.25);
              
              solve_nonlinear_timestep(solution_delta, CONVERGED);
              
                      if (CONVERGED == true)
                              break;
              
                  compute_delta_t();
                  }

	if (CONVERGED != true)
		break;

        solution_n += solution_delta;

        
        output_results();
        
        std::pair<double, double> cell_range = get_cell_density_range_end_time_step();
	timeing << time.get_timestep() << "," << time.current() << "," << cell_range.first << "," << cell_range.second <<"\n";
        time.increment();
      }
  timeing.close();
  Assert((CONVERGED==true),ExcMessage("Not CONVERGED!"));
  }



  template <int dim>
  struct Solid<dim>::PerTaskData_K
  {
    FullMatrix<double>        cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_K(const unsigned int dofs_per_cell)
      :
      cell_matrix(dofs_per_cell, dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_matrix = 0.0;
    }
  };


  template <int dim>
  struct Solid<dim>::ScratchData_K
  {
    FEValues<dim> fe_values_ref;

    std::vector<std::vector<std::vector<double> > >            Nx_c; // first the cell type second the qp third dof
    std::vector<std::vector<std::vector<Tensor<1, dim> > > >   grad_Nx_c;
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx_u;
    std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx_u;

    ScratchData_K(const FiniteElement<dim> &fe_cell,
                  const QGauss<dim> &qf_cell,
                  const UpdateFlags uf_cell)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      Nx_c(4, std::vector<std::vector<double> >(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell))),
      grad_Nx_c(4,  std::vector<std::vector<Tensor<1, dim> > >(qf_cell.size(), std::vector<Tensor<1, dim> > (fe_cell.dofs_per_cell))),
      grad_Nx_u(qf_cell.size(), std::vector<Tensor<2, dim> > (fe_cell.dofs_per_cell)),
      symm_grad_Nx_u(qf_cell.size(), std::vector<SymmetricTensor<2, dim> > (fe_cell.dofs_per_cell))
    {}

    ScratchData_K(const ScratchData_K &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
      Nx_c(rhs.Nx_c),
      grad_Nx_c(rhs. grad_Nx_c),
      grad_Nx_u(rhs.grad_Nx_u),
      symm_grad_Nx_u(rhs.symm_grad_Nx_u)
    {}

    void reset()
      {
        const unsigned int n_dofs_per_cell = Nx_c[0][0].size();
        unsigned int ii=0;
        
        for (unsigned int q_point = 0; q_point < Nx_c[0].size(); ++q_point)
        {
            Assert( grad_Nx_u[q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( symm_grad_Nx_u[q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( Nx_c[0][q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( grad_Nx_c[0][q_point].size() == n_dofs_per_cell,ExcInternalError());
            
            for (unsigned int k = 0; k < Nx_c[0][0].size(); ++k)
            {
                grad_Nx_u[q_point][k] = 0.0;
                symm_grad_Nx_u[q_point][k] = 0.0;
                
                for (; ii < Nx_c.size(); ++ii){
                    Nx_c[ii][q_point][k] = 0.0;
                    grad_Nx_c[ii][q_point][k] = 0.0;
                }
            }
        }
    }
  };

template <int dim>
  struct Solid<dim>::PerTaskData_SC
  {
        FullMatrix<double>        cell_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
      
        FullMatrix<double> k_orig;
        FullMatrix<double> k_bar_rg;
        FullMatrix<double> k_bar_ip;
        FullMatrix<double> k_bar_or;
        FullMatrix<double> k_bar_nu;
      
        FullMatrix<double> k_u_rg;
        FullMatrix<double> k_rg_u;
        FullMatrix<double> k_B_rg;
        FullMatrix<double> k_rg_rg;
        FullMatrix<double> k_rg_rg_inv;
      
        FullMatrix<double> k_u_ip;
        FullMatrix<double> k_ip_u;
        FullMatrix<double> k_B_ip;
        FullMatrix<double> k_ip_ip;
        FullMatrix<double> k_ip_ip_inv;
      
        FullMatrix<double> k_u_or;
        FullMatrix<double> k_or_u;
        FullMatrix<double> k_B_or;
        FullMatrix<double> k_or_or;
        FullMatrix<double> k_or_or_inv;
      
        FullMatrix<double> k_u_nu;
        FullMatrix<double> k_nu_u;
        FullMatrix<double> k_B_nu;
        FullMatrix<double> k_nu_nu;
        FullMatrix<double> k_nu_nu_inv;
      
      


    PerTaskData_SC(const unsigned int dofs_per_cell, const unsigned int n_u,
                   const unsigned int n_rg, const unsigned int n_ip,
                   const unsigned int n_or, const unsigned int n_nu):
      cell_matrix(dofs_per_cell, dofs_per_cell),
      local_dof_indices(dofs_per_cell),
      k_orig (dofs_per_cell, dofs_per_cell),
      k_bar_rg (n_u ,n_u),
      k_bar_ip (n_u ,n_u),
      k_bar_or (n_u ,n_u),
      k_bar_nu (n_u ,n_u),
      k_u_rg (n_u, n_rg),
      k_rg_u (n_rg, n_u),
      k_B_rg (n_rg, n_u),
      k_rg_rg (n_rg, n_rg),
      k_rg_rg_inv (n_rg, n_rg),
      k_u_ip (n_u, n_ip),
      k_ip_u (n_ip, n_u),
      k_B_ip (n_ip, n_u),
      k_ip_ip (n_ip, n_ip),
      k_ip_ip_inv (n_ip, n_ip),
      k_u_or (n_u, n_or),
      k_or_u (n_or, n_u),
      k_B_or (n_or, n_u),
      k_or_or (n_or, n_or),
      k_or_or_inv (n_or, n_or),
      k_u_nu (n_u, n_nu),
      k_nu_u (n_nu, n_u),
      k_B_nu (n_nu, n_u),
      k_nu_nu (n_nu, n_nu),
      k_nu_nu_inv (n_nu, n_nu)
    {}
    void reset()
    {}
  };

  template <int dim>
  struct Solid<dim>::ScratchData_SC
  {
    void reset()
    {}
  };

  template <int dim>
  struct Solid<dim>::PerTaskData_RHS
  {
    Vector<double>            cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_RHS(const unsigned int dofs_per_cell)
      :
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_rhs = 0.0;
    }
  };


  template <int dim>
  struct Solid<dim>::ScratchData_RHS
  {
    FEValues<dim>     fe_values_ref;
    FEFaceValues<dim> fe_face_values_ref;

    std::vector<std::vector<std::vector<double> > >            Nx_c; // first the cell type second the qp third dof  RG IP OR N
    std::vector<std::vector<std::vector<Tensor<1, dim> > > >   grad_Nx_c; // IP OR N
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx_u;
    std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx_u;

    ScratchData_RHS(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                    const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      fe_face_values_ref(fe_cell, qf_face, uf_face),
      Nx_c(4, std::vector<std::vector<double> >(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell))),
      grad_Nx_c(4,  std::vector<std::vector<Tensor<1, dim>> >(qf_cell.size(), std::vector<Tensor<1, dim> > (fe_cell.dofs_per_cell))),
      grad_Nx_u(qf_cell.size(), std::vector<Tensor<2, dim> > (fe_cell.dofs_per_cell)),
      symm_grad_Nx_u(qf_cell.size(), std::vector<SymmetricTensor<2, dim> > (fe_cell.dofs_per_cell))
      {}

    ScratchData_RHS(const ScratchData_RHS &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
      fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                         rhs.fe_face_values_ref.get_quadrature(),
                         rhs.fe_face_values_ref.get_update_flags()),
      Nx_c(rhs.Nx_c),
      grad_Nx_c(rhs. grad_Nx_c),
      grad_Nx_u(rhs.grad_Nx_u),
      symm_grad_Nx_u(rhs.symm_grad_Nx_u)
      {}

    void reset()
    {
        const unsigned int n_dofs_per_cell = Nx_c[0][0].size();
        unsigned int ii=0;
        for (unsigned int q_point = 0; q_point < Nx_c[0].size(); ++q_point)
        {
            Assert( grad_Nx_u[q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( symm_grad_Nx_u[q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( Nx_c[0][q_point].size() == n_dofs_per_cell, ExcInternalError());
            Assert( grad_Nx_c[0][q_point].size() == n_dofs_per_cell,ExcInternalError());
            for (unsigned int k = 0; k < Nx_c[0][0].size(); ++k)
            {
                grad_Nx_u[q_point][k] = 0.0;
                symm_grad_Nx_u[q_point][k] = 0.0;
                
                for (; ii < Nx_c.size(); ++ii){
                    Nx_c[ii][q_point][k] = 0.0;
                    grad_Nx_c[ii][q_point][k] = 0.0;
                }
            }
        }
    }

  };


 template <int dim>
  struct Solid<dim>::PerTaskData_SQPH
  {
    void reset()
    {}
  };


  template <int dim>
  struct Solid<dim>::ScratchData_SQPH
  {

    FEValues<dim>                fe_values_ref;

    ScratchData_SQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim> &qf_cell,
                     const UpdateFlags uf_cell)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell)
      {}

    ScratchData_SQPH(const ScratchData_SQPH &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags())
    {}

    void reset()
    {}
  };

  template <int dim>
  struct Solid<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };


  template <int dim>
  struct Solid<dim>::ScratchData_UQPH
  {
    const BlockVector<double>      &solution_total;
    const BlockVector<double>      &old_solution;
    const BlockVector<double>      &old_old_solution;
      
    std::vector<Tensor<2, dim> >   solution_grads_u;
    std::vector<Tensor<2, dim> >   old_solution_grads_u;
    std::vector<Tensor<2, dim> >   old_old_solution_grads_u;
      
    std::vector<std::vector<Tensor<1, dim> > > solution_grads_c;
    std::vector<std::vector<double> >          solution_value_c;  //first vector for cell types and the second for qp
    std::vector<std::vector<double> >          old_solution_value_c;
    std::vector<std::vector<double> >          old_old_solution_value_c;
      
    bool                         update_growth;
    FEValues<dim>                fe_values_ref;


    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim> &qf_cell,
                     const UpdateFlags uf_cell,
                     const BlockVector<double> &solution_total,
                     const BlockVector<double> &old_solution,
                     const BlockVector<double> &old_old_solution,
                     bool update)
      :
      solution_total(solution_total),
      old_solution(old_solution),
      old_old_solution(old_old_solution),
      solution_grads_u(qf_cell.size()),
      old_solution_grads_u(qf_cell.size()),
      old_old_solution_grads_u(qf_cell.size()),
      solution_grads_c(4, std::vector<Tensor<1,dim> >(qf_cell.size())), 
      solution_value_c(4, std::vector<double>(qf_cell.size())), // RG IP OR N
      old_solution_value_c(4, std::vector<double>(qf_cell.size())), // RG IP OR N
      old_old_solution_value_c(4, std::vector<double>(qf_cell.size())), // RG IP OR N
      update_growth(update),
      fe_values_ref(fe_cell, qf_cell, uf_cell)
      {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      :
      solution_total(rhs.solution_total),
      old_solution(rhs.old_solution),
      old_old_solution(rhs.old_old_solution),
      solution_grads_u(rhs.solution_grads_u),
      old_solution_grads_u(rhs.old_solution_grads_u),
      old_old_solution_grads_u(rhs.old_old_solution_grads_u),
      solution_grads_c(rhs.solution_grads_c),
      solution_value_c(rhs.solution_value_c),
      old_solution_value_c(rhs.old_solution_value_c),
      old_old_solution_value_c(rhs.old_old_solution_value_c),
      update_growth (rhs.update_growth),
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags())
      {}

    void reset()
    {
        unsigned int ii =0;
        for (unsigned int q = 0; q < solution_grads_u.size(); ++q)
        {
            solution_grads_u[q] = 0.0;
            old_solution_grads_u[q] = 0.0;
            old_old_solution_grads_u[q] = 0.0;
        
            for (; ii<solution_grads_c.size(); ++ii){
                solution_grads_c[ii][q] = 0.0;
                solution_value_c[ii][q] = 0.0;
                old_solution_value_c[ii][q] = 0.0;
          	old_old_solution_value_c[ii][q]=0.0;
            }

        }

    }
  };



 template <int dim>
  void Solid<dim>::make_grid()
{
    const double PI = 3.1451592654;
    const double angle = PI/2;
    if (dim ==2){
        const Point<dim> Center (0.0, 0.0);
        GridGenerator::quarter_hyper_shell(triangulation, Center, 0.2*parameters.initial_radius , 1.0*parameters.initial_radius,0,true);
        global_Omega_diameter = GridTools::diameter(triangulation);
        GridTools::rotate(angle, triangulation);
        const SphericalManifold<dim> manifold(Center);
        triangulation.set_all_manifold_ids_on_boundary(0);
        triangulation.refine_global(std::max (1U, parameters.global_refinements));
        triangulation.set_manifold (0, manifold);
    }
    
    else if (dim ==3){
        const Point<dim> Center (0.0, 0.0,0.0);
        GridGenerator::half_hyper_shell(triangulation, Center, 0.2*parameters.initial_radius , 1.0*parameters.initial_radius,0,true);
        global_Omega_diameter = GridTools::diameter(triangulation);
        GridTools::transform (Rotate3d<dim>(angle, 2), triangulation);
        const SphericalManifold<dim> manifold(Center);
        triangulation.set_all_manifold_ids_on_boundary(0);
        triangulation.refine_global(std::max (1U, parameters.global_refinements));
        triangulation.set_manifold (0, manifold);

    }
    else
        Assert(dim<3, ExcInternalError());
    

  vol_reference = GridTools::volume(triangulation);
  vol_current = vol_reference;
  std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;

}



  template <int dim>
  void Solid<dim>::system_setup()
  {
    timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(n_components, u_dof); // Displacement
    block_component[c_rg_component] = c_rg_dof; // rg_cell_density
    block_component[c_ip_component] = c_ip_dof; // ip_cell_density
    block_component[c_or_component] = c_or_dof; // or_cell_density
    block_component[c_n_component]  = c_nu_dof;  // n_cell_density


    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);
    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: " << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
              << std::endl;


    tangent_matrix.clear();
    {
      /*const types::global_dof_index n_dofs_u    = dofs_per_block[u_dof];
        const types::global_dof_index n_dofs_c_rg = dofs_per_block[c_rg_dof];
        const types::global_dof_index n_dofs_c_ip = dofs_per_block[c_ip_dof];
        const types::global_dof_index n_dofs_c_or = dofs_per_block[c_or_dof];
        const types::global_dof_index n_dofs_c_n  = dofs_per_block[c_nu_dof];*/

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        
        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
         for (unsigned int ii = 0; ii < n_components; ++ii)
           for (unsigned int jj = 0; jj < n_components; ++jj)
             if (((ii == c_rg_component) && (jj > c_rg_component)) ||
                 ((ii > c_rg_component) && (jj == c_rg_component)) ||
                 ((ii == c_ip_component) && (jj > c_ip_component)) ||
                 ((ii > c_ip_component) && (jj == c_ip_component)) ||
                 ((ii == c_or_component) && (jj == c_n_component)) ||
                 ((ii == c_n_component) && (jj == c_or_component)))
               coupling[ii][jj] = DoFTools::none;
             else
               coupling[ii][jj] = DoFTools::always;
        
      DoFTools::make_sparsity_pattern(dof_handler_ref,
                                      coupling,
                                      dsp,
                                      constraints,
                                      false);
      sparsity_pattern.copy_from(dsp);
	unsigned int number_entries = sparsity_pattern.n_nonzero_elements();
	std::cout<<"Size of sparsity-pattern: "<<number_entries<<std::endl;
        std::ofstream out("sparsity_patteren.svg");
        sparsity_pattern.print_svg(out);
    }

    tangent_matrix.reinit(sparsity_pattern);
    matrix_sc.reinit (sparsity_pattern);
    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    solution_n.reinit(dofs_per_block);
    solution_n.collect_sizes();
    old_solution.reinit(dofs_per_block);
    old_solution.collect_sizes();
    old_old_solution.reinit(dofs_per_block);
    old_old_solution.collect_sizes();

    setup_qph();

    timer.leave_subsection();
  }



  template <int dim>
  void
  Solid<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_c_rg.clear();
    element_indices_c_ip.clear();
    element_indices_c_or.clear();
    element_indices_c_nu.clear();

    for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_dof)
          element_indices_u.push_back(k);
        else if (k_group == c_rg_dof)
          element_indices_c_rg.push_back(k);
        else if (k_group == c_ip_dof)
          element_indices_c_ip.push_back(k);
        else if (k_group == c_or_dof)
          element_indices_c_or.push_back(k);
        else if (k_group == c_nu_dof)
          element_indices_c_nu.push_back(k);
        else
          {
            Assert(k_group <= c_nu_dof, ExcInternalError());
          }
      }
  }


    template <int dim>
  void Solid<dim>::setup_qph()
  {

    std::cout << "    Setting up quadrature point data..." << std::endl;

    {
      triangulation.clear_user_data();
      {
        std::vector<PointHistory<dim> > tmp;
        tmp.swap(quadrature_point_history);
      }

      quadrature_point_history.resize(triangulation.n_active_cells() * n_q_points);

      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator cell =
             triangulation.begin_active(); cell != triangulation.end();
           ++cell)
        {
          cell->set_user_pointer(&quadrature_point_history[history_index]);
          history_index += n_q_points;
        }

      Assert(history_index == quadrature_point_history.size(),
             ExcInternalError());
    }


    const UpdateFlags uf_SQPH(update_quadrature_points);
    PerTaskData_SQPH per_task_data_SQPH;
    ScratchData_SQPH scratch_data_SQPH(fe, qf_cell, uf_SQPH);


    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::setup_qph_one_cell,
                    &Solid::copy_local_to_global_SQPH,
                    scratch_data_SQPH,
                    per_task_data_SQPH);

  }

template <int dim>
  void
  Solid<dim>::setup_qph_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_SQPH &scratch,
                                              PerTaskData_SQPH &/*data*/)
  {
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
    Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());
	
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
	Point<dim> P =scratch.fe_values_ref.quadrature_point(q_point);
        lqph[q_point].setup_lqp(parameters, P, time.get_delta_t());
     }
  }


  template <int dim>
  void Solid<dim>::update_qph_incremental(const BlockVector<double> &solution_delta, bool update_growth)
  {
    timer.enter_subsection("Update QPH data");
    if(!update_growth)
        std::cout << " UQPH " << std::flush;

    const BlockVector<double> solution_total(get_total_solution(solution_delta));

    const UpdateFlags uf_UQPH(update_values | update_gradients | update_quadrature_points);
    PerTaskData_UQPH per_task_data_UQPH;
    ScratchData_UQPH scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total, old_solution, old_old_solution,update_growth);


    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::update_qph_incremental_one_cell,
                    &Solid::copy_local_to_global_UQPH,
                    scratch_data_UQPH,
                    per_task_data_UQPH);

    timer.leave_subsection();
  }



  template <int dim>
  void
  Solid<dim>::update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_UQPH &scratch,
                                              PerTaskData_UQPH &/*data*/)
  {
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
    Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

    Assert(scratch.solution_grads_u.size() == n_q_points, ExcInternalError());
    Assert(scratch.old_solution_grads_u.size() == n_q_points, ExcInternalError());
    Assert(scratch.old_old_solution_grads_u.size() == n_q_points, ExcInternalError());
    for (unsigned int i=0; i<4 ; ++i){
	  Assert(scratch.solution_grads_c[i].size() == n_q_points, ExcInternalError());
   	  Assert(scratch.solution_value_c[i].size() == n_q_points, ExcInternalError());
	  Assert(scratch.old_solution_value_c[i].size() == n_q_points, ExcInternalError());
	  Assert(scratch.old_old_solution_value_c[i].size() == n_q_points, ExcInternalError());
	}

    scratch.reset();

    scratch.fe_values_ref.reinit(cell);
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u);
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.old_solution, scratch.old_solution_grads_u);
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.old_old_solution, scratch.old_old_solution_grads_u);
    scratch.fe_values_ref[c_rg_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_c[RG]);
    scratch.fe_values_ref[c_ip_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_c[IP]);
    scratch.fe_values_ref[c_or_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_c[OR]);
    scratch.fe_values_ref[c_n_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_c[NU]);
    scratch.fe_values_ref[c_rg_fe].get_function_values(scratch.solution_total, scratch.solution_value_c[RG]);
    scratch.fe_values_ref[c_ip_fe].get_function_values(scratch.solution_total, scratch.solution_value_c[IP]);
    scratch.fe_values_ref[c_or_fe].get_function_values(scratch.solution_total, scratch.solution_value_c[OR]);
    scratch.fe_values_ref[c_n_fe].get_function_values(scratch.solution_total, scratch.solution_value_c[NU]);
    scratch.fe_values_ref[c_rg_fe].get_function_values(scratch.old_solution, scratch.old_solution_value_c[RG]);
    scratch.fe_values_ref[c_ip_fe].get_function_values(scratch.old_solution, scratch.old_solution_value_c[IP]);
    scratch.fe_values_ref[c_or_fe].get_function_values(scratch.old_solution, scratch.old_solution_value_c[OR]);
    scratch.fe_values_ref[c_n_fe].get_function_values(scratch.old_solution, scratch.old_solution_value_c[NU]);
    scratch.fe_values_ref[c_rg_fe].get_function_values(scratch.old_old_solution, scratch.old_old_solution_value_c[RG]);
    scratch.fe_values_ref[c_ip_fe].get_function_values(scratch.old_old_solution, scratch.old_old_solution_value_c[IP]);
    scratch.fe_values_ref[c_or_fe].get_function_values(scratch.old_old_solution, scratch.old_old_solution_value_c[OR]);
    scratch.fe_values_ref[c_n_fe].get_function_values(scratch.old_old_solution, scratch.old_old_solution_value_c[NU]);


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
        std::vector<Tensor<1,dim>> grad_cell_density(4, Tensor<1,dim>());
        std::vector<double> cell_density_value(4, 0);
        std::vector<double> old_cell_density_vlaue(4, 0);
        std::vector<double> old_old_cell_density_vlaue(4, 0);
        unsigned ii=0;
        for(; ii<grad_cell_density.size(); ++ii) {
            grad_cell_density[ii] = scratch.solution_grads_c[ii][q_point];
            cell_density_value[ii] = scratch.solution_value_c[ii][q_point];
            old_cell_density_vlaue[ii] = scratch.old_solution_value_c[ii][q_point];
            old_old_cell_density_vlaue[ii] = scratch.old_old_solution_value_c[ii][q_point];
        }
        
        
        
      lqph[q_point].update_values(scratch.solution_grads_u[q_point],grad_cell_density, cell_density_value,
                      scratch.old_solution_grads_u[q_point], scratch.old_old_solution_grads_u[q_point],
                        old_cell_density_vlaue, old_old_cell_density_vlaue,
                       time.current(), time.get_delta_t(),scratch.update_growth);
     }
  }



  template <int dim>
  void
  Solid<dim>::solve_nonlinear_timestep(BlockVector<double> &solution_delta, bool &CONVERGED)
  {
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ Current time "
      << time.current() << " @  Gestational week" << std::trunc(0.3* time.current()+4)  << "@  delta t "<<time.get_delta_t()<< std::endl;

    CONVERGED = false;
    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();


     update_qph_incremental(solution_delta,true);
    
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_number_newton_iterations;
         ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        tangent_matrix = 0.0;
        matrix_sc = 0.0;
        system_rhs = 0.0;
        assemble_system_rhs();
        get_error_residual(error_residual);

        if (newton_iteration == 0)
          error_residual_0 = error_residual;

   
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        if (newton_iteration > 0 && error_update_norm.norm <= parameters.tol_u && error_residual_norm.u<=parameters.tolerance_residual_u 
		&&  error_residual_norm.c_rg < parameters.tolerance_residual_c && error_residual_norm.c_ip < parameters.tolerance_residual_c
        && error_residual_norm.c_or < parameters.tolerance_residual_c && error_residual_norm.c_n < parameters.tolerance_residual_c)
          {
         	        std::cout << " CONVERGED! " << std::endl;
                        CONVERGED=true;
		
			break;
          }


        assemble_system_tangent();
        make_constraints(newton_iteration);
        constraints.condense(tangent_matrix, system_rhs);

        const std::pair<unsigned int, double>
        lin_solver_output = solve_linear_system(newton_update);

        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;


        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        solution_delta += newton_update;
        update_qph_incremental(solution_delta,false);

		std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
			<< std::scientific << lin_solver_output.first << "  "
			<< lin_solver_output.second << "  " << error_residual_norm.norm 
			<< "  " << error_residual_norm.u << "  " << error_residual_norm.c_rg << "  "<< error_residual_norm.c_ip << "  "
            << error_residual_norm.c_or << "  "<< error_residual_norm.c_n << "  "
            <<  error_update_norm.norm << "  " << error_update_norm.u << "  "  << error_update_norm.c_n << std::endl;
      }


    print_conv_footer();

      //{
	 //time.set_delta_t(10);
        // solution_delta = 0.0;
	 //solve_nonlinear_timestep(solution_delta);
		//}
  }



  template <int dim>
  void Solid<dim>::print_conv_header()
  {


	const unsigned int l_width = 160;
	for (unsigned int i = 0; i < l_width; ++i)
	{
		std::cout << "_";
	}
	std::cout << std::endl;

	std::cout << "             SOLVER STEP              "
				<< " |   LIN_IT   LIN_RES   RES_NORM  "
                                << " RES_U     RES_RG    RES_IP    RES_OR    RES_N     NU_NORM     NU_U     NU_N     "
				<< std::endl;

	for (unsigned int i = 0; i < l_width; ++i)
	{
		std::cout << "_";
	}
	std::cout << std::endl;
  }



  template <int dim>
  void Solid<dim>::print_conv_footer()
  {
    static const unsigned int l_width = 160;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;
   
    std::cout << "\n Relative errors:" << std::endl
              << "\t Displacment:     \t\t" << error_residual_norm.u << std::endl
              << "\t RG_Cell_Density: \t\t" << error_residual_norm.c_rg << std::endl
              << "\t IP_Cell_Density: \t\t" << error_residual_norm.c_ip << std::endl
              << "\t OR_Cell_Density: \t\t" << error_residual_norm.c_or << std::endl
              << "\t N_Cell_Density:  \t\t" << error_residual_norm.c_n << std::endl;
    std::cout << std::endl;
  }



  template <int dim>
  void Solid<dim>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);

    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_dof).l2_norm();
    error_residual.c_rg = error_res.block(c_rg_dof).l2_norm();
    error_residual.c_ip = error_res.block(c_ip_dof).l2_norm();
    error_residual.c_or = error_res.block(c_or_dof).l2_norm();
    error_residual.c_n  = error_res.block(c_nu_dof).l2_norm();

  }



  template <int dim>
  void Solid<dim>::get_error_update(const BlockVector<double> &newton_update,
                                    Errors &error_update)
  {
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_dof).l2_norm();
    error_update.c_rg = error_ud.block(c_rg_dof).l2_norm();
    error_update.c_ip = error_ud.block(c_ip_dof).l2_norm();
    error_update.c_or = error_ud.block(c_or_dof).l2_norm();
    error_update.c_n  = error_ud.block(c_nu_dof).l2_norm();
  }




  template <int dim>
  BlockVector<double>
  Solid<dim>::get_total_solution(const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
  }


template <int dim>
std::pair<double, double> Solid<dim>::get_cell_density_range_end_time_step() const
  {
       double min_cell_density = std::numeric_limits<double>::max(),
              max_cell_density = -std::numeric_limits<double>::max(),
              total_cell_density = 0;
        

	for(unsigned int i=0; i< solution_n.block(c_nu_dof).size(); ++i)
            {
                total_cell_density = solution_n.block(c_rg_dof)[i] + solution_n.block(c_ip_dof)[i] + solution_n.block(c_or_dof)[i] +solution_n.block(c_nu_dof)[i];
                  min_cell_density = std::min(total_cell_density, min_cell_density);
                  max_cell_density = std::max(total_cell_density, max_cell_density);
                 }
      
 	return std::make_pair(min_cell_density, max_cell_density);
    }

template <int dim>
 void Solid<dim>::compute_delta_t() 
{
 
    double min_delta_t = time.get_delta_t();
     min_delta_t = min_delta_t/2 ;
    
//           typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_ref.begin_active(),
//           endc=dof_handler_ref.end();
//          for(;cell !=endc ; ++cell)
//            {
//		    PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
//		    double max_speed_in_cell = 0;
//
//		    for(unsigned int q=0;q<n_q_points ; ++q)
//		       max_speed_in_cell = std::max(max_speed_in_cell, lqph[q].get_velocity().norm());
//
//                     double current_delta_t = ((parameters.c_k) * cell->diameter()) / max_speed_in_cell;
//		     min_delta_t = std::min(min_delta_t, current_delta_t);
//		    }
     
         time.set_delta_t(min_delta_t);
 
 }

  template <int dim>
  double Solid<dim>::compute_viscosity( PointHistory<dim> *lqph ,
	  const double cell_diameter, const int cell_type) const
	{
	  const double beta  = parameters.Betta * dim;

	  double max_velocity = 0;
        
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
            const Tensor<1, dim>        old_velocity_values = lqph[q].get_old_velocity_values(cell_type);
            const Tensor<1 ,dim>    old_old_velocity_values = lqph[q].get_old_old_velocity_values(cell_type);
           
	      const Tensor<1, dim> u = 
          (old_velocity_values + old_old_velocity_values) / 2;
            
	      max_velocity = std::max(std::sqrt(u * u), max_velocity);
	    }

	  return (beta * max_velocity * cell_diameter);

	}



  template <int dim>
  void Solid<dim>::assemble_system_tangent()
  {
    timer.enter_subsection("Assemble tangent matrix");
    std::cout << " ASM_K " << std::flush;

    tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values    |
                              update_gradients |
                              update_JxW_values);

    PerTaskData_K per_task_data(dofs_per_cell);
    ScratchData_K scratch_data(fe, qf_cell, uf_cell);

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_system_tangent_one_cell,
                    &Solid::copy_local_to_global_K,
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();
  }


  template <int dim>
  void Solid<dim>::copy_local_to_global_K(const PerTaskData_K &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
        matrix_sc.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
     }
  }


  template <int dim>
  void
  Solid<dim>::assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               ScratchData_K &scratch,
                                               PerTaskData_K &data)
  {
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> F_inv = lqph[q_point].get_inv_F();
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;

              if (k_group == u_dof)
                 {
                    scratch.grad_Nx_u[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                    scratch.symm_grad_Nx_u[q_point][k] = symmetrize(scratch.grad_Nx_u[q_point][k]);
                 }
                
              else if (k_group == c_rg_dof)
                {
                    scratch.grad_Nx_c[RG][q_point][k] = scratch.fe_values_ref[c_rg_fe].gradient(k, q_point)* F_inv;
                    scratch.Nx_c[RG][q_point][k] = scratch.fe_values_ref[c_rg_fe].value(k, q_point);
                 }
                
              else if (k_group == c_ip_dof)
                {
                    scratch.grad_Nx_c[IP][q_point][k] = scratch.fe_values_ref[c_ip_fe].gradient(k, q_point)* F_inv;
                    scratch.Nx_c[IP][q_point][k] = scratch.fe_values_ref[c_ip_fe].value(k, q_point);
                 }
                
              else if (k_group == c_or_dof)
                {
                    scratch.grad_Nx_c[OR][q_point][k] = scratch.fe_values_ref[c_or_fe].gradient(k, q_point)* F_inv;
                    scratch.Nx_c[OR][q_point][k] = scratch.fe_values_ref[c_or_fe].value(k, q_point);
                 }
                
              else if (k_group == c_nu_dof)
                {
                    scratch.grad_Nx_c[NU][q_point][k] = scratch.fe_values_ref[c_n_fe].gradient(k, q_point)* F_inv;
                    scratch.Nx_c[NU][q_point][k] = scratch.fe_values_ref[c_n_fe].value(k, q_point);
                 }
      
           else
             Assert(k_group <= c_nu_dof, ExcInternalError());
         }
      }

      const double nue_rg =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), RG);
      const double nue_ip =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), IP);
      const double nue_or =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), OR);
      const double nue_nu =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), NU);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
                  const SymmetricTensor<2,dim>     tau = lqph[q_point].get_tau();
                  const SymmetricTensor<4, dim>     Jc = lqph[q_point].get_elastic_tensor();
                  const Tensor<3, dim>    Jdq_rg_dF_Ft = lqph[q_point].get_Jdq_dF_Ft(RG);
                  const Tensor<3, dim>    Jdq_ip_dF_Ft = lqph[q_point].get_Jdq_dF_Ft(IP);
                  const Tensor<3, dim>    Jdq_or_dF_Ft = lqph[q_point].get_Jdq_dF_Ft(OR);
                  const Tensor<3, dim>    Jdq_nu_dF_Ft = lqph[q_point].get_Jdq_dF_Ft(NU);
                  const Tensor<2, dim>         dtau_dc = lqph[q_point].get_dtau_dc();
                  const Tensor<1, dim> Jq_nu_migration = lqph[q_point].get_Jq_migration(NU);
                  const Tensor<1, dim> Jq_rg_migration = lqph[q_point].get_Jq_migration(RG);
                  const Tensor<1, dim> Jq_ip_migration = lqph[q_point].get_Jq_migration(IP);
                  const Tensor<1, dim> Jq_or_migration = lqph[q_point].get_Jq_migration(OR);
                  const Tensor<1, dim>       Jdq_rg_dc = lqph[q_point].get_Jdq_dc(RG);
                  const Tensor<1, dim>       Jdq_ip_dc = lqph[q_point].get_Jdq_dc(IP);
                  const Tensor<1, dim>       Jdq_or_dc = lqph[q_point].get_Jdq_dc(OR);
                  const Tensor<1, dim>       Jdq_nu_dc = lqph[q_point].get_Jdq_dc(NU);
                  const double                       J = lqph[q_point].get_J();
		          const double                     J_n = lqph[q_point].get_J_old();
                  const double                    c_rg = lqph[q_point].get_c(RG);
                  const double                    c_ip = lqph[q_point].get_c(IP);
                  const double                    c_or = lqph[q_point].get_c(OR);
                  const double                    c_nu = lqph[q_point].get_c(NU);
		          const double                  c_rg_n = lqph[q_point].get_c_old(RG);
                  const double                  c_ip_n = lqph[q_point].get_c_old(IP);
                  const double                  c_or_n = lqph[q_point].get_c_old(OR);
                  const double                  c_nu_n = lqph[q_point].get_c_old(NU);
		          const double               time_step = time.get_delta_t();
          
                  const Tensor<2, dim>               I = Physics::Elasticity::StandardTensors< dim >::I;
                  const Tensor<2, dim>  Jdq_rg_dgrad_c = J * (lqph[q_point].get_dcc_r(RG) + nue_rg) * I;
                  const Tensor<1, dim> Jq_rg_diffusion = (Jdq_rg_dgrad_c * lqph[q_point].get_grad_c_spatial(RG));
                  const Tensor<1, dim>           Jq_rg = Jq_rg_migration + Jq_rg_diffusion;

                  const Tensor<2, dim>  Jdq_ip_dgrad_c = J * (lqph[q_point].get_dcc_r(IP) + nue_ip) * I;
                  const Tensor<1, dim> Jq_ip_diffusion = (Jdq_ip_dgrad_c * lqph[q_point].get_grad_c_spatial(IP));
                  const Tensor<1, dim>           Jq_ip = Jq_ip_migration + Jq_ip_diffusion;

                  const Tensor<2, dim>  Jdq_or_dgrad_c = J * (lqph[q_point].get_dcc_r(OR) + nue_or) * I;
                  const Tensor<1, dim> Jq_or_diffusion = (Jdq_or_dgrad_c * lqph[q_point].get_grad_c_spatial(OR));
                  const Tensor<1, dim>           Jq_or = Jq_or_migration + Jq_or_diffusion;

                  const Tensor<2, dim>  Jdq_nu_dgrad_c = J * (lqph[q_point].get_dcc_r(NU) + nue_nu) * I;
                  const Tensor<1, dim> Jq_nu_diffusion = (Jdq_nu_dgrad_c * lqph[q_point].get_grad_c_spatial(NU));
                  const Tensor<1, dim>           Jq_nu = Jq_nu_migration + Jq_nu_diffusion;

 

                const double JxW = scratch.fe_values_ref.JxW(q_point);        
                const std::vector<SymmetricTensor<2, dim> > &sym_grad_Nx = scratch.symm_grad_Nx_u[q_point];
                const std::vector<Tensor<2, dim> >    &grd_Nx = scratch.grad_Nx_u[q_point];
                const std::vector<Tensor<1, dim> > &grd_Nx_rg = scratch.grad_Nx_c[RG][q_point];
                const std::vector<Tensor<1, dim> > &grd_Nx_ip = scratch.grad_Nx_c[IP][q_point];
                const std::vector<Tensor<1, dim> > &grd_Nx_or = scratch.grad_Nx_c[OR][q_point];
                const std::vector<Tensor<1, dim> > &grd_Nx_nu = scratch.grad_Nx_c[NU][q_point];
                const std::vector<double>              &Nx_rg = scratch.Nx_c[RG][q_point];
                const std::vector<double>              &Nx_ip = scratch.Nx_c[IP][q_point];
                const std::vector<double>              &Nx_or = scratch.Nx_c[OR][q_point];
                const std::vector<double>              &Nx_nu = scratch.Nx_c[NU][q_point];
          



        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int i_group     = fe.system_to_base_index(i).first.first;

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int j_group     = fe.system_to_base_index(j).first.first;

  
                if ((i_group == j_group) && (i_group == u_dof)){
                  data.cell_matrix(i,j) += ( (symmetrize (transpose(grd_Nx[i]) * grd_Nx[j]) * tau) //geometrical contribution
                                            + (sym_grad_Nx[i] * Jc * sym_grad_Nx[j]) )* JxW;  }// The material contribution
        
                else if((i_group == u_dof) && (j_group == c_rg_dof))
                    data.cell_matrix(i,j) += scalar_product(grd_Nx[i], dtau_dc) * Nx_rg[j]  * JxW;
                  
                else if((i_group == u_dof) && (j_group == c_ip_dof))
                    data.cell_matrix(i,j) += scalar_product(grd_Nx[i], dtau_dc) * Nx_ip[j]  * JxW;
                  
                else if((i_group == u_dof) && (j_group == c_or_dof))
                    data.cell_matrix(i,j) += scalar_product(grd_Nx[i], dtau_dc) * Nx_or[j]  * JxW;
                  
                else if((i_group == u_dof) && (j_group == c_nu_dof))
                    data.cell_matrix(i,j) += scalar_product(grd_Nx[i], dtau_dc) * Nx_nu[j]  * JxW;

                else if((i_group == c_rg_dof) && (j_group == u_dof)){
                    data.cell_matrix(i,j) += Nx_rg[i] * (J/ time_step) * (2*c_rg -c_rg_n)  * scalar_product(I, grd_Nx[j]) *JxW; //(J/ (2*time_step)) * (6*c-4*c_n+c_n_1)
                    data.cell_matrix(i,j) -= grd_Nx[j] * Jq_rg * grd_Nx_rg[i] * JxW;
                    data.cell_matrix(i,j) += grd_Nx_ip[i] * Jq_rg * scalar_product(I, grd_Nx[j])*JxW;
                    data.cell_matrix(i,j) += scalar_product((grd_Nx_rg[i] * Jdq_rg_dF_Ft),grd_Nx[j]) * JxW;}
                  
                else if((i_group == c_ip_dof) && (j_group == u_dof)){
                    data.cell_matrix(i,j) += Nx_ip[i] * (J/ time_step) * (2*c_ip-c_ip_n)  * scalar_product(I, grd_Nx[j]) *JxW; //(J/ (2*time_step)) * (6*c-4*c_n+c_n_1)
                    data.cell_matrix(i,j) -= grd_Nx[j] * Jq_ip * grd_Nx_ip[i] * JxW;
                    data.cell_matrix(i,j) += grd_Nx_ip[i] * Jq_ip * scalar_product(I, grd_Nx[j])*JxW;
                    data.cell_matrix(i,j) += scalar_product((grd_Nx_ip[i] * Jdq_ip_dF_Ft),grd_Nx[j]) * JxW;}
                  
                else if((i_group == c_or_dof) && (j_group == u_dof)){
                    data.cell_matrix(i,j) += Nx_or[i] * (J/ time_step) * (2*c_or-c_or_n)  * scalar_product(I, grd_Nx[j]) *JxW; //(J/ (2*time_step)) * (6*c-4*c_n+c_n_1)
                    data.cell_matrix(i,j) -= grd_Nx[j] * Jq_or * grd_Nx_or[i] * JxW;
                    data.cell_matrix(i,j) += grd_Nx_or[i] * Jq_or * scalar_product(I, grd_Nx[j])*JxW;
                    data.cell_matrix(i,j) += scalar_product((grd_Nx_or[i] * Jdq_or_dF_Ft),grd_Nx[j]) * JxW;}
                  
                else if((i_group == c_nu_dof) && (j_group == u_dof)){
                    data.cell_matrix(i,j) += Nx_nu[i] * (J/ time_step) * (2*c_nu-c_nu_n)  * scalar_product(I, grd_Nx[j]) *JxW; //(J/ (2*time_step)) * (6*c-4*c_n+c_n_1)
                    data.cell_matrix(i,j) -= grd_Nx[j] * Jq_nu * grd_Nx_nu[i] * JxW;
                    data.cell_matrix(i,j) += grd_Nx_nu[i] * Jq_nu * scalar_product(I, grd_Nx[j])*JxW;
                    data.cell_matrix(i,j) += scalar_product((grd_Nx_nu[i] * Jdq_nu_dF_Ft),grd_Nx[j]) * JxW;}

                else if((i_group == j_group) && (i_group == c_rg_dof)){
                       data.cell_matrix(i,j) += Nx_rg[i] * ((2*J - J_n)/time_step) * Nx_rg[j] * JxW; //((6*J-4*J_n+J_n_1)/(2*time_step))
                       data.cell_matrix(i,j) += grd_Nx_rg[i] * Jdq_rg_dc * Nx_rg[j] * JxW;
                       data.cell_matrix(i,j) += grd_Nx_rg[i] * Jdq_rg_dgrad_c * grd_Nx_rg[j] * JxW;}
                  
                else if((i_group == j_group) && (i_group == c_ip_dof)) {
                           data.cell_matrix(i,j) += Nx_ip[i] * ((2*J - J_n)/time_step) * Nx_ip[j] * JxW; //((6*J-4*J_n+J_n_1)/(2*time_step))
                           data.cell_matrix(i,j) += grd_Nx_ip[i] * Jdq_ip_dc * Nx_ip[j] * JxW;
                           data.cell_matrix(i,j) += grd_Nx_ip[i] * Jdq_ip_dgrad_c * grd_Nx_ip[j] * JxW;}

                else if((i_group == j_group) && (i_group == c_or_dof)) {
                           data.cell_matrix(i,j) += Nx_or[i] * ((2*J - J_n)/time_step) * Nx_or[j] * JxW; //((6*J-4*J_n+J_n_1)/(2*time_step))
                           data.cell_matrix(i,j) += grd_Nx_or[i] * Jdq_or_dc * Nx_or[j] * JxW;
                           data.cell_matrix(i,j) += grd_Nx_or[i] * Jdq_or_dgrad_c * grd_Nx_or[j] * JxW;}

                else if((i_group == j_group) && (i_group == c_nu_dof)) {
                       data.cell_matrix(i,j) += Nx_nu[i] * ((2*J - J_n)/time_step) * Nx_nu[j] * JxW; //((6*J-4*J_n+J_n_1)/(2*time_step))
                       data.cell_matrix(i,j) += grd_Nx_nu[i] * Jdq_nu_dc * Nx_nu[j] * JxW;
                       data.cell_matrix(i,j) += grd_Nx_nu[i] * Jdq_nu_dgrad_c * grd_Nx_nu[j] * JxW;}
                else
                  Assert((i_group <= c_nu_dof) && (j_group <= c_nu_dof),
                         ExcInternalError());
              }
          }
      }


  }


  template <int dim>
  void Solid<dim>::assemble_system_rhs()
  {
    timer.enter_subsection("Assemble system right-hand side");
    std::cout << " ASM_R " << std::flush;
    system_rhs = 0.0;
    const UpdateFlags uf_cell(update_values |
                              update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_normal_vectors |
                              update_JxW_values);

    PerTaskData_RHS per_task_data(dofs_per_cell);
    ScratchData_RHS scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face);
    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_system_rhs_one_cell,
                    &Solid::copy_local_to_global_rhs,
                    scratch_data,
                    per_task_data);
    timer.leave_subsection();
  }



  template <int dim>
  void Solid<dim>::copy_local_to_global_rhs(const PerTaskData_RHS &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      system_rhs(data.local_dof_indices[i]) += data.cell_rhs(i);
  }



  template <int dim>
  void
  Solid<dim>::assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           ScratchData_RHS &scratch,
                                           PerTaskData_RHS &data)
  {
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);


    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const Tensor<2, dim> F_inv = lqph[q_point].get_inv_F();
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
             {
               const unsigned int k_group = fe.system_to_base_index(k).first.first;

               if (k_group == u_dof)
                  {
                     scratch.grad_Nx_u[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                     scratch.symm_grad_Nx_u[q_point][k] = symmetrize(scratch.grad_Nx_u[q_point][k]);
                  }
                 
               else if (k_group == c_rg_dof)
                 {
                     scratch.grad_Nx_c[RG][q_point][k] = scratch.fe_values_ref[c_rg_fe].gradient(k, q_point)* F_inv;
                     scratch.Nx_c[RG][q_point][k] = scratch.fe_values_ref[c_rg_fe].value(k, q_point);
                  }
                 
               else if (k_group == c_ip_dof)
                 {
                     scratch.grad_Nx_c[IP][q_point][k] = scratch.fe_values_ref[c_ip_fe].gradient(k, q_point)* F_inv;
                     scratch.Nx_c[IP][q_point][k] = scratch.fe_values_ref[c_ip_fe].value(k, q_point);
                  }
                 
               else if (k_group == c_or_dof)
                 {
                     scratch.grad_Nx_c[OR][q_point][k] = scratch.fe_values_ref[c_or_fe].gradient(k, q_point)* F_inv;
                     scratch.Nx_c[OR][q_point][k] = scratch.fe_values_ref[c_or_fe].value(k, q_point);
                  }
                 
               else if (k_group == c_nu_dof)
                 {
                     scratch.grad_Nx_c[NU][q_point][k] = scratch.fe_values_ref[c_n_fe].gradient(k, q_point)* F_inv;
                     scratch.Nx_c[NU][q_point][k] = scratch.fe_values_ref[c_n_fe].value(k, q_point);
                  }
       
            else
              Assert(k_group <= c_nu_dof, ExcInternalError());
          }
      }
 
      const double nue_rg =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), RG);
      const double nue_ip =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), IP);
      const double nue_or =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), OR);
      const double nue_nu =  compute_viscosity(reinterpret_cast<PointHistory<dim>*>(cell->user_pointer()), cell->diameter(), NU);
      
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
          
        const SymmetricTensor<2,dim>     tau = lqph[q_point].get_tau();
        const Tensor<1, dim> Jq_nu_migration = lqph[q_point].get_Jq_migration(NU);
        const Tensor<1, dim> Jq_rg_migration = lqph[q_point].get_Jq_migration(RG);
        const Tensor<1, dim> Jq_ip_migration = lqph[q_point].get_Jq_migration(IP);
        const Tensor<1, dim> Jq_or_migration = lqph[q_point].get_Jq_migration(OR);
        const double                  F_c_rg = lqph[q_point].get_F_c(RG);
        const double                  F_c_ip = lqph[q_point].get_F_c(IP);
        const double                  F_c_or = lqph[q_point].get_F_c(OR);
        const double                  F_c_nu = lqph[q_point].get_F_c(NU);
        const double                       J = lqph[q_point].get_J();
	const double                     J_n = lqph[q_point].get_J_old();
        const double                    c_rg = lqph[q_point].get_c(RG);
        const double                    c_ip = lqph[q_point].get_c(IP);
        const double                    c_or = lqph[q_point].get_c(OR);
        const double                    c_nu = lqph[q_point].get_c(NU);
        const double                  c_rg_n = lqph[q_point].get_c_old(RG);
        const double                  c_ip_n = lqph[q_point].get_c_old(IP);
        const double                  c_or_n = lqph[q_point].get_c_old(OR);
        const double                  c_nu_n = lqph[q_point].get_c_old(NU);
        const double               time_step = time.get_delta_t();
     
		  const Tensor<2, dim>               I = Physics::Elasticity::StandardTensors< dim >::I;
                  const Tensor<2, dim>  Jdq_rg_dgrad_c = J * (lqph[q_point].get_dcc_r(RG) + nue_rg) * I;
                  const Tensor<1, dim> Jq_rg_diffusion = (Jdq_rg_dgrad_c * lqph[q_point].get_grad_c_spatial(RG));
                  const Tensor<1, dim>           Jq_rg = Jq_rg_migration + Jq_rg_diffusion;

                  const Tensor<2, dim>  Jdq_ip_dgrad_c = J * (lqph[q_point].get_dcc_r(IP) + nue_ip) * I;
                  const Tensor<1, dim> Jq_ip_diffusion = (Jdq_ip_dgrad_c * lqph[q_point].get_grad_c_spatial(IP));
                  const Tensor<1, dim>           Jq_ip = Jq_ip_migration + Jq_ip_diffusion;

                  const Tensor<2, dim>  Jdq_or_dgrad_c = J * (lqph[q_point].get_dcc_r(OR) + nue_or) * I;
                  const Tensor<1, dim> Jq_or_diffusion = (Jdq_or_dgrad_c * lqph[q_point].get_grad_c_spatial(OR));
                  const Tensor<1, dim>           Jq_or = Jq_or_migration + Jq_or_diffusion;

                  const Tensor<2, dim>  Jdq_nu_dgrad_c = J * (lqph[q_point].get_dcc_r(NU) + nue_nu) * I;
                  const Tensor<1, dim> Jq_nu_diffusion = (Jdq_nu_dgrad_c * lqph[q_point].get_grad_c_spatial(NU));
                  const Tensor<1, dim>           Jq_nu = Jq_nu_migration + Jq_nu_diffusion;
          
          const double                  JxW = scratch.fe_values_ref.JxW(q_point);
          const std::vector<SymmetricTensor<2, dim> > &sym_grad_Nx = scratch.symm_grad_Nx_u[q_point];
          const std::vector<Tensor<1, dim> > &grd_Nx_rg = scratch.grad_Nx_c[RG][q_point];
          const std::vector<Tensor<1, dim> > &grd_Nx_ip = scratch.grad_Nx_c[IP][q_point];
          const std::vector<Tensor<1, dim> > &grd_Nx_or = scratch.grad_Nx_c[OR][q_point];
          const std::vector<Tensor<1, dim> > &grd_Nx_nu = scratch.grad_Nx_c[NU][q_point];
          const std::vector<double>              &Nx_rg = scratch.Nx_c[RG][q_point];
          const std::vector<double>              &Nx_ip = scratch.Nx_c[IP][q_point];
          const std::vector<double>              &Nx_or = scratch.Nx_c[OR][q_point];
          const std::vector<double>              &Nx_nu = scratch.Nx_c[NU][q_point];


        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
		     const unsigned int i_group     = fe.system_to_base_index(i).first.first;
          	 if(i_group == u_dof)
		          data.cell_rhs(i) -= (sym_grad_Nx[i] * tau) * JxW;
              
             else if(i_group == c_rg_dof){
                 data.cell_rhs(i) -= Nx_rg[i] * ((J - J_n)/time_step) * c_rg * JxW;  //((3*J-4*J_n+J_n_1)/(2*time_step))
                 data.cell_rhs(i) -= J * Nx_rg[i] * ((c_rg - c_rg_n)/time_step) * JxW; //((3*c-4*c_n+c_n_1)/(2*time_step))
                 data.cell_rhs(i) -= grd_Nx_rg[i] * Jq_rg * JxW;
                 data.cell_rhs(i) += Nx_rg[i] * F_c_rg * JxW;}
              
             else if(i_group == c_ip_dof){
                  data.cell_rhs(i) -= Nx_ip[i] * ((J - J_n)/time_step) * c_ip * JxW;  //((3*J-4*J_n+J_n_1)/(2*time_step))
                  data.cell_rhs(i) -= J * Nx_ip[i] * ((c_ip - c_ip_n)/time_step) * JxW; //((3*c-4*c_n+c_n_1)/(2*time_step))
                  data.cell_rhs(i) -= grd_Nx_ip[i] * Jq_ip * JxW;
                  data.cell_rhs(i) += Nx_ip[i] * F_c_ip * JxW;}
              
             else if(i_group == c_or_dof){
                  data.cell_rhs(i) -= Nx_or[i] * ((J - J_n)/time_step) * c_or * JxW;  //((3*J-4*J_n+J_n_1)/(2*time_step))
                  data.cell_rhs(i) -= J * Nx_or[i] * ((c_or - c_or_n)/time_step) * JxW; //((3*c-4*c_n+c_n_1)/(2*time_step))
                  data.cell_rhs(i) -= grd_Nx_or[i] * Jq_or * JxW;
                  data.cell_rhs(i) += Nx_or[i] * F_c_or * JxW;}
              
             else if(i_group == c_nu_dof){
                  data.cell_rhs(i) -= Nx_nu[i] * ((J - J_n)/time_step) * c_nu * JxW;  //((3*J-4*J_n+J_n_1)/(2*time_step))
                  data.cell_rhs(i) -= J * Nx_nu[i] * ((c_nu - c_nu_n)/time_step) * JxW; //((3*c-4*c_n+c_n_1)/(2*time_step))
                  data.cell_rhs(i) -= grd_Nx_nu[i] * Jq_nu * JxW;
                  data.cell_rhs(i) += Nx_nu[i] * F_c_nu * JxW;}
      }
 }

 /* for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      if (cell->face(face)->at_boundary() == true
          && cell->face(face)->boundary_id() == 2)
        {
          scratch.fe_face_values_ref.reinit(cell, face);

          for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
               ++f_q_point)
            {
              const Tensor<1, dim> &N =
                scratch.fe_face_values_ref.normal_vector(f_q_point);


              static const double  p0        = -0.001
                                               /
                                               (parameters.scale * parameters.scale);
              const double         time_ramp = (time.current() / time.end());
              const double         pressure  = p0 * 1 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int i_group =
                    fe.system_to_base_index(i).first.first;

                  if (i_group == u_dof)
                    {
                      const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                      const double Ni =
                        scratch.fe_face_values_ref.shape_value(i,
                                                               f_q_point);
                      const double JxW = scratch.fe_face_values_ref.JxW(
                                           f_q_point);

                      data.cell_rhs(i) += (Ni * traction[component_i])
                                          * JxW;
                    }
                }
            }
        }*/
  }



  template <int dim>
  void Solid<dim>::make_constraints(const int &it_nr)
{
  std::cout << " CST " << std::flush;

    if (dim==2){
        if (it_nr > 1)
            return;
        constraints.clear();
        const bool apply_dirichlet_bc = (it_nr == 0);
        const FEValuesExtractors::Vector displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);
        const FEValuesExtractors::Scalar x_displacement(0);
        
        
        {
            const int boundary_id = 2;
            
            if (apply_dirichlet_bc == true)
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(y_displacement));
                
            }
            else
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(y_displacement));
                
            }
        }
        {
            const int boundary_id = 3;
            
            if (apply_dirichlet_bc == true)
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(x_displacement));
                
            }
            else
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(x_displacement));
                
            }
        }
        {
            const int boundary_id = 0;
            if (apply_dirichlet_bc == true)
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(displacement));
            }
            else
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(displacement));
            }
        }
    }
  
    else if (dim==3){
        if (it_nr > 1)
          return;
        constraints.clear();
        const bool apply_dirichlet_bc = (it_nr == 0);
        const FEValuesExtractors::Vector displacement(0);
        const FEValuesExtractors::Scalar x_displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);
        const FEValuesExtractors::Scalar z_displacement(2);
        
        
        {
                const int boundary_id = 2;

            if (apply_dirichlet_bc == true)
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(y_displacement));
            
            }
            else
            {
                 VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                                                        boundary_id,
                                                         ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(y_displacement));

            }
        } // for half geomtery

        {
            const int boundary_id = 0;
            if (apply_dirichlet_bc == true)
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(displacement));
            }
            else
            {
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(displacement));
            }
        }
        
    }

  constraints.close();
}


  template <int dim>
  void Solid<dim>::assemble_sc()
  {

    std::cout << " sc " << std::flush;
    PerTaskData_SC per_task_data(dofs_per_cell, element_indices_u.size(), element_indices_c_rg.size(),
                                 element_indices_c_ip.size(),  element_indices_c_or.size(), element_indices_c_nu.size());
    ScratchData_SC scratch_data;
    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    *this,
                    &Solid::assemble_sc_one_cell,
                    &Solid::copy_local_to_global_sc,
                    scratch_data,
                    per_task_data);

  }
  template <int dim>
  void Solid<dim>::copy_local_to_global_sc(const PerTaskData_SC &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
  }

template <int dim>
void Solid<dim>::assemble_sc_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                   ScratchData_SC &scratch,
                                   PerTaskData_SC &data)
{

        data.reset();
        scratch.reset();
        cell->get_dof_indices(data.local_dof_indices);

        data.k_orig.extract_submatrix_from  (tangent_matrix, data.local_dof_indices, data.local_dof_indices);
        data.k_rg_u.extract_submatrix_from  (data.k_orig, element_indices_c_rg, element_indices_u);
        data.k_u_rg.extract_submatrix_from  (data.k_orig, element_indices_u,    element_indices_c_rg);
        data.k_rg_rg.extract_submatrix_from (data.k_orig, element_indices_c_rg, element_indices_c_rg);
        data.k_ip_u.extract_submatrix_from  (data.k_orig, element_indices_c_ip, element_indices_u);
        data.k_u_ip.extract_submatrix_from  (data.k_orig, element_indices_u,    element_indices_c_ip);
        data.k_ip_ip.extract_submatrix_from (data.k_orig, element_indices_c_ip, element_indices_c_ip);
        data.k_or_u.extract_submatrix_from  (data.k_orig, element_indices_c_or, element_indices_u);
        data.k_u_or.extract_submatrix_from  (data.k_orig, element_indices_u,    element_indices_c_or);
        data.k_or_or.extract_submatrix_from (data.k_orig, element_indices_c_or, element_indices_c_or);
        data.k_nu_u.extract_submatrix_from  (data.k_orig, element_indices_c_nu, element_indices_u);
        data.k_u_nu.extract_submatrix_from  (data.k_orig, element_indices_u,    element_indices_c_nu);
        data.k_nu_nu.extract_submatrix_from (data.k_orig, element_indices_c_nu, element_indices_c_nu);
    

        // k_cc^-1
        data.k_rg_rg_inv.invert(data.k_rg_rg);
        data.k_ip_ip_inv.invert(data.k_ip_ip);
        data.k_or_or_inv.invert(data.k_or_or);
        data.k_nu_nu_inv.invert(data.k_nu_nu);


        // k_B = k_cc^-1 * k_cu
        data.k_rg_rg_inv.mmult(data.k_B_rg, data.k_rg_u);
        data.k_ip_ip_inv.mmult(data.k_B_ip, data.k_ip_u);
        data.k_or_or_inv.mmult(data.k_B_or, data.k_or_u);
        data.k_nu_nu_inv.mmult(data.k_B_nu, data.k_nu_u);
    


        // k_bar = k_uc * k_B
        data.k_u_rg.mmult(data.k_bar_rg, data.k_B_rg);
        data.k_u_ip.mmult(data.k_bar_ip, data.k_B_ip);
        data.k_u_or.mmult(data.k_bar_or, data.k_B_or);
        data.k_u_nu.mmult(data.k_bar_nu, data.k_B_nu);


    
        // k_bar = - k_bar
        data.k_bar_rg *= -1;
        data.k_bar_ip *= -1;
        data.k_bar_or *= -1;
        data.k_bar_nu *= -1;



        // k_cc^-1 = k_cc^-1 - k_cc
        data.k_rg_rg_inv.add(-1, data.k_rg_rg);
        data.k_ip_ip_inv.add(-1, data.k_ip_ip);
        data.k_or_or_inv.add(-1, data.k_or_or);
        data.k_nu_nu_inv.add(-1, data.k_nu_nu);



        // k_B = k_B - k_cu
        data.k_B_rg.add(-1, data.k_rg_u);
        data.k_B_ip.add(-1, data.k_ip_u);
        data.k_B_or.add(-1, data.k_or_u);
        data.k_B_nu.add(-1, data.k_nu_u);


        data.k_bar_rg.scatter_matrix_to (element_indices_u, element_indices_u, data.cell_matrix);
        data.k_bar_ip.scatter_matrix_to (element_indices_u, element_indices_u, data.cell_matrix);
        data.k_bar_or.scatter_matrix_to (element_indices_u, element_indices_u, data.cell_matrix);
        data.k_bar_nu.scatter_matrix_to (element_indices_u, element_indices_u, data.cell_matrix);
    
        data.k_B_rg.scatter_matrix_to (element_indices_c_rg, element_indices_u, data.cell_matrix);
        data.k_B_ip.scatter_matrix_to (element_indices_c_ip, element_indices_u, data.cell_matrix);
        data.k_B_or.scatter_matrix_to (element_indices_c_or, element_indices_u, data.cell_matrix);
        data.k_B_nu.scatter_matrix_to (element_indices_c_nu, element_indices_u, data.cell_matrix);

        data.k_rg_rg_inv.scatter_matrix_to (element_indices_c_rg, element_indices_c_rg, data.cell_matrix);
        data.k_ip_ip_inv.scatter_matrix_to (element_indices_c_ip, element_indices_c_ip, data.cell_matrix);
        data.k_or_or_inv.scatter_matrix_to (element_indices_c_or, element_indices_c_or, data.cell_matrix);
        data.k_nu_nu_inv.scatter_matrix_to (element_indices_c_nu, element_indices_c_nu, data.cell_matrix);

 
} 

template <int dim>
std::pair<unsigned int, double>
Solid<dim>::solve_linear_system(BlockVector<double> &newton_update)
{

       	unsigned int lin_it = 0;
        double lin_res = 0.0;

        

          assemble_sc();
	
        BlockVector<double> A(dofs_per_block);
	    BlockVector<double> B_1(dofs_per_block);
        BlockVector<double> B_2(dofs_per_block);
        BlockVector<double> B_3(dofs_per_block);
        BlockVector<double> B_4(dofs_per_block);
        BlockVector<double> C(dofs_per_block);

        // A_c = k_cc^-1 * F_c
        tangent_matrix.block(c_rg_dof, c_rg_dof).vmult(A.block(c_rg_dof), system_rhs.block(c_rg_dof));
        tangent_matrix.block(c_ip_dof, c_ip_dof).vmult(A.block(c_ip_dof), system_rhs.block(c_ip_dof));
        tangent_matrix.block(c_or_dof, c_or_dof).vmult(A.block(c_or_dof), system_rhs.block(c_or_dof));
        tangent_matrix.block(c_nu_dof, c_nu_dof).vmult(A.block(c_nu_dof), system_rhs.block(c_nu_dof));
    
        // B_u = k_uc * A_c
        tangent_matrix.block(u_dof, c_rg_dof).vmult(B_1.block(u_dof), A.block(c_rg_dof));
        tangent_matrix.block(u_dof, c_ip_dof).vmult(B_2.block(u_dof), A.block(c_ip_dof));
        tangent_matrix.block(u_dof, c_or_dof).vmult(B_3.block(u_dof), A.block(c_or_dof));
        tangent_matrix.block(u_dof, c_nu_dof).vmult(B_4.block(u_dof), A.block(c_nu_dof));
    
        // F_con_u = F_u - B
        system_rhs.block(u_dof) -= (B_1.block(u_dof) + B_2.block(u_dof) + B_3.block(u_dof) +B_4.block(u_dof));

         

	std::cout << " SLV " << std::flush;
	if (parameters.solver_type == "CG")
	{
		const int solver_its = tangent_matrix.block(u_dof, u_dof).m() * parameters.multiplier_max_iterations_linear_solver;
		const double tol_sol = 1e-9*system_rhs.block(u_dof).l2_norm();

		SolverControl solver_control(solver_its, tol_sol);

		GrowingVectorMemory<Vector<double> > GVM;
		SolverCG<Vector<double> > solver_CG(solver_control, GVM);

		PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor", 1.2);

	        preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

		solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
						newton_update.block(u_dof),
						system_rhs.block(u_dof),
						preconditioner);
		lin_it = solver_control.last_step();
		lin_res = solver_control.last_value();
	}
        
        else if (parameters.solver_type == "Direct")
        {
              SparseDirectUMFPACK A_direct;
	      A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
	      A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));
	      lin_it = 1;
	      lin_res = 0.0;
	    }
	else
	{
		
		Assert (false, ExcMessage("Linear solver type not implemented"));
	}

	constraints.distribute(newton_update);

     {
	
          // C_c = K_cu * delt_u
          matrix_sc.block(c_rg_dof ,u_dof).vmult(C.block(c_rg_dof), newton_update.block(u_dof));
          matrix_sc.block(c_ip_dof ,u_dof).vmult(C.block(c_ip_dof), newton_update.block(u_dof));
          matrix_sc.block(c_or_dof ,u_dof).vmult(C.block(c_or_dof), newton_update.block(u_dof));
          matrix_sc.block(c_nu_dof ,u_dof).vmult(C.block(c_nu_dof), newton_update.block(u_dof));

         // F_c = F_c - C_c
         system_rhs.block(c_rg_dof) -= C.block(c_rg_dof);
         system_rhs.block(c_ip_dof) -= C.block(c_ip_dof);
         system_rhs.block(c_or_dof) -= C.block(c_or_dof);
         system_rhs.block(c_nu_dof) -= C.block(c_nu_dof);

 
         if (parameters.solver_type == "CG")
	{
        {	const int solver_its = matrix_sc.block(c_rg_dof, c_rg_dof).m() * parameters.multiplier_max_iterations_linear_solver;
            const double tol_sol = 1e-12*system_rhs.block(c_rg_dof).l2_norm();
            
            SolverControl solver_control(solver_its, tol_sol);
            
            GrowingVectorMemory<Vector<double> > GVM;
            SolverCG<Vector<double> > solver_CG(solver_control, GVM);
            
            PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor", 1.2);
            
            preconditioner.use_matrix(matrix_sc.block(c_rg_dof, c_rg_dof));
            
            solver_CG.solve(matrix_sc.block(c_rg_dof, c_rg_dof),
                            newton_update.block(c_rg_dof),
                            system_rhs.block(c_rg_dof),
                            preconditioner);
            lin_it = solver_control.last_step();
            lin_res = solver_control.last_value();
        }
        
        {    const int solver_its = matrix_sc.block(c_ip_dof, c_ip_dof).m() * parameters.multiplier_max_iterations_linear_solver;
            const double tol_sol = 1e-12*system_rhs.block(c_ip_dof).l2_norm();
            
            SolverControl solver_control(solver_its, tol_sol);
            
            GrowingVectorMemory<Vector<double> > GVM;
            SolverCG<Vector<double> > solver_CG(solver_control, GVM);
            
            PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor", 1.2);
            
            preconditioner.use_matrix(matrix_sc.block(c_ip_dof, c_ip_dof));
            
            solver_CG.solve(matrix_sc.block(c_ip_dof, c_ip_dof),
                            newton_update.block(c_ip_dof),
                            system_rhs.block(c_ip_dof),
                            preconditioner);
            lin_it = solver_control.last_step();
            lin_res = solver_control.last_value();
        }
        
        {    const int solver_its = matrix_sc.block(c_or_dof, c_or_dof).m() * parameters.multiplier_max_iterations_linear_solver;
            const double tol_sol = 1e-12*system_rhs.block(c_or_dof).l2_norm();
            
            SolverControl solver_control(solver_its, tol_sol);
            
            GrowingVectorMemory<Vector<double> > GVM;
            SolverCG<Vector<double> > solver_CG(solver_control, GVM);
            
            PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor", 1.2);
            
            preconditioner.use_matrix(matrix_sc.block(c_or_dof, c_or_dof));
            
            solver_CG.solve(matrix_sc.block(c_or_dof, c_or_dof),
                            newton_update.block(c_or_dof),
                            system_rhs.block(c_or_dof),
                            preconditioner);
            lin_it = solver_control.last_step();
            lin_res = solver_control.last_value();
        }
        
        {    const int solver_its = matrix_sc.block(c_nu_dof, c_nu_dof).m() * parameters.multiplier_max_iterations_linear_solver;
            const double tol_sol = 1e-12*system_rhs.block(c_nu_dof).l2_norm();
            
            SolverControl solver_control(solver_its, tol_sol);
            
            GrowingVectorMemory<Vector<double> > GVM;
            SolverCG<Vector<double> > solver_CG(solver_control, GVM);
            
            PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor", 1.2);
            
            preconditioner.use_matrix(matrix_sc.block(c_nu_dof, c_nu_dof));
            
            solver_CG.solve(matrix_sc.block(c_nu_dof, c_nu_dof),
                            newton_update.block(c_nu_dof),
                            system_rhs.block(c_nu_dof),
                            preconditioner);
            lin_it = solver_control.last_step();
            lin_res = solver_control.last_value();
        }
	}
        
        else if (parameters.solver_type == "Direct")
        {
            {
                SparseDirectUMFPACK B_1_direct;
                B_1_direct.initialize(matrix_sc.block(c_rg_dof, c_rg_dof));
                B_1_direct.vmult(newton_update.block(c_rg_dof), system_rhs.block(c_rg_dof));
                lin_it = 1;
                lin_res = 0.0;
            }
            
            {
                SparseDirectUMFPACK B_2_direct;
                B_2_direct.initialize(matrix_sc.block(c_ip_dof, c_ip_dof));
                B_2_direct.vmult(newton_update.block(c_ip_dof), system_rhs.block(c_ip_dof));
                lin_it = 1;
                lin_res = 0.0;
            }
            
            {
                SparseDirectUMFPACK B_3_direct;
                B_3_direct.initialize(matrix_sc.block(c_or_dof, c_or_dof));
                B_3_direct.vmult(newton_update.block(c_or_dof), system_rhs.block(c_or_dof));
                lin_it = 1;
                lin_res = 0.0;
            }
            
            {
                SparseDirectUMFPACK B_4_direct;
                B_4_direct.initialize(matrix_sc.block(c_nu_dof, c_nu_dof));
                B_4_direct.vmult(newton_update.block(c_nu_dof), system_rhs.block(c_nu_dof));
                lin_it = 1;
                lin_res = 0.0;
            }
	    }
	else
	{
		
		Assert (false, ExcMessage("Linear solver type not implemented"));
	}                                           
                                          
           constraints.distribute(newton_update);
         }


	return std::make_pair(lin_it, lin_res);
}

    
template <int dim>
void Solid<dim>::output_results()
{

    
DataOut<dim> data_out;
std::vector<DataComponentInterpretation::DataComponentInterpretation>
data_component_interpretation(dim,  DataComponentInterpretation::component_is_part_of_vector);
data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);


std::vector<std::string> solution_name(dim, "displacement");
solution_name.push_back("RG_cell_desnity");
solution_name.push_back("IP_cell_desnity");
solution_name.push_back("ORG_cell_desnity");
solution_name.push_back("N_cell_desnity");


data_out.attach_dof_handler(dof_handler_ref);
data_out.add_data_vector(solution_n,
                         solution_name,
                         DataOut<dim>::type_dof_data,
                         data_component_interpretation);

 BlockVector<double> stresses_projected(dofs_per_block);
 Vector<double> stresses_averaged(triangulation.n_active_cells());
 compute_stress_projection_and_average (stresses_projected,stresses_averaged);

 BlockVector<double> sptial_density_grads(dofs_per_block);
 sptial_density_grads_projection(sptial_density_grads);
  
 BlockVector<double> growth_norm(dofs_per_block);
 compute_growth_norm_projection(growth_norm);

 BlockVector<double> Stiffness(dofs_per_block);
 compute_Stiffness_projection(Stiffness);

BlockVector<double>    growth_factor(dofs_per_block);
growth_factor_projection (growth_factor);
  
//BlockVector<double>    source_terms(dofs_per_block);
//source_terms_projection (source_terms);
  
BlockVector<double>    velocity_values(dofs_per_block);
velocity_projection (velocity_values);
  
BlockVector<double> diffusion_values(dofs_per_block);
  diffusion_projection(diffusion_values);
    
BlockVector<double> total_cell_density(dofs_per_block);
 
std::vector<std::string> solution_name2(dim, "vMStress");
    solution_name2.push_back("non11");
    solution_name2.push_back("non12");
    solution_name2.push_back("non13");
    solution_name2.push_back("non14");


std::vector<std::string> solution_name3(1, "vMStress_averaged");
  
std::vector<std::string>  solution_name4(dim , "sptial_density_grads");
      solution_name4.push_back("non21");
      solution_name4.push_back("non22");
      solution_name4.push_back("non23");
      solution_name4.push_back("non24");


std::vector<std::string>  solution_name5(dim , "growth_factor");
    solution_name5.push_back("non31");
    solution_name5.push_back("non32");
    solution_name5.push_back("non33");
    solution_name5.push_back("non34");

std::vector<std::string>  solution_name6(dim , "growth_norm");
    solution_name6.push_back("non41");
    solution_name6.push_back("non42");
    solution_name6.push_back("non43");
    solution_name6.push_back("non44");

std::vector<std::string>  solution_name7(dim , "Stiffness");
    solution_name7.push_back("non51");
    solution_name7.push_back("non52");
    solution_name7.push_back("non53");
    solution_name7.push_back("non54");
  
//std::vector<std::string>  solution_name8(dim , "source_terms");
//    solution_name8.push_back("non61");
//    solution_name8.push_back("non62");
//    solution_name8.push_back("non63");
//    solution_name8.push_back("non64");

std::vector<std::string>  solution_name9(dim , "velocity");
  solution_name9.push_back("non71");
  solution_name9.push_back("non72");
  solution_name9.push_back("non73");
  solution_name9.push_back("non74");
    
std::vector<std::string>  solution_name10(dim , "diffusion");
 solution_name10.push_back("non81");
 solution_name10.push_back("non82");
 solution_name10.push_back("non83");
 solution_name10.push_back("non84");
    
    std::vector<std::string> solution_name11(dim, "non90");
    solution_name11.push_back("non91");
    solution_name11.push_back("non92");
    solution_name11.push_back("non93");
    solution_name11.push_back("total_cell_density");
    
data_out.add_data_vector(stresses_projected,
                         solution_name2,
                         DataOut<dim>::type_dof_data,
                         data_component_interpretation);

data_out.add_data_vector(stresses_averaged,
                         solution_name3,
                         DataOut<dim>::type_cell_data);

  data_out.add_data_vector(sptial_density_grads,
                           solution_name4,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  
data_out.add_data_vector(growth_factor,
                         solution_name5,
                         DataOut<dim>::type_dof_data,
                         data_component_interpretation);
  
data_out.add_data_vector(growth_norm,
                         solution_name6,
                         DataOut<dim>::type_dof_data,
                         data_component_interpretation);

data_out.add_data_vector(Stiffness,
                         solution_name7,
                         DataOut<dim>::type_dof_data,
                         data_component_interpretation);
  
//data_out.add_data_vector(source_terms,
//                        solution_name8,
//                        DataOut<dim>::type_dof_data,
//                        data_component_interpretation);
  
data_out.add_data_vector(velocity_values,
                        solution_name9,
                        DataOut<dim>::type_dof_data,
                        data_component_interpretation);
  
data_out.add_data_vector(diffusion_values,
                            solution_name10,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
    



for(unsigned int i=0; i< solution_n.block(c_nu_dof).size(); ++i)
{
    total_cell_density.block(c_nu_dof)[i] = solution_n.block(c_rg_dof)[i] + solution_n.block(c_ip_dof)[i] + solution_n.block(c_or_dof)[i] +solution_n.block(c_nu_dof)[i];
}

    data_out.add_data_vector(total_cell_density,
                                solution_name11,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);

Vector<double> soln(solution_n.size());
for (unsigned int i = 0; i < soln.size(); ++i)
  soln(i) = solution_n(i);


Vector<double> growth_values(triangulation.n_active_cells());
//Vector<double> c_k_vlaues(triangulation.n_active_cells());

 typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_ref.begin_active(),
endc=dof_handler_ref.end();
for(;cell !=endc ; ++cell)
    {
        PointHistory<dim> *lqph = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        //double max_speed_in_cell = 0;
        for(unsigned int q=0;q<n_q_points ; ++q){
           growth_values[cell->active_cell_index()]+=lqph[q].get_growth_factor_t()/n_q_points;
          // max_speed_in_cell = std::max(max_speed_in_cell, lqph[q].get_velocity().norm());
           }
         //c_k_vlaues[cell->active_cell_index()]= (0.5 * cell->diameter()) / max_speed_in_cell;
        }
 
/*  stretch_max[2] = stretch_max[1];
  stretch_max[1] = stretch_max[0];
 for(unsigned int t=0; t<growth_values.size(); ++t)
  {
       if(growth_values[t] > stretch_max[0])
        stretch_max[0] = growth_values[t];
         }
*/

data_out.add_data_vector(growth_values, "growth_stretch" ,DataOut<dim>::type_cell_data);
// data_out.add_data_vector(c_k_vlaues, "proposed_time_step" ,DataOut<dim>::type_cell_data);


MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
data_out.build_patches(q_mapping ,degree );





std::string name ("Output_3_" + Utilities::int_to_string(time.get_timestep(), 2));

std::ofstream output(( name + ".vtk"));

data_out.write_vtk(output);

}


template<int dim>
void Solid<dim>::compute_stress_projection_and_average( BlockVector<double> &stress_projected,
							Vector<double> &stress_averaged)
{
	std::cout << " Project Stresses " << std::flush;


    AffineConstraints<double>  constraints_1;
	constraints_1.clear();
	constraints_1.close();
	
	BlockVector<double>   			stress_projected_rhs(dofs_per_block);
	BlockSparseMatrix<double>		stress_projected_matrix;


	stress_projected_matrix.reinit (sparsity_pattern);

	

	FEValues<dim> fe_values (fe, qf_cell, update_values | update_gradients | update_JxW_values);
	
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	AssertThrow (stress_projected.size() == dof_handler_ref.n_dofs()
				&& stress_averaged.size() == triangulation.n_active_cells(),
				ExcMessage("Compute stress projection - sizes of vectors do not match")
				);
	stress_projected=0;
	stress_averaged=0;
	stress_projected_rhs=0;
	stress_projected_matrix=0;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		
	
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
									endc = dof_handler_ref.end();
												
	
	for(unsigned int cell_counter=0; cell!=endc;++cell,++cell_counter)
	   {
		cell_matrix=0.0;
		cell_rhs=0.0;
		fe_values.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

              PointHistory<dim> *lqph =
                    reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
		
		double cell_averaged_vMises_stress=0;
		double cell_volume=0;
		for(unsigned int k=0; k<n_q_points;++k)
		{
			
			 const Tensor<2,dim> tau = lqph[k].get_tau();
                         const double det_F = lqph[k].get_J(); 
			Tensor<2,dim> Stress = (1/det_F)* tau;

			LAPACKFullMatrix<double> stress_matrix(3,3);
			for(unsigned int s=0;s<dim;++s)
				for(unsigned int t=0;t<dim;++t)
				{
					stress_matrix(s,t) = Stress[s][t];
				}
			if(dim==2)
			{
				double nu = parameters.Poisson;
				double sxx = stress_matrix(0,0);
				double syy = stress_matrix(1,1);
				stress_matrix(2,2) = nu*(sxx+syy);
			}
			else if(dim!=3)
			{
				AssertThrow(false,ExcMessage( "Stress projection - choose dim==2 or dim==3"));
			}
			stress_matrix.compute_eigenvalues(); // afterwards the matrix itselft is unusable 
			double stress_value=0;
			double s1=stress_matrix.eigenvalue(0).real();
			double s2=stress_matrix.eigenvalue(1).real();
			double s3=stress_matrix.eigenvalue(2).real();
			stress_value=std::sqrt(  0.5*( (s1-s2)*(s1-s2) + (s2-s3)*(s2-s3) + (s3-s1)*(s3-s1) )  );
			const double JxW = fe_values.JxW(k);
			
			
			cell_averaged_vMises_stress+=(stress_value)*JxW;
			cell_volume+=JxW;
			
			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
			    double shape_fun_i = fe_values.shape_value(i,k);

                             if (i_group == u_dof)				
				  cell_rhs(i)+= (shape_fun_i * stress_value * JxW);

				for(unsigned int j=0; j<dofs_per_cell; ++j) {
                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))  					
				{
					double shape_fun_j = fe_values.shape_value(j,k);					
					cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);		
				}
                            }
			}
		}
		stress_averaged(cell_counter) = cell_averaged_vMises_stress/cell_volume;
		     for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                    stress_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
				    for (unsigned int j = 0; j < dofs_per_cell; ++j)
				       stress_projected_matrix.add(local_dof_indices[i],
				 	 		 local_dof_indices[j],
							 cell_matrix(i, j));
                                       }
	}

        const int solver_its = stress_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
                                
                                
	SolverControl solver_control(solver_its, 1e-9);
	GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);
	PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(stress_projected_matrix.block(u_dof, u_dof)); 
	solver_CG.solve(stress_projected_matrix.block(u_dof, u_dof), 
                        stress_projected.block(u_dof),
                        stress_projected_rhs.block(u_dof),
			preconditioner);
        

       stress_projected.block(c_rg_dof) = 0;
       stress_projected.block(c_ip_dof) = 0;
       stress_projected.block(c_or_dof) = 0;
       stress_projected.block(c_nu_dof) = 0;


	constraints_1.distribute(stress_projected);
	
}

template <int dim>
void Solid<dim>::sptial_density_grads_projection(BlockVector<double> &sptial_density_grads)
{

    std::cout << " sptial_density_grads " << std::flush;

    AffineConstraints<double>  constraints_1;
    constraints_1.clear();
    constraints_1.close();
    
    BlockVector<double>                sptial_density_grads_projected_rhs(dofs_per_block);
    BlockSparseMatrix<double>               sptial_density_grads_projected_matrix;


    sptial_density_grads_projected_matrix.reinit (sparsity_pattern);

    

    FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
    
    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    AssertThrow (sptial_density_grads.size() == dof_handler_ref.n_dofs(),
                ExcMessage("Compute sptial_density_grads projection - sizes of vectors do not match")
                );
    sptial_density_grads =0;
    sptial_density_grads_projected_rhs=0;
    sptial_density_grads_projected_matrix=0;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
    
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
        cell_matrix=0.0;
        cell_rhs=0.0;
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

                      PointHistory<dim> *lqph =
                            reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        
        
        for(unsigned int k=0; k<n_q_points;++k)
        {
            const Tensor<1, dim>         grad_c_s = lqph[k].get_grad_c_spatial(NU);
            
               
            const double JxW = fe_values.JxW(k);
            
            
            for(unsigned int i=0; i<dofs_per_cell; ++i)
            {
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
                double shape_fun_i = fe_values.shape_value(i,k);

                             if (i_group == u_dof){
                                 if(component_i == 0)
                                     cell_rhs(i)+= (shape_fun_i * grad_c_s[0]* JxW);
   
                                 else if(component_i == 1)
                                     cell_rhs(i)+= (shape_fun_i * grad_c_s[1] * JxW);
                                 }


                for(unsigned int j=0; j<dofs_per_cell; ++j) {

                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))
                {
                    double shape_fun_j = fe_values.shape_value(j,k);
                    cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);
                }
                            }
            }
        }
        
             for (unsigned int i = 0; i < dofs_per_cell; ++i){
                 sptial_density_grads_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        sptial_density_grads_projected_matrix.add(local_dof_indices[i],
                               local_dof_indices[j],
                             cell_matrix(i, j));
                                       }
    }
    const int solver_its = sptial_density_grads_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
    SolverControl solver_control(solver_its, 1e-9);
    GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);

    PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(sptial_density_grads_projected_matrix.block(u_dof, u_dof));
    solver_CG.solve(sptial_density_grads_projected_matrix.block(u_dof, u_dof),
                    sptial_density_grads.block(u_dof),
                    sptial_density_grads_projected_rhs.block(u_dof),
            preconditioner);
        
    sptial_density_grads.block(c_rg_dof)=0;
    sptial_density_grads.block(c_ip_dof)=0;
    sptial_density_grads.block(c_or_dof)=0;
    sptial_density_grads.block(c_nu_dof)=0;

    constraints_1.distribute(sptial_density_grads);
    
}


template <int dim>
void Solid<dim>::growth_factor_projection(BlockVector<double> &growth_factor)
{
  std::cout << " Project growth values " << std::flush;


    AffineConstraints<double>  constraints_1;
	constraints_1.clear();
	constraints_1.close();
	
	BlockVector<double>				growth_projected_rhs(dofs_per_block);
	BlockSparseMatrix<double>	         	growth_projected_matrix;


	growth_projected_matrix.reinit (sparsity_pattern);

	

	FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
	
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	AssertThrow (growth_factor.size() == dof_handler_ref.n_dofs(),
				ExcMessage("Compute growth projection - sizes of vectors do not match")
				);
	growth_factor =0;
	growth_projected_rhs=0;
	growth_projected_matrix=0;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		
	
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
		cell_matrix=0.0;
		cell_rhs=0.0;
		fe_values.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

                      PointHistory<dim> *lqph =
                            reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
		
		
		for(unsigned int k=0; k<n_q_points;++k)
		{
			
			 const double v_t = lqph[k].get_growth_factor_t();
                         const double v_r = lqph[k].get_growth_factor_r();

			const double JxW = fe_values.JxW(k);
			
			
			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
			    double shape_fun_i = fe_values.shape_value(i,k);

                             if (i_group == u_dof){
                                 if(component_i == 0)				
				     cell_rhs(i)+= (shape_fun_i * v_t * JxW);
   
                                 else if(component_i == 1)
                                     cell_rhs(i)+= (shape_fun_i * v_r * JxW);
                                 }


				for(unsigned int j=0; j<dofs_per_cell; ++j) {

                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))  					
				{
					double shape_fun_j = fe_values.shape_value(j,k);					
					cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);		
				}
                            }
			}
		}
		
		     for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                    growth_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
				    for (unsigned int j = 0; j < dofs_per_cell; ++j)
				       growth_projected_matrix.add(local_dof_indices[i],
				 	 		 local_dof_indices[j],
							 cell_matrix(i, j));
                                       }
	}
	const int solver_its = growth_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
	SolverControl solver_control(solver_its, 1e-9);
	GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);

	PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(growth_projected_matrix.block(u_dof, u_dof)); 
	solver_CG.solve(growth_projected_matrix.block(u_dof, u_dof), 
                        growth_factor.block(u_dof),
                        growth_projected_rhs.block(u_dof),
			preconditioner);
        
    growth_factor.block(c_rg_dof)=0;
    growth_factor.block(c_ip_dof)=0;
    growth_factor.block(c_or_dof)=0;
    growth_factor.block(c_nu_dof)=0;

	constraints_1.distribute(growth_factor);
	
}

template <int dim>
void Solid<dim>::compute_Stiffness_projection(BlockVector<double> &Stiffness)
{
  std::cout << " Project Stiffness " << std::flush;


    AffineConstraints<double>  constraints_1;
	constraints_1.clear();
	constraints_1.close();
	
	BlockSparseMatrix<double>   		Stiffness_projected_matrix;
	BlockVector<double>	        	Stiffness_projected_rhs(dofs_per_block);


	Stiffness_projected_matrix.reinit (sparsity_pattern);


	FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
	
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	AssertThrow (Stiffness.size() == dof_handler_ref.n_dofs(),
				ExcMessage("Compute Stiffness - sizes of vectors do not match")
				);
	Stiffness =0;
	Stiffness_projected_rhs=0;
	Stiffness_projected_matrix=0;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		
	
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
		cell_matrix=0.0;
		cell_rhs=0.0;
		fe_values.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

                  PointHistory<dim> *lqph =
                      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
		
		
		for(unsigned int k=0; k<n_q_points;++k)
		{
			
			 const double st = lqph[k].get_elastic_modulus();
                
			const double JxW = fe_values.JxW(k);
			
			
		for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
			    double shape_fun_i = fe_values.shape_value(i,k);

                             if(i_group == u_dof)				
				  cell_rhs(i)+= shape_fun_i * st * JxW;

				for(unsigned int j=0; j<dofs_per_cell; ++j) {
                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))  					
				{
					double shape_fun_j = fe_values.shape_value(j,k);					
					cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);		
				}
                            }
			}
		}
		
		         for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                    Stiffness_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
				    for (unsigned int j = 0; j < dofs_per_cell; ++j)
				       Stiffness_projected_matrix.add(local_dof_indices[i],
				 	 		 local_dof_indices[j],
							 cell_matrix(i, j));
                                       }
	}

        const int solver_its = Stiffness_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
	SolverControl solver_control(solver_its, 1e-9);
	GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);
	PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(Stiffness_projected_matrix.block(u_dof, u_dof)); 
	solver_CG.solve(Stiffness_projected_matrix.block(u_dof, u_dof), 
                        Stiffness.block(u_dof),
                        Stiffness_projected_rhs.block(u_dof),
			preconditioner);
        

    Stiffness.block(c_rg_dof)=0;
    Stiffness.block(c_ip_dof)=0;
    Stiffness.block(c_or_dof)=0;
    Stiffness.block(c_nu_dof)=0;

	constraints_1.distribute(Stiffness);
	
}

template <int dim>
void Solid<dim>::compute_growth_norm_projection(BlockVector<double> &growth_norm)
{
  std::cout << " Project growth norm " << std::flush;


    AffineConstraints<double>  constraints_1;
	constraints_1.clear();
	constraints_1.close();
	
	BlockSparseMatrix<double>   		growth_projected_matrix;
	BlockVector<double>	        	growth_projected_rhs(dofs_per_block);


	growth_projected_matrix.reinit (sparsity_pattern);

	

	FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
	
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	AssertThrow (growth_norm.size() == dof_handler_ref.n_dofs(),
				ExcMessage("Compute growth norm - sizes of vectors do not match")
				);
	growth_norm =0;
	growth_projected_rhs=0;
	growth_projected_matrix=0;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		
	
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
		cell_matrix=0.0;
		cell_rhs=0.0;
		fe_values.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

                PointHistory<dim> *lqph =
                        reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
		
		
		for(unsigned int k=0; k<n_q_points;++k)
		{
			
			 const double v_norm = lqph[k].get_growth_norm();
                
			const double JxW = fe_values.JxW(k);
			
			
		for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
			    double shape_fun_i = fe_values.shape_value(i,k);

                             if(i_group == u_dof)				
				  cell_rhs(i)+= shape_fun_i * v_norm * JxW;

				for(unsigned int j=0; j<dofs_per_cell; ++j) {
                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))  					
				{
					double shape_fun_j = fe_values.shape_value(j,k);					
					cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);		
				}
                            }
			}
		}
		
		         for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                    growth_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
				    for (unsigned int j = 0; j < dofs_per_cell; ++j)
				       growth_projected_matrix.add(local_dof_indices[i],
				 	 		 local_dof_indices[j],
							 cell_matrix(i, j));
                                       }
	}

        const int solver_its = growth_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
	SolverControl solver_control(solver_its, 1e-9);
	GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);
	PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(growth_projected_matrix.block(u_dof, u_dof)); 
	solver_CG.solve(growth_projected_matrix.block(u_dof, u_dof), 
                        growth_norm.block(u_dof),
                        growth_projected_rhs.block(u_dof),
			preconditioner);
        

    growth_norm.block(c_rg_dof)=0;
    growth_norm.block(c_ip_dof)=0;
    growth_norm.block(c_or_dof)=0;
    growth_norm.block(c_nu_dof)=0;

	constraints_1.distribute(growth_norm);
	
  }

//template <int dim>
//void Solid<dim>::source_terms_projection(BlockVector<double> &source_terms)
//{
//  std::cout << " Project source terms values " << std::flush;
//
//
//    AffineConstraints<double>  constraints_1;
//    constraints_1.clear();
//    constraints_1.close();
//
//    BlockVector<double>                source_projected_rhs(dofs_per_block);
//    BlockSparseMatrix<double>                 source_projected_matrix;
//
//
//    source_projected_matrix.reinit (sparsity_pattern);
//
//
//
//    FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
//
//    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
//    Vector<double> cell_rhs (dofs_per_cell);
//    AssertThrow (source_terms.size() == dof_handler_ref.n_dofs(),
//                ExcMessage("Compute source terms projection - sizes of vectors do not match")
//                );
//    source_terms =0;
//    source_projected_rhs=0;
//    source_projected_matrix=0;
//    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
//
//
// for (const auto &cell : dof_handler_ref.active_cell_iterators()){
//        cell_matrix=0.0;
//        cell_rhs=0.0;
//        fe_values.reinit(cell);
//
//        cell->get_dof_indices(local_dof_indices);
//
//                      PointHistory<dim> *lqph =
//                            reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
//
//
//        for(unsigned int k=0; k<n_q_points;++k)
//        {
//
//             const double F_c = lqph[k].get_F_c();
//             const double F_c_2 = lqph[k].get_F_c_2();
//
//            const double JxW = fe_values.JxW(k);
//
//
//            for(unsigned int i=0; i<dofs_per_cell; ++i)
//            {
//                            const unsigned int component_i = fe.system_to_component_index(i).first;
//                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
//                double shape_fun_i = fe_values.shape_value(i,k);
//
//                             if (i_group == u_dof){
//                                 if(component_i == 0)
//                     cell_rhs(i)+= (shape_fun_i * F_c * JxW);
//
//                                 else if(component_i == 1)
//                                     cell_rhs(i)+= (shape_fun_i * F_c_2 * JxW);
//                                 }
//
//
//                for(unsigned int j=0; j<dofs_per_cell; ++j) {
//
//                                  const unsigned int component_j = fe.system_to_component_index(j).first;
//                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;
//
//                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))
//                {
//                    double shape_fun_j = fe_values.shape_value(j,k);
//                    cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);
//                }
//                            }
//            }
//        }
//
//             for (unsigned int i = 0; i < dofs_per_cell; ++i){
//                                    source_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
//                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
//                       source_projected_matrix.add(local_dof_indices[i],
//                               local_dof_indices[j],
//                             cell_matrix(i, j));
//                                       }
//    }
//    const int solver_its = source_projected_matrix.block(u_dof, u_dof).m()
//                                 * parameters.multiplier_max_iterations_linear_solver;
//    SolverControl solver_control(solver_its, 1e-9);
//    GrowingVectorMemory<Vector<double> > GVM;
//        SolverCG<Vector<double> > solver_CG(solver_control, GVM);
//
//    PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
//
//          preconditioner.use_matrix(source_projected_matrix.block(u_dof, u_dof));
//    solver_CG.solve(source_projected_matrix.block(u_dof, u_dof),
//                        source_terms.block(u_dof),
//                        source_projected_rhs.block(u_dof),
//            preconditioner);
//
//    source_terms.block(c_rg_dof)=0;
//    source_terms.block(c_ip_dof)=0;
//    source_terms.block(c_or_dof)=0;
//    source_terms.block(c_nu_dof)=0;
//    constraints_1.distribute(source_terms);
//
//    }

template <int dim>
void Solid<dim>::velocity_projection(BlockVector<double> &velocity_values)
{
  std::cout << " Project velocity values " << std::flush;


    AffineConstraints<double>  constraints_1;
    constraints_1.clear();
    constraints_1.close();
    
    BlockVector<double>                velocity_projected_rhs(dofs_per_block);
    BlockSparseMatrix<double>                 velocity_projected_matrix;


    velocity_projected_matrix.reinit (sparsity_pattern);

    

    FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
    
    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    AssertThrow (velocity_values.size() == dof_handler_ref.n_dofs(),
                ExcMessage("Compute velocity projection - sizes of vectors do not match")
                );
    velocity_values =0;
    velocity_projected_rhs=0;
    velocity_projected_matrix=0;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
    
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
        cell_matrix=0.0;
        cell_rhs=0.0;
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

                      PointHistory<dim> *lqph =
                            reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        
        
        for(unsigned int k=0; k<n_q_points;++k)
        {
            
            Tensor<1,dim> v = lqph[k].get_velocity();
            
             const double v1 = v[0];
             const double v2 = v[1];

            const double JxW = fe_values.JxW(k);
            
            
            for(unsigned int i=0; i<dofs_per_cell; ++i)
            {
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
                double shape_fun_i = fe_values.shape_value(i,k);

                             if (i_group == u_dof){
                                 if(component_i == 0)
                     cell_rhs(i)+= (shape_fun_i * v1 * JxW);
   
                                 else if(component_i == 1)
                                     cell_rhs(i)+= (shape_fun_i * v2 * JxW);
                                 }


                for(unsigned int j=0; j<dofs_per_cell; ++j) {

                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))
                {
                    double shape_fun_j = fe_values.shape_value(j,k);
                    cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);
                }
                            }
            }
        }
        
             for (unsigned int i = 0; i < dofs_per_cell; ++i){
                       velocity_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        velocity_projected_matrix.add(local_dof_indices[i],
                               local_dof_indices[j],
                             cell_matrix(i, j));
                                       }
    }
    const int solver_its = velocity_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
    SolverControl solver_control(solver_its, 1e-9);
    GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);

    PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(velocity_projected_matrix.block(u_dof, u_dof));
    solver_CG.solve(velocity_projected_matrix.block(u_dof, u_dof),
                        velocity_values.block(u_dof),
                        velocity_projected_rhs.block(u_dof),
            preconditioner);
        
    velocity_values.block(c_rg_dof)=0;
    velocity_values.block(c_ip_dof)=0;
    velocity_values.block(c_or_dof)=0;
    velocity_values.block(c_nu_dof)=0;

    constraints_1.distribute(velocity_values);
    
    }
template <int dim>
void Solid<dim>::diffusion_projection(BlockVector<double> &diffusion_values)
{
  std::cout << " diffusion projection " << std::flush;


    AffineConstraints<double>  constraints_1;
    constraints_1.clear();
    constraints_1.close();
    
    BlockSparseMatrix<double>           diffusion_projected_matrix;
    BlockVector<double>                diffusion_projected_rhs(dofs_per_block);


    diffusion_projected_matrix.reinit (sparsity_pattern);


    FEValues<dim> fe_values (fe, qf_cell, update_values  | update_JxW_values);
    
    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    AssertThrow (diffusion_values.size() == dof_handler_ref.n_dofs(),
                ExcMessage("Compute diffusion - sizes of vectors do not match")
                );
    diffusion_values =0;
    diffusion_projected_rhs=0;
    diffusion_projected_matrix=0;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
    
 for (const auto &cell : dof_handler_ref.active_cell_iterators()){
        cell_matrix=0.0;
        cell_rhs=0.0;
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

                  PointHistory<dim> *lqph =
                      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
        
        
        for(unsigned int k=0; k<n_q_points;++k)
        {
            
             const double d = lqph[k].get_dcc_r(NU);
                
            const double JxW = fe_values.JxW(k);
            
            
        for(unsigned int i=0; i<dofs_per_cell; ++i)
            {
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const unsigned int i_group     = fe.system_to_base_index(i).first.first;
                double shape_fun_i = fe_values.shape_value(i,k);

                             if(i_group == u_dof)
                  cell_rhs(i)+= shape_fun_i * d * JxW;

                for(unsigned int j=0; j<dofs_per_cell; ++j) {
                                  const unsigned int component_j = fe.system_to_component_index(j).first;
                                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                                     if (component_i== component_j && ((i_group == j_group) && (i_group == u_dof)))
                {
                    double shape_fun_j = fe_values.shape_value(j,k);
                    cell_matrix(i,j)+=  (shape_fun_i * shape_fun_j * JxW);
                }
                            }
            }
        }
        
                 for (unsigned int i = 0; i < dofs_per_cell; ++i){
                     diffusion_projected_rhs(local_dof_indices[i]) += cell_rhs(i);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        diffusion_projected_matrix.add(local_dof_indices[i],
                               local_dof_indices[j],
                             cell_matrix(i, j));
                                       }
    }

        const int solver_its = diffusion_projected_matrix.block(u_dof, u_dof).m()
                                 * parameters.multiplier_max_iterations_linear_solver;
    SolverControl solver_control(solver_its, 1e-9);
    GrowingVectorMemory<Vector<double> > GVM;
        SolverCG<Vector<double> > solver_CG(solver_control, GVM);
    PreconditionSelector<SparseMatrix<double>, Vector<double> >  preconditioner ("ssor",1.2);
          
          preconditioner.use_matrix(diffusion_projected_matrix.block(u_dof, u_dof));
    solver_CG.solve(diffusion_projected_matrix.block(u_dof, u_dof),
                    diffusion_values.block(u_dof),
                    diffusion_projected_rhs.block(u_dof),
            preconditioner);
        

    diffusion_values.block(c_rg_dof)=0;
    diffusion_values.block(c_ip_dof)=0;
    diffusion_values.block(c_or_dof)=0;
    diffusion_values.block(c_nu_dof)=0;

    constraints_1.distribute(diffusion_values);
    
}
}

int main (int argc, char* argv[])
{
  using namespace dealii;
  using namespace Brain_growth;

    Assert(argc==3, ExcMessage("This project needs two arguments as input, the first is the name of the parameters file, and the second is the geometry dimensionality of problem 2 or 3."));

  try
    {
       int dim = strtol(argv[2],NULL, 10);
        if (dim == 2){
            Solid<2> solid_2d(argv[1]);
            solid_2d.run();

        }
        
        else if (dim==3){
            Solid<3> solid_3d(argv[1]);
            solid_3d.run();
        }
      else
        Assert(dim<3, ExcInternalError());
        
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
