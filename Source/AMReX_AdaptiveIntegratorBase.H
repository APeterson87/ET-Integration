#ifndef AMREX_ADAPTIVE_INTEGRATOR_BASE_H
#define AMREX_ADAPTIVE_INTEGRATOR_BASE_H
#include <AMReX_IntegratorBase.H>

class AdaptiveIntegratorBase : public IntegratorBase
{
private:
    bool use_adaptive_timestep;
    amrex::Real order;
    int limiting_component;
    amrex::Vector<amrex::Real> error_abs_tol;
    amrex::Vector<amrex::Real> error_rel_tol;
    amrex::Real safety_factor_lo, safety_factor_hi;
    virtual void compute_error(amrex::MultiFab& solution_error) = 0;
    void initialize_adaptivity();

protected:
    amrex::Real compute_adaptive_timestep(const amrex::MultiFab& solution, amrex::MultiFab& solution_error);
    void set_order(int method_order);
    bool using_adaptivity() { return use_adaptive_timestep; }

public:
    AdaptiveIntegratorBase(amrex::MultiFab& S_old_external, amrex::MultiFab& S_new_external, amrex::Real time);
};

#endif
