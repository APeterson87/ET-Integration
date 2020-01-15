#ifndef ET_INTEGRATION_INIT_K_H 
#define ET_INTEGRATION_INIT_K_H 

#include <AMReX_REAL.H> 
#include <AMReX_Array4.H> 
#include <ET_Integration_Setup.H> 

AMREX_GPU_DEVICE 
inline 
void 
state_init(int i, int j, int k, 
        amrex::Array4<amrex::Real> const& state_fab, 
        amrex::Real time, 
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx) noexcept 
{
        amrex::Real x = (i-128 + 0.5)*dx[0];
        amrex::Real y = (j-128 + 0.5)*dx[1];
        amrex::Real z = (k-128 + 0.5)*dx[2];

        state_fab(i, j, k, Idx::phi) = 0;

        state_fab(i, j, k, Idx::gbar00) = 1;

        state_fab(i, j, k, Idx::gbar01) = 0;

        state_fab(i, j, k, Idx::gbar02) = 0;

        state_fab(i, j, k, Idx::gbar11) = 1;

        state_fab(i, j, k, Idx::gbar12) = 0;

        state_fab(i, j, k, Idx::gbar22) = 1;

        state_fab(i, j, k, Idx::Ksclr) = 0;

        state_fab(i, j, k, Idx::Abar00) = 0;

        state_fab(i, j, k, Idx::Abar01) = 0;

        state_fab(i, j, k, Idx::Abar02) = 0;

        state_fab(i, j, k, Idx::Abar11) = 0;

        state_fab(i, j, k, Idx::Abar12) = 0;

        state_fab(i, j, k, Idx::Abar22) = 0;

        state_fab(i, j, k, Idx::GambarU0) = 0;

        state_fab(i, j, k, Idx::GambarU1) = 0;

        state_fab(i, j, k, Idx::GambarU2) = 0;

        state_fab(i, j, k, Idx::alpha) = 1;

        state_fab(i, j, k, Idx::betaU0) = 0;

        state_fab(i, j, k, Idx::betaU1) = 0;

        state_fab(i, j, k, Idx::betaU2) = 0;

        state_fab(i, j, k, Idx::BU0) = 0;

        state_fab(i, j, k, Idx::BU1) = 0;

        state_fab(i, j, k, Idx::BU2) = 0;

}
#endif