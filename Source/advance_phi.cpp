
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace amrex;

void fill_phi_rhs (MultiFab& phi_rhs_mf, MultiFab& phi_old_mf, const Real* dx)
{
    for ( MFIter mfi(phi_rhs_mf); mfi.isValid(); ++mfi )
    {
      const Box& bx = mfi.validbox();
      const auto ncomp = phi_old_mf.nComp();

      const auto& phi_rhs_fab = phi_rhs_mf.array(mfi);
      const auto& phi_old_fab = phi_old_mf.array(mfi);

      // For each grid, loop over all the valid points
      AMREX_FOR_4D(bx, ncomp, i, j, k, n,
      {
         // Set phi_rhs_fab = 1.0
         phi_rhs_fab(i, j, k, n) = 1.0;
      });
    }
}

void advance_phi (MultiFab& phi_new_mf, MultiFab& phi_old_mf, Real time, Real dt, const Real* dx)
{
    int ncomp = phi_new_mf.nComp();

    // Fill ghost cells for each grid from valid regions of another grid
    phi_old_mf.FillBoundary();

    // Create a MultiFab containing the time integration RHS
    MultiFab rhs_mf(phi_new_mf.boxArray(), phi_new_mf.DistributionMap(), ncomp, 0);
    fill_phi_rhs(rhs_mf, phi_old_mf, dx);

    // Loop over grids to do a forward euler integration in time
    for ( MFIter mfi(phi_new_mf); mfi.isValid(); ++mfi )
    {
      const Box& bx = mfi.validbox();

      const auto& phi_new_fab = phi_new_mf.array(mfi);
      const auto& phi_old_fab = phi_old_mf.array(mfi);
      const auto&     rhs_fab =     rhs_mf.array(mfi);

      // For each grid, loop over all the valid points
      AMREX_FOR_4D(bx, ncomp, i, j, k, n,
      {
         // Right now rhs_fab(i,j,k,n) = 1 so this adds dt to every value in every time step
         phi_new_fab(i,j,k,n) = phi_old_fab(i,j,k,n) + dt * rhs_fab(i,j,k,n);
      });
    }

    // Fill ghost cells for each grid from valid regions of another grid
    phi_new_mf.FillBoundary();
}
