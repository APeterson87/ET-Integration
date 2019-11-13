#include "AMReX_AdaptiveIntegratorBase.H"

using namespace amrex;

AdaptiveIntegratorBase::AdaptiveIntegratorBase(amrex::MultiFab& S_old_external,
                                               amrex::MultiFab& S_new_external,
                                               amrex::Real initial_time) : IntegratorBase(S_old_external, S_new_external, initial_time),
                                                                           order(0),
                                                                           use_adaptive_timestep(false),
                                                                           error_abs_tol({1.0}),
                                                                           error_rel_tol({1.0}),
                                                                           safety_factor_lo(0.9),
                                                                           safety_factor_hi(1.1)
{
    initialize_adaptivity();
}

void AdaptiveIntegratorBase::initialize_adaptivity()
{
    ParmParse pp("integration.adaptive");
    pp.query("use_adaptive_timestep", use_adaptive_timestep);
    pp.queryarr("error_abs_tol", error_abs_tol);
    pp.queryarr("error_rel_tol", error_rel_tol);
    pp.query("safety_factor_lo", safety_factor_lo);
    pp.query("safety_factor_hi", safety_factor_hi);
}

void AdaptiveIntegratorBase::set_order(int method_order)
{
    order = static_cast<amrex::Real>(method_order);
}

amrex::Real AdaptiveIntegratorBase::compute_adaptive_timestep(const amrex::MultiFab& solution, amrex::MultiFab& solution_error)
{
    // Compute the error and put it in solution_error
    compute_error(solution_error);

    // List the components to include for calculating the error
    Vector<int> comp_to_norm(solution_error.nComp());
    std::iota(comp_to_norm.begin(), comp_to_norm.end(), 0);

    // Calculate L-2 norm of solution error
    auto serr_norm2 = solution_error.norm2(comp_to_norm);

    // Calculate L-2 norm of the solution
    auto snew_norm2 = solution.norm2(comp_to_norm);

    // Convert L-2 norms to RMS norms
    auto sqrt_num_cells = std::sqrt(solution.boxArray().d_numPts());
    for (auto&& x : serr_norm2) x = x/sqrt_num_cells;
    for (auto&& x : snew_norm2) x = x/sqrt_num_cells;

    // Calculate timestep to match the target error for each component
    Vector<Real> target_timesteps;
    for (int i = 0; i < solution.nComp(); ++i)
    {
        Real abs_err, rel_err;

        if (error_abs_tol.size() == 1) abs_err = error_abs_tol[0];
        else abs_err = error_abs_tol[i];

        if (error_rel_tol.size() == 1) rel_err = error_rel_tol[0];
        else rel_err = error_rel_tol[i];

        Real actual_err = serr_norm2[i];
        Real target_err = abs_err + rel_err * snew_norm2[i];

        Real target_dt = timestep * std::pow(target_err/actual_err, 1./order);
        target_timesteps.push_back(target_dt);
    }

    // Take the minimum of the component timesteps
    auto next_timestep_idx = std::distance(target_timesteps.begin(),
                                           std::min_element(target_timesteps.begin(), target_timesteps.end()));
    Real next_timestep = target_timesteps[next_timestep_idx];
    limiting_component = next_timestep_idx;

    // Apply safety factors to the next timestep
    next_timestep = std::min(std::max(next_timestep * safety_factor_lo,
                                      timestep / safety_factor_hi),
                             timestep * safety_factor_hi);

    return next_timestep;
}
