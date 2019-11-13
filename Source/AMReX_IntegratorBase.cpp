#include "AMReX_IntegratorBase.H"

using namespace amrex;

IntegratorBase::IntegratorBase(amrex::MultiFab& S_old_external,
                               amrex::MultiFab& S_new_external,
                               amrex::Real initial_time) : time(initial_time),
                                                           S_old(S_old_external),
                                                           S_new(S_new_external) {}

void IntegratorBase::rhs(amrex::MultiFab& S_rhs, const amrex::MultiFab& S_data, const amrex::Real time)
{
    Fun(S_rhs, S_data, time);
}

amrex::MultiFab& IntegratorBase::get_new_data()
{
    return S_new;
}

amrex::MultiFab& IntegratorBase::get_old_data()
{
    return S_old;
}

amrex::Real IntegratorBase::get_time()
{
    return time;
}

amrex::Real IntegratorBase::compute_adaptive_timestep(const amrex::MultiFab& solution, amrex::MultiFab& solution_error)
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
    next_timestep = std::min(std::max(next_timestep * adaptive_factor_lo,
                                      timestep / adaptive_factor_hi),
                             timestep * adaptive_factor_hi);

    return next_timestep;
}
