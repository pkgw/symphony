#include "pkgw_pitchy_power_law.h"
#include "distribution_function_common_routines.h"
#include <gsl/gsl_sf_hyperg.h>

/* This is like the stock powerlaw distribution, but we depend on pitch angle
 * as sin(pitch angle)^k. We integrate in cosxi = cos(pitch angle), so the
 * factor becomes (1 - cosxi^2)^(k/2). To maintain normalization as k and
 * cosxi might vary, we divide by the integral of this term, which turns out
 * to be the 2_F_1 Gausss hypergeometric function.
 */

static double
normalization_function(double gamma, void *params_v)
{
    struct parameters *params = (struct parameters *) params_v;

    if (gamma < params->gamma_min || gamma > params->gamma_max)
        return 0;

    double prefactor = (params->power_law_p - 1) /
        (pow(params->gamma_min, 1 - params->power_law_p) - pow(params->gamma_max, 1 - params->power_law_p));
    double body = pow(gamma, -params->power_law_p) * exp(-gamma / params->gamma_cutoff);

    return 4 * params->pi * prefactor * body;
}


double
pkgw_pitchy_power_law_f(double gamma, double cosxi, struct parameters *params)
{
    static double norm = 0;
    static double previous_power_law_p = 0;
    static double previous_gamma_min = 0;
    static double previous_gamma_max = 0;
    static double previous_gamma_cutoff = 0;

    if (norm == 0. ||
        previous_power_law_p != params->power_law_p ||
        previous_gamma_min != params->gamma_min ||
        previous_gamma_max != params->gamma_max ||
        previous_gamma_cutoff != params->gamma_cutoff)
    {
        norm = 1. / normalize_f(normalization_function, params);
        previous_power_law_p = params->power_law_p;
        previous_gamma_min = params->gamma_min;
        previous_gamma_max = params->gamma_max;
        previous_gamma_cutoff = params->gamma_cutoff;
    }

    if (gamma < params->gamma_min || gamma > params->gamma_max)
        return 0;

    double beta = sqrt(1 - 1. / (gamma * gamma));

    double prefactor = params->electron_density * (params->power_law_p - 1)
        / (pow(params->gamma_min, 1 - params->power_law_p) - pow(params->gamma_max, 1 - params->power_law_p));

    double body = pow(gamma, -params->power_law_p) * exp(-gamma / params->gamma_cutoff);

    double pa_term = pow(1 - cosxi * cosxi, 0.5 * params->pppl_k) /
        gsl_sf_hyperg_2F1(0.5, -0.5 * params->pppl_k, 1.5, 1.);

    return norm * prefactor * body * pa_term /
        (pow(params->mass_electron, 3.) * pow(params->speed_light, 3.) * gamma * gamma * beta);
}
