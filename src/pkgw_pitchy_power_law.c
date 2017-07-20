#include "pkgw_pitchy_power_law.h"
#include "distribution_function_common_routines.h"

/* This is currently identical to the plain power law! */

static double
normalization_function(double gamma, void *params_v)
{
    struct parameters *params = (struct parameters *) params_v;

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

    double beta = sqrt(1 - 1. / (gamma * gamma));

    double prefactor = params->electron_density * (params->power_law_p - 1)
        / (pow(params->gamma_min, 1 - params->power_law_p) - pow(params->gamma_max, 1 - params->power_law_p));

    double body = pow(gamma, -params->power_law_p) * exp(-gamma / params->gamma_cutoff);

    //double pa_term = sin(cosxi * cosxi * M_PI);
    //pa_term = pa_term * pa_term * 2.6459460971005484;
    double pa_term = sin(cosxi * M_PI);
    pa_term = 2 * pa_term * pa_term;

    return norm * prefactor * body * pa_term /
        (pow(params->mass_electron, 3.) * pow(params->speed_light, 3.) * gamma * gamma * beta);
}
