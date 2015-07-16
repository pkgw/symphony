//GSL libraries
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_errno.h>

//other C header files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

//choose distribution function
#define THERMAL (0)
#define POWER_LAW (1)
#define KAPPA_DIST (2)
#define DISTRIBUTION_FUNCTION (POWER_LAW)

//choose absorptivity or emissivity
#define ABSORP (10)
#define EMISS  (11)
#define MODE   (EMISS)

//choose polarization mode
#define STOKES_I (15)
#define STOKES_Q (16)
#define STOKES_U (17)
#define STOKES_V (18)
#define EXTRAORDINARY_MODE (19)
#define ORDINARY_MODE (20)
#define POL_MODE (STOKES_I)

//function declarations
double n_peak(double nu);
double polarization_term(double gamma, double n, double nu);
double my_Bessel_J(double n, double x);
double my_Bessel_dJ(double n, double x);
double maxwell_juttner_f(double gamma);
double integrand_without_extra_factor(double gamma, double n, double nu);
double gamma_integrand(double gamma, void * params);
double gamma_integration_result(double n, void * params);
double n_summation(double nu);
double n_integration(double n_minus, double nu);
double gsl_integrate(double min, double max, double n, double nu);
double normalize_f();
double power_law_to_be_normalized(double gamma, void * params);
double power_law_f(double gamma);
double kappa_to_be_normalized(double gamma, void * params);
double kappa_f(double gamma);
double derivative(double n_start, double nu);
double differential_of_f(double gamma, double nu);