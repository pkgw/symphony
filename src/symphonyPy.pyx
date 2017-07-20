# -*- mode: python; coding: utf-8 -*-

from symphonyHeaders cimport j_nu, alpha_nu, j_nu_fit, alpha_nu_fit, rho_nu_fit, compute_pkgw_pitchy, sample_synchrotron

def j_nu_py(double nu,
            double magnetic_field,
            double electron_density,
            double observer_angle,
            int distribution,
            int polarization,
            double theta_e,
            double power_law_p,
            double gamma_min,
            double gamma_max,
            double gamma_cutoff,
            double kappa,
            double kappa_width):

  """Returns j_nu(nu, magnetic_field, electron_density, observer_angle, 
                  distribution, polarization, theta_e, power_law_p, 
                  gamma_min, gamma_max, gamma_cutoff, kappa, kappa_width).
     Keys for distribution functions: symphonyPy.MAXWELL_JUETTNER, 
                                      symphonyPy.POWER_LAW, 
                                      symphonyPy.KAPPA_DIST
     Keys for Stokes parameter: symphonyPy.STOKES_I,
                                symphonyPy.STOKES_Q,
                                symphonyPy.STOKES_U,
                                symphonyPy.STOKES_V"""

  cdef char* error_message = NULL
  result = j_nu(nu, magnetic_field, electron_density,
                observer_angle, distribution, polarization,
                theta_e, power_law_p, gamma_min, gamma_max,
                gamma_cutoff, kappa, kappa_width, &error_message)
  if error_message:
    raise RuntimeError (error_message)
  return result

def alpha_nu_py(double nu,
                double magnetic_field,
                double electron_density,
                double observer_angle,
                int distribution,
                int polarization,
                double theta_e,
                double power_law_p,
                double gamma_min,
                double gamma_max,
                double gamma_cutoff,
                double kappa,
                double kappa_width):

  """Returns alpha_nu(nu, magnetic_field, electron_density, observer_angle,
                      distribution, polarization, theta_e, power_law_p, 
                      gamma_min, gamma_max, gamma_cutoff, kappa, kappa_width).
     Keys for distribution functions: symphonyPy.MAXWELL_JUETTNER, 
                                      symphonyPy.POWER_LAW, 
                                      symphonyPy.KAPPA_DIST
     Keys for Stokes parameter: symphonyPy.STOKES_I,
                                symphonyPy.STOKES_Q,
                                symphonyPy.STOKES_U,
                                symphonyPy.STOKES_V"""

  cdef char* error_message = NULL
  result = alpha_nu(nu, magnetic_field, electron_density,
                    observer_angle, distribution, polarization,
                    theta_e, power_law_p, gamma_min, gamma_max,
                    gamma_cutoff, kappa, kappa_width, &error_message)
  if error_message:
    raise RuntimeError (error_message)
  return result

def j_nu_fit_py(double nu,
                double magnetic_field,
                double electron_density,
                double observer_angle,
                int distribution,
                int polarization,
                double theta_e,
                double power_law_p,
                double gamma_min,
                double gamma_max,
                double gamma_cutoff,
                double kappa,
                double kappa_width):

  """Returns j_nu_fit(nu, magnetic_field, electron_density, observer_angle, 
                    distribution, polarization, theta_e, power_law_p, 
                    gamma_min, gamma_max, gamma_cutoff, kappa, kappa_width).
     Keys for distribution functions: symphonyPy.MAXWELL_JUETTNER, 
                                      symphonyPy.POWER_LAW, 
                                      symphonyPy.KAPPA_DIST
     Keys for Stokes parameter: symphonyPy.STOKES_I,
                                symphonyPy.STOKES_Q,
                                symphonyPy.STOKES_U,
                                symphonyPy.STOKES_V"""

  return j_nu_fit(nu, magnetic_field, electron_density,
                  observer_angle, distribution, polarization,
                  theta_e, power_law_p, gamma_min, gamma_max,
                  gamma_cutoff, kappa, kappa_width)

def alpha_nu_fit_py(double nu,
                    double magnetic_field,
                    double electron_density,
                    double observer_angle,
                    int distribution,
                    int polarization,
                    double theta_e,
                    double power_law_p,
                    double gamma_min,
                    double gamma_max,
                    double gamma_cutoff,
                    double kappa,
                    double kappa_width):

  """Returns alpha_nu_fit(nu, magnetic_field, electron_density, observer_angle, 
                        distribution, polarization, theta_e, power_law_p, 
                        gamma_min, gamma_max, gamma_cutoff, kappa, kappa_width).
     Keys for distribution functions: symphonyPy.MAXWELL_JUETTNER, 
                                      symphonyPy.POWER_LAW, 
                                      symphonyPy.KAPPA_DIST
     Keys for Stokes parameter: symphonyPy.STOKES_I,
                                symphonyPy.STOKES_Q,
                                symphonyPy.STOKES_U,
                                symphonyPy.STOKES_V"""

  return alpha_nu_fit(nu, magnetic_field, electron_density,
                      observer_angle, distribution, polarization,
                      theta_e, power_law_p, gamma_min, gamma_max,
                      gamma_cutoff, kappa, kappa_width)


def rho_nu_fit_py(double nu,
                  double magnetic_field=30.,
                  double electron_density=1.,
                  double observer_angle=1.0472,
                  int distribution=0,
                  int polarization=18,
                  double theta_e=10.,
                  double power_law_p=3.,
                  double gamma_min=1.,
                  double gamma_max=1000.,
                  double gamma_cutoff=1e10,
                  double kappa=3.5,
                  double kappa_width=10.):

  """Returns rho_nu_fit(nu, magnetic_field, electron_density, observer_angle, 
                        distribution, polarization, theta_e, power_law_p, 
                        gamma_min, gamma_max, gamma_cutoff, kappa, kappa_width).
     Keys for distribution functions: symphonyPy.MAXWELL_JUETTNER, 
                                      symphonyPy.POWER_LAW, 
                                      symphonyPy.KAPPA_DIST
     Keys for Stokes parameter: symphonyPy.STOKES_I,
                                symphonyPy.STOKES_Q,
                                symphonyPy.STOKES_U,
                                symphonyPy.STOKES_V"""

  return rho_nu_fit(nu, magnetic_field, electron_density,
                      observer_angle, distribution, polarization,
                      theta_e, power_law_p, gamma_min, gamma_max,
                      gamma_cutoff, kappa, kappa_width)


def compute_pkgw_pitchy_py(int mode,
                           int polarization,
                           double nu,
                           double magnetic_field,
                           double electron_density,
                           double observer_angle,
                           double power_law_p,
                           double gamma_min,
                           double gamma_max,
                           double gamma_cutoff):
  cdef char* error_message = NULL

  result = compute_pkgw_pitchy(mode, polarization, nu, magnetic_field,
                               electron_density, observer_angle, power_law_p, gamma_min,
                               gamma_max, gamma_cutoff, &error_message)

  if error_message:
    raise RuntimeError(error_message)
  return result


def sample_synchrotron_py(int mode,
                          int polarization,
                          int is_pitchy,
                          double nu,
                          double magnetic_field,
                          double electron_density,
                          double observer_angle,
                          double power_law_p,
                          double gamma,
                          double n):
  cdef char* error_message = NULL

  result = sample_synchrotron(mode, polarization, is_pitchy, nu, magnetic_field,
                              electron_density, observer_angle, power_law_p,
                              gamma, n, &error_message)

  if error_message:
    raise RuntimeError(error_message)
  return result


#DEFINE KEYS FOR DISTRIBUTION FUNCTIONS
MAXWELL_JUETTNER = 0
POWER_LAW        = 1
KAPPA_DIST       = 2

#DEFINE KEYS FOR STOKES PARAMETERS
STOKES_I         = 15
STOKES_Q         = 16
STOKES_U         = 17
STOKES_V         = 18

ABSORPTIVITY = 10
EMISSIVITY = 11
