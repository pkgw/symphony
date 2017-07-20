cdef extern from "symphony.h":
    
    double j_nu(double nu,
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
                double kappa_width,
                char **error_message)

    double alpha_nu(double nu,
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
                    double kappa_width,
                    char **error_message)

    double j_nu_fit(double nu,
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
                    double kappa_width)

    double alpha_nu_fit(double nu,
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
                        double kappa_width)

    double rho_nu_fit(double nu,
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
                        double kappa_width)


    double compute_pkgw_pitchy(int mode, int polarization, double nu, double magnetic_field,
                               double electron_density, double observer_angle,
                               double power_law_p, double gamma_min, double gamma_max,
                               double gamma_cutoff, char **error_message)

    double sample_synchrotron(int mode, int polarization, int is_pitchy, double nu, double magnetic_field,
                              double electron_density, double observer_angle,
                              double power_law_p, double gamma, double n,
                              char **error_message)
