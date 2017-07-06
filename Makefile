CC = icc
C_INCLUDES = -Isrc -I../stack/include \
  -I$(shell python -c 'import numpy as np; print(np.get_include())') \
  -I$(shell python -c 'from distutils import sysconfig as sc; print(sc.get_python_inc())')
CFLAGS = -Dsymphony_EXPORTS -std=c99 -O3 -g -fPIC $(C_INCLUDES)
LINKFLAGS = -fPIC -shared
LDFLAGS = -L../stack/lib -Wl,-rpath,$(shell cd .. && pwd)/stack/lib -lgsl -lgslcblas

sources = \
  src/bessel_mod.c \
  src/demo.c \
  src/distribution_function_common_routines.c \
  src/fits.c \
  src/integrator/integrands.c \
  src/integrator/integrate.c \
  src/kappa/kappa.c \
  src/kappa/kappa_fits.c \
  src/maxwell_juettner/maxwell_juettner.c \
  src/maxwell_juettner/maxwell_juettner_fits.c \
  src/params.c \
  src/pkgw_pitchy_power_law.c \
  src/power_law/power_law.c \
  src/power_law/power_law_fits.c \
  src/symphony.c

symphonyPy.so: symphonyPy.o $(patsubst %.c,%.o,$(sources))
	$(CC) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

symphonyPy.c: src/symphonyPy.pyx
	cython $(C_INCLUDES) -2 --output-file $@ $<

clean:
	-find -name '*.o' -delete
	-rm -f symphonyPy.c symphonyPy.so
