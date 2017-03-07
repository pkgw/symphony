CC = icc
C_INCLUDES = -I../stack/include \
  -I../stack/lib/python2.7/site-packages/numpy/core/include -Isrc \
  -I../stack/include/python2.7
CFLAGS = -Dsymphony_EXPORTS -std=c99 -O3 -g -fPIC $(C_INCLUDES)
LINKFLAGS = -fPIC -shared
LDFLAGS = -L../stack/lib -Wl,-rpath,../stack/lib -lgsl -lgslcblas

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
