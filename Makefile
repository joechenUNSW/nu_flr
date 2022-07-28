CFLAGS=-O3 
PATHS=-I/apps/gsl/2.6/include -L/apps/gsl/2.6/lib/
LIBS=-lgsl -lgslcblas -lm

nu_flr: nu_flr.c Makefile pcu.h 
	gcc nu_flr.c -o nu_flr $(CFLAGS) $(PATHS) $(LIBS) 

clean:
	$(RM) nu_flr

