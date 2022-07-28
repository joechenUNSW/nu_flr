#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>

#include "pcu.h"

////////////////////////////////// CONSTANTS ///////////////////////////////////
//All dimensionful quantities are in units of Mpc/h to the appropriate power,
//unless otherwise noted.  Values declared as const double or const int  may be
//modified by the user, while those in #define statements are derived parameters
//which should not be changed.

//conformal hubble today
const double Hc0h = 3.33564095198152e-04; //(1e2/299792.458)
#define Hc0h2 (Hc0h*Hc0h)

//initial scale factor, and max value of eta=ln(a/a_in)
const double aeta_in = 1e-3; 
#define eta_stop (-log(aeta_in))

//density fractions today; assume flat universe with CB, nu, photons, Lambda
const double h = 0.6724; //H_0 / (100 km/sec/Mpc)
const double T_CMB_0_K = 2.7255; //CMB temperature today, in Kelvins

const double Omega_cb_0 = 0.3121; //CDM+Baryon density fraction today
const double Omega_nu_0 = 0.0; //massive nu density fraction today
/*
const double m_nu1 = 0.0584031;
const double m_nu2 = 0.0314153;
const double m_nu3 = 0.0301815;
*/
const double N_nu_eff = 3.044; //effective number of neutrinos in early univ
const double N_nu_massive = 0.0; //3.046; //number of massive neutrinos
const int N_tau = 0; //number of neutrino streams; maximum 900 for this pcu.h
//const int N_mu = 20; //number of multipoles to track for each stream

#define m_nu_eV (93.259*Omega_nu_0*h*h/N_nu_massive)
#define m_nu_sum (93.259*Omega_nu_0*h*h)
#define Omega_nu_t_0 (Omega_nu_0/N_tau)

#define T_CMB_0_K_4 (T_CMB_0_K*T_CMB_0_K*T_CMB_0_K*T_CMB_0_K)
#define T_NUREL_0_K (0.713765855503608*T_CMB_0_K)
#define m_T_nu (m_nu_eV * 11604.51812 / T_NUREL_0_K)
#define Omega_gam_0 ((4.46911743913795e-07)*T_CMB_0_K_4/(h*h))
#define Omega_nurel_0 (0.227107317660239*(N_nu_eff-N_nu_massive)*Omega_gam_0)
#define Omega_rel_0 (Omega_gam_0+Omega_nurel_0)
#define Omega_de_0 (1.0-Omega_cb_0-Omega_nu_0-Omega_rel_0)
//#define Omega_gam_0 (0.0)
//#define Omega_nurel_0 (0.0)
//#define Omega_rel_0 (0.0)
//#define Omega_de_0 (1.0-Omega_cb_0-Omega_nu_0)

//code switches and parameters
const int SWITCH_OUTPUT_ALLFLUIDS = 0; //output all nu,cb perturbations
const double PARAM_DETA0 = 1e-6; //default starting step size in eta
const double PARAM_EABS = 0; //absolute error tolerance
const double PARAM_EREL = 1e-6; //relative error tolerance
const int DEBUG_NU_MOMENTA = 1;

//////////////////////////////////// NEUTRINOS /////////////////////////////////

//total number of equations:
//  2 (delta and theta for CDM+Baryons)
#define N_EQ (2)

double eta_convert (double a) {
    return log(a/aeta_in);
}

double a_convert (double eta) {
    return aeta_in * exp(eta);
}

//homogeneous-universe momentum [eV], used to identify neutrino streams
const int FREE_TAU_TABLE = -4375643; //some negative integer, pick any
double tau_t_eV(int t){

  if(N_tau==0) return 0.0;
  static int init = 0;
  static double *tau_table_eV;

  if(!init){
    tau_table_eV = malloc(N_tau * sizeof(double));
    gsl_interp_accel *spline_accel = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_steffen,pcu_N);
    gsl_spline_init(spline,pcu_prob,pcu_tau,pcu_N);

    if(DEBUG_NU_MOMENTA) printf("#tau_t_eV: momenta [eV]:");
    
    for(int t=0; t<N_tau; t++){
      double prob = (0.5+t) / N_tau;
      tau_table_eV[t] = gsl_spline_eval(spline,prob,spline_accel);
      if(DEBUG_NU_MOMENTA) printf(" %g",tau_table_eV[t]);
    }

    if(DEBUG_NU_MOMENTA) printf("\n");
    gsl_spline_free(spline);
    gsl_interp_accel_free(spline_accel);
    init = 1;
  }

  if(t == FREE_TAU_TABLE){
    free(tau_table_eV);
    init = 0;
    return 0;
  }
  return tau_table_eV[t];
}

//speed -tau_t / tau0_t of each neutrino species
double v_t_eta(int t, double eta){
  double t_ma = tau_t_eV(t) / ( m_nu_eV * aeta_in*exp(eta) );
  return (t_ma<1 ? t_ma : 1);
}

double v2_t_eta(int t, double eta){
  double vt = v_t_eta(t,eta);
  return vt*vt;
}

//density ratio rho_t(eta)/rho_t(eta_stop) * aeta^2 and its log deriv
double Ec_t_eta(int t, double eta){ return 1.0/ ( aeta_in*exp(eta) ); }

double dlnEc_t_eta(int t, double eta){ return -1.0; }

//relativistic versions of the above, for Hubble rate calculation
double v2_t_eta_REL(int t, double eta){
  double m_aeta_tau = m_nu_eV * aeta_in*exp(eta) / tau_t_eV(t);
  return 1.0 / (1.0 + m_aeta_tau*m_aeta_tau);
}

double v_t_eta_REL(int t, double eta){ return sqrt(v2_t_eta_REL(t,eta)); }

double Ec_t_eta_REL(int t, double eta){
  double vt2 = v2_t_eta_REL(t,eta), aeta = aeta_in*exp(eta);
  if(1-vt2 < 1e-12){
    double ma_tau = m_nu_eV * aeta / tau_t_eV(t);
    return sqrt(1.0 + ma_tau*ma_tau) / (aeta*ma_tau); 
  }
  return 1.0 / (aeta * sqrt(1.0 - vt2));
}

double dlnEc_t_eta_REL(int t, double eta){ return -1.0 - v2_t_eta_REL(t,eta);}

//////////////////////////// HOMOGENEOUS COSMOLOGY /////////////////////////////

//equations of state: cdm, photons, DE
const double w_eos_cdm = 0, w_eos_gam = 0.333333333333333,
  w0_eos_de = -1.0, wa_eos_de = 0.0;

//a(eta)^2 * rho_de(eta) / rho_de_0 and its derivative
double Ec_de_eta(double eta){
  double aeta = aeta_in * exp(eta);
  return pow(aeta,-1.0 - 3.0*(w0_eos_de + wa_eos_de)) *
    exp(3.0*wa_eos_de*(aeta-1.0));
}

double dlnEc_de_eta(double eta){
  double aeta = aeta_in * exp(eta);
  return -1.0 - 3.0*(w0_eos_de + wa_eos_de) + 3.0*wa_eos_de*aeta;
}

//conformal hubble parameter
double Hc2_Hc02_eta(double eta){

  //scale factor
  double aeta = aeta_in * exp(eta), aeta2 = aeta*aeta, Ec_de = Ec_de_eta(eta);

  //sum Omega_{t,0} aeta^2 rho_t(eta)/rho_t_0 over CDM, photons, and DE
  double sum_OEc = Omega_cb_0/aeta + Omega_rel_0/aeta2 + Omega_de_0*Ec_de;
   
  //neutrinos
  if(Omega_nu_0 != 0.0) {
    for(int t=0; t<N_tau; t++) sum_OEc += Omega_nu_t_0 * Ec_t_eta_REL(t,eta);
  }

  return sum_OEc;
}

double Hc_eta(double eta){ return Hc0h * sqrt(Hc2_Hc02_eta(eta)); }

//d log(Hc) / d eta
double dlnHc_eta(double eta){
  
  double aeta = aeta_in*exp(eta), aeta2 = aeta*aeta;
  double pre = 1.0 / ( 2.0 * Hc2_Hc02_eta(eta) );
  
  double sum_OdEc = -(1.0 + 3.0*w_eos_cdm) *  Omega_cb_0/aeta //CDM
    - (1.0 + 3.0*w_eos_gam) * Omega_rel_0/aeta2 //photons + massless nu
    + dlnEc_de_eta(eta) * Omega_de_0 * Ec_de_eta(eta); //DE
  
  if(Omega_nu_0 != 0.0) {
    for(int t=0; t<N_tau; t++)//neutrino fluids
      sum_OdEc +=  dlnEc_t_eta_REL(t,eta) * Ec_t_eta_REL(t,eta) * Omega_nu_t_0;
  }
    //sum_OdEc += -(1.0 + 3.0*w_eos_cdm) * Omega_nu_0/aeta;
    
  return pre * sum_OdEc;
}

//density fraction in spatially-flat universe
double OF_eta(int F, double eta){
  
  double Hc02_Hc2 = 1.0/Hc2_Hc02_eta(eta), aeta = aeta_in*exp(eta);

  if(F == N_tau) //CDM
    return Omega_cb_0 * pow(aeta,-1.0-3.0*w_eos_cdm) * Hc02_Hc2;
  else if(F == N_tau+1) //photons + massless nu
    return Omega_rel_0 * pow(aeta,-1.0-3.0*w_eos_gam) * Hc02_Hc2;
  else if(F == N_tau+2) //dark energy, assumed Lambda
    return Omega_de_0 * Ec_de_eta(eta) * Hc02_Hc2;
  else if(F<0 || F>N_tau+3) return 0.0; //no fluids should have these indices
    return Omega_nu_t_0 * Ec_t_eta(F,eta) * Hc02_Hc2;
    //return Omega_nu_0 * pow(aeta, -1.0-3.0*w_eos_cdm) * Hc02_Hc2;
}

//Poisson equation for Phi
double Poisson(double eta, double k, double dcb, double dnu){
    double aeta = aeta_in * exp(eta), Hc02_Hc2 = 1.0/Hc2_Hc02_eta(eta);
  double Hc2 = Hc0h2 * Hc2_Hc02_eta(eta), pre = -1.5 * Hc2 / (k*k);
  double sum_Od = OF_eta(N_tau,eta) * dcb;
  for(int t=0; t<N_tau; t++) sum_Od += OF_eta(t,eta) * dnu;
    //sum_Od += Omega_nu_0 * pow(aeta, -1.0-3.0*w_eos_cdm) * Hc02_Hc2 * dnu;
  return pre * sum_Od;
}

//dimensionless superconformal time
#define NTCONF (10000)

double shconf_integrand(double eta, void *input){
  return 1.0 / (aeta_in * exp(eta) * sqrt(Hc2_Hc02_eta(eta)));
}

//linear interpolation of shconf vs eta
double shconf(double eta){

  static int init = 0;
  static gsl_interp_accel *spline_accel;
  static gsl_spline *lin_sconf;

  if(!init){
    lin_sconf = gsl_spline_alloc(gsl_interp_steffen,NTCONF);
    double Hcr2_Hc02 = Omega_gam_0;
    for(int t=0; t<N_tau; t++) Hcr2_Hc02 += Omega_nu_t_0*tau_t_eV(t)/m_nu_eV;
    
    double eta_i[NTCONF], s_i[NTCONF], deta = (eta_stop+3) / (NTCONF-1);
    eta_i[0] = -2;
    s_i[0] = 0;

    int ws_size = 10000, type = 6;
    double epsabs = PARAM_EABS, epsrel = PARAM_EREL, dsh, err, dum;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(ws_size);
    gsl_function F;
    F.function = &shconf_integrand;
    F.params = &dum;
    
    for(int i=1; i<NTCONF; i++){
      eta_i[i] = eta_i[i-1] + deta;
      gsl_integration_qag(&F, eta_i[i-1], eta_i[i], epsabs, epsrel,
                          ws_size, type, w, &dsh, &err);
      s_i[i] = s_i[i-1] + dsh;
    }

    gsl_spline_init(lin_sconf,eta_i,s_i,NTCONF);
    gsl_integration_workspace_free(w);
    init = 1;
  }

  return gsl_spline_eval(lin_sconf,eta,spline_accel);
}

//////////////////////////////// UTILITY FUNCTIONS /////////////////////////////

//minimum, maximum, absolute value functions
inline double fmin(double x,double y){ return (x<y ? x : y); }
inline double fmax(double x,double y){ return (x>y ? x : y); }
inline double fabs(double x){ return (x>0 ? x : -x); }

//fractional difference
double fdiff(double x, double y){
  return fabs(x-y) / (fabs(x)+fabs(y)+1e-100); }

//print all fluid perturbations
int print_results(double eta, const double *w, double dnu){
  printf("%g",eta);
  for(int i=0; i<N_EQ; i++) printf(" %g",w[i]);
  printf(" %g",dnu);
  printf("\n");
  fflush(stdout);
  return 0;
}

/////////////////////////// FAST LINEAR RESPONSE ///////////////////////////////

const double C_kfs = 0.759364372216230; //sqrt(ln(2) / zeta(3))

double kFs_gen(double k, double a, double m) {
  return 1.5 * sqrt(a*(Omega_cb_0+Omega_nu_0)) * m;
}

double K_i(double k, double a, double m) {
  double omega_nu_i = m / 93.141613;
  double fnu = omega_nu_i / ((Omega_cb_0 + Omega_nu_0) * h * h);
  return (kFs_gen(k,a,m)*kFs_gen(k,a,m)) / ((k+kFs_gen(k,a,m))*(k+kFs_gen(k,a,m)) - kFs_gen(k,a,m)*kFs_gen(k,a,m)*fnu);
}
/*
double delta_nu_flr(double eta, double k, double dcb) {
  double a = aeta_in * exp(eta);
  double C1, C2, C3;
  double fnu1, fnu2, fnu3, fcb;
  double omega_nu_1, omega_nu_2, omega_nu_3;

  omega_nu_1 = m_nu1 / 93.141613;
  omega_nu_2 = m_nu2 / 93.141613;
  omega_nu_3 = m_nu3 / 93.141613;

  fnu1 = omega_nu_1 / ((Omega_cb_0 + Omega_nu_0) * h * h);
  fnu2 = omega_nu_2 / ((Omega_cb_0 + Omega_nu_0) * h * h);
  fnu3 = omega_nu_3 / ((Omega_cb_0 + Omega_nu_0) * h * h);
  fcb  = Omega_cb_0 / (Omega_cb_0 + Omega_nu_0);

  C1 = - K_i(k,a,m_nu1)*(1.+fnu2*K_i(k,a,m_nu2))*(1.+fnu3*K_i(k,a,m_nu3)) / (-1. + fnu2*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3) + fnu1*K_i(k,a,m_nu1) * (fnu3*K_i(k,a,m_nu3) + fnu2*(K_i(k,a,m_nu2) + 2.*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3))));

  C2 = - (1.+fnu1*K_i(k,a,m_nu1))*K_i(k,a,m_nu2)*(1.+fnu3*K_i(k,a,m_nu3)) / (-1. + fnu2*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3) + fnu1*K_i(k,a,m_nu1) * (fnu3*K_i(k,a,m_nu3) + fnu2*(K_i(k,a,m_nu2) + 2.*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3))));

  C3 = - (1.+fnu1*K_i(k,a,m_nu1))*(1.+fnu2*K_i(k,a,m_nu2))*K_i(k,a,m_nu3) / (-1. + fnu2*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3) + fnu1*K_i(k,a,m_nu1) * (fnu3*K_i(k,a,m_nu3) + fnu2*(K_i(k,a,m_nu2) + 2.*fnu3*K_i(k,a,m_nu2)*K_i(k,a,m_nu3))));

  //printf("dnu = %f, dcb = %f\n", (fnu1*C1 + fnu2*C2 + fnu3*C3) * dcb / (fnu1 + fnu2 + fnu3), dcb);
  //printf("C1 = %f, C2 = %f, C3 = %f\n", C1, C2, C3);
  //printf("fnu1 = %f, fnu2 = %f, fnu3 = %f\n", fnu1, fnu2, fnu3);

  return (fnu1*C1 + fnu2*C2 + fnu3*C3) * fcb * dcb / (fnu1 + fnu2 + fnu3);
}
*/

double delta_nu_flr(double eta, double k, double dcb){
  double Hc = Hc_eta(eta), Omega_cb = OF_eta(N_tau,eta), Omega_nu = 0;
  for(int t=0; t<N_tau; t++) Omega_nu += OF_eta(t,eta);
  double Omega_m = Omega_cb + Omega_nu, f_nu = Omega_nu/Omega_m, f_cb=1.0-f_nu;
  double kfs = Hc * aeta_in*exp(eta)*m_T_nu * C_kfs * sqrt(Omega_m);
  double kpkfs = k + kfs, kpkfs2 = kpkfs*kpkfs, kfs2 = kfs*kfs;
  double dnu = dcb * kfs2*f_cb / (kpkfs2 - f_nu*kfs2);

  return dnu;
}

/////////////////////////////// DERIVATIVES ////////////////////////////////////

//cdm perturbation variables
//  y[0] = delta_{CDM}
//  y[1] = theta_{CDM}

int der(double eta, const double *y, double *dy, void *par){

  //initialize
  double *pd = (double *)par, k = pd[0], k_H = k/Hc_eta(eta), k2_H2 = k_H*k_H,
    dlnHc = dlnHc_eta(eta), Phi = Poisson(eta,k,y[0],delta_nu_flr(eta,k,y[0]));

  //cdm perturbations     
  dy[0] = y[1];
  dy[1] = -(1.0 + dlnHc)*y[1] -  k2_H2*Phi;
  
  return GSL_SUCCESS;
}

///////////////////////////////// EVOLUTION ////////////////////////////////////

#ifndef ETA_INTERVALS
#define ETA_INTERVALS (1024)
#endif

//evolve from aeta_in to input redshift
int evolve_to_z(double k, double z, double *w, double dcb_in){
  
  //initialize perturbations at eta=0
  double Omega_m_0 = Omega_cb_0+Omega_nu_0, fnu0 = Omega_nu_0/Omega_m_0;
  double aeta_eq = Omega_rel_0 / Omega_cb_0;
  for(int F=0; F<N_EQ; F++) w[F] = 0;

  //CDM+Baryon perturbations
  w[0] = dcb_in * (aeta_in + (2.0/3.0)*aeta_eq);
  w[1] = dcb_in * aeta_in;

  //print initial conditions
  if(SWITCH_OUTPUT_ALLFLUIDS) print_results(0,w,delta_nu_flr(0,k,w[0]));

  //initialize GSL ODE integration
  int status = GSL_SUCCESS;
  double eta0 = 0, aeta1 = 1.0/(1.0+z), eta1 = log(aeta1/aeta_in), par = k;
  gsl_odeiv2_system sys = {der, NULL, N_EQ, &par};  
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
						       gsl_odeiv2_step_rkf45,
						       PARAM_DETA0,
						       PARAM_EABS,
						       PARAM_EREL);
  gsl_odeiv2_driver_set_hmax(d, 0.1);

  //integrate to input redshift, printing results at regular intervals
  double deta = 0.01, eta = eta0, etai = deta;

  while(eta < eta1-1e-6
	&& status == GSL_SUCCESS){

    etai = fmin(eta+deta, eta1);
    status = gsl_odeiv2_driver_apply(d, &eta, etai, w);

    //make sure we got all the way to eta+deta
    if(fdiff(eta,etai) > 1e-6){
      printf("ERROR in evolve_to_z: failed at eta=%g, quitting.\n", eta);
      fflush(stdout);
      abort();
    }
    
    if(SWITCH_OUTPUT_ALLFLUIDS)
      print_results(eta,w,delta_nu_flr(eta,k,w[0]));
  }

  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

int evolve_step(double k, double z0, double z1, double *w){
  
  double aeta0 = 1.0/(1.0+z0), aeta1 = 1.0/(1.0+z1), par[] = {k};
  double eta0 = log(aeta0/aeta_in), eta1 = log(aeta1/aeta_in), eta = eta0;
  
  //initialize GSL ODE integration
  int status = GSL_SUCCESS;
  gsl_odeiv2_system sys = {der, NULL, N_EQ, par};  
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
                                                       gsl_odeiv2_step_rkf45,
                                                       PARAM_DETA0,
                                                       PARAM_EABS,
                                                       PARAM_EREL);
  gsl_odeiv2_driver_set_hmax(d, 0.1);

  //integrate to z1
  status = gsl_odeiv2_driver_apply(d, &eta, eta1, w);
  
  //make sure we got all the way to eta1
  if(fdiff(eta,eta1) > 1e-6
     || status != GSL_SUCCESS){
    printf("ERROR in evolve_step: ODE step failed at eta=%g, quitting.\n",
	   eta);
    fflush(stdout);
    abort();
  }
  
  if(SWITCH_OUTPUT_ALLFLUIDS)
    print_results(eta,w,delta_nu_flr(eta,k,w[0]));  

  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

int compute_growths(int nz, const double *z_n, int nk, const double *k_n,
		    double *Dcb, double *Fcb, double *Tcb, double *Dnu, double *dcb){

#pragma omp parallel for schedule(dynamic)
  for(int ik=0; ik<nk; ik++){
    double y[N_EQ], k=k_n[ik], aeta=1.0/(1.0+z_n[0]), eta=log(aeta/aeta_in);
    evolve_to_z(k,z_n[0],y,1.0);

    Dcb[0*nk + ik] = y[0];
    Fcb[0*nk + ik] = y[1] / y[0];
    Tcb[0*nk + ik] = y[1]; 
    Dnu[0*nk + ik] = delta_nu_flr(eta,k,y[0]);
    dcb[0*nk + ik] = y[0];

    for(int iz=1; iz<nz; iz++){
      evolve_step(k,z_n[iz-1],z_n[iz],y);
      aeta = 1.0/(1.0+z_n[iz]);
      eta = log(aeta/aeta_in);
      Dcb[iz*nk + ik] = y[0];
      Fcb[iz*nk + ik] = y[1] / y[0];
      Tcb[iz*nk + ik] = y[1];
      Dnu[iz*nk + ik] = delta_nu_flr(eta,k,y[0]);
      dcb[iz*nk + ik] = y[0];
    }

    //normalize growths
    if(z_n[nz-1] > 0) evolve_step(k,z_n[nz-1],0,y);
    for(int iz=0; iz<nz; iz++){
      Dcb[iz*nk + ik] /= y[0];
      //Tcb[iz*nk + ik] /= y[0];
      Dnu[iz*nk + ik] /= y[0];
    }
    
  }//end parallel for

  return 0;
}

double Hubble_Extern(double a) {
    return Hc2_Hc02_eta(eta_convert(a))/(a*a);
}

double f1(double a) {
    double omega_a;
    
    omega_a = Omega_cb_0 / (Omega_cb_0 + a * a * a * Omega_de_0);
    
    return pow(omega_a, 0.6);
}

//////////////////////////////////// MAIN //////////////////////////////////////
//various tests of code

int main(int argn, char *args[]){

  //initialize
  tau_t_eV(0);
  shconf(1);
  
  //compute nu and cb growth factors and rates over a range of k in parallel

#define NKD (512)
#define NZD (4)
  
  double kmin=1e-4, kmax=70, dlnk=log(kmax/kmin)/(NKD-1), k_n[NKD], z_n[NZD];
  for(int i=0; i<NKD; i++) k_n[i] = kmin * exp(dlnk*i);
  //for(int i=0; i<NZD; i++) z_n[i] = 1.0/(0.02*(i+1)) - 1.0;
  z_n[0]=99.;
  z_n[1]=49.;
  z_n[2]=3.;
  z_n[3]=0.;

  double Dnu[NZD*NKD], Dcb[NZD*NKD], Tcb[NZD*NKD], Fcb[NZD*NKD], dcb[NZD*NKD];
  compute_growths(NZD,z_n,NKD,k_n,Dcb,Fcb,Tcb,Dnu,dcb);

  for(int iz=0; iz<NZD; iz++){
    for(int ik=0; ik<NKD; ik++){
      int izk = iz*NKD + ik;
      printf("%g %g %g %g %g %g %g\n",z_n[iz],k_n[ik],Dcb[izk],Fcb[izk],Tcb[izk],Dnu[izk],dcb[izk]);
    }
    printf("\n\n");
  }

  tau_t_eV(FREE_TAU_TABLE);
  return 0;
}
