// Experimental float64 Cu PTW history. One independent history per CUDA thread.
// Equations mirror physical_models_functions.py; build with --fmad=false.
extern "C" __global__ void cu_history(
    const double* times, const double* rates, const double* initial,
    const double* params, const double* c, const double avogadro_root,
    const int n, const int nh, double* out, double* final_aux)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= n) return;
    const double theta=params[j], p=params[n+j], s0=params[2*n+j];
    const double beta=params[3*n+j], si=params[4*n+j], kappa=params[5*n+j];
    const double lg=params[6*n+j], y0=params[7*n+j], yi=params[8*n+j];
    const double y1=params[9*n+j], y2=params[10*n+j];
    double T=initial[j], eps=initial[n+j], stress=initial[2*n+j];
    double Cv=0.0, tm=0.0;
    for (int i=0; i<nh; ++i) {
        const int prev = i == 0 ? 0 : i-1;
        const double rate=rates[prev*n+j];
        const double dt=i == 0 ? 0.0 : times[i*n+j]-times[(i-1)*n+j];
        Cv=c[1]+c[2]*T+c[3]*(T*T);
        const double rho=c[4]+c[5]*T;
        T=T+double(rate>1.0e-6)*c[0]*stress*rate*dt/(Cv*rho);
        eps=eps+rate*dt;
        tm=c[6]+c[7]*rho;
        const double cold=c[8]*pow(rho/c[9],4.0/3.0)*exp(
            6.0*c[10]*(1.0/cbrt(c[9])-1.0/cbrt(rho))
            +2.0*c[11]/c[12]*(pow(c[9],-c[12])-pow(rho,-c[12])));
        double G=cold*(1.0-c[13]*(T/tm));
        if(T>=tm) G=0.0;
        if(G<0.0) G=0.0;
        const double afact=(4.0/3.0)*3.141592653589793*c[14]/c[15];
        const double xi=0.5*cbrt(afact)*sqrt(G/c[14])*avogadro_root;
        const double log_x=log(xi/rate);
        const double er=erf(kappa*(T/tm)*(lg+log_x));
        const double saturation1=s0-(s0-si)*er;
        const double saturation2=s0*exp(beta*(-lg-log_x));
        const double ts=saturation1>saturation2 ? saturation1 : saturation2;
        const double ay=y0-(y0-yi)*er;
        const double by=y1*exp(-y2*(lg+log_x));
        const double cy=s0*exp(-beta*(lg+log_x));
        const double dy=by<cy ? by : cy;
        const double ty=ay>dy ? ay : dy;
        double scaled=ts;
        if(p>1.0e-10 && fabs(ts-ty)>1.0e-10) {
            const double a1=p*(ts-ty)/(s0-ty);
            const double a2=(eps*p*theta)/(s0-ty)/(exp(a1)-1.0);
            const double thelog=log(1.0-(1.0-exp(-a1))*exp(-a2));
            scaled=ts+(s0-ty)*thelog/p;
        }
        if(p<=1.0e-10 && ts>ty)
            scaled=ts-(ts-ty)*exp(-eps*theta/(ts-ty));
        stress=scaled*G*2.0;
        out[(i*6+0)*n+j]=times[i*n+j];
        out[(i*6+1)*n+j]=eps;
        out[(i*6+2)*n+j]=stress;
        out[(i*6+3)*n+j]=T;
        out[(i*6+4)*n+j]=G;
        out[(i*6+5)*n+j]=rho;
    }
    final_aux[j]=Cv;
    final_aux[n+j]=tm;
}
