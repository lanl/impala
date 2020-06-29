################################################################################
## hoppy bar data
################################################################################
setwd('~/git/immpala/code')
dat<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T-196C.csv')
plot(dat)
dat[,2]<-dat[,2]*1e-5
dat[1,1]<-0
dat<-dat[-1,] # can't get the zero with the models, might make sense to delete the first few sometimes

################################################################################
## connect python and R
################################################################################
library(reticulate)
smdl <- import_from_path("strength_models_add_ptw")

emax   = 0.5
edot   = .001 # strain per second, using time units of microsecond
tt     = -196+273.15 # +273.15 takes from C to K
Nhist  = 100
strain_history = smdl$generate_strain_history(emax, edot, Nhist)





##### JC

material_parameters = smdl$ModelParameters(update_parameters=smdl$update_parameters_JC)
material_parameters$A      = 270.0e-5 # MBar
material_parameters$B      = 470.0e-5 # MBar
material_parameters$C      = 0.0105   # -
material_parameters$n      = 0.600    # -
material_parameters$m      = 1.200    # -
material_parameters$Tref   = 298.0    # K
material_parameters$Tmelt0 = 933.0    # K
material_parameters$edot0  = 1.0e-6   # 1/mu-s
material_parameters$rho0   = 2.683    # g/cc
material_parameters$Cv0    = 0.900e-5 # MBar - cm^3
material_parameters$G0     = 0.70     # MBar
material_parameters$chi    = 0.90     # -

# define material model
#   - notice we are using several default sub-component models
#   - example: constant_specific_heat, etc.

material_model = smdl$MaterialModel(
  parameters          = material_parameters,
  flow_stress_model   = smdl$johnson_cook)

# ----------------- Step 3: integrate/evaluate model over history ------------------
# evaluate state history from material model
  results_01 = smdl$compute_state_history(material_model, strain_history)

x.rr<-matrix(c(.0005,.01,.0005,.01,0,.03,0,1.5,0,3),ncol=2,byrow=T)



##### PTW


# ----------------- Step 5: define another material model -------------------------
#    here we set up a PTW model

material_parameters=smdl$ModelParameters(update_parameters=smdl$update_parameters_PTW)
material_parameters$theta   = 0.0183   # dimensionless
material_parameters$p       = 3.0      #  "
material_parameters$s0      = 0.019497 #  "
material_parameters$sInf    = 0.002902 #  "
material_parameters$kappa   = 0.3276   #  "
material_parameters$gamma   = 1.0E-5   #  "
material_parameters$y0      = 0.002948 #  "
material_parameters$yInf    = 0.001730 #  "
material_parameters$y1      = 0.094    #  "
material_parameters$y2      = 0.575    #  "
material_parameters$beta    = 0.25     #  "
material_parameters$alpha   = 0.2      #  "
material_parameters$matomic = 27.0     # approximate atomic mass
material_parameters$Tref    = 298.0    # K
material_parameters$Tmelt0  = 933.0    # K
material_parameters$rho0    = 2.683    # g/cc
material_parameters$Cv0     = 0.900e-5 # MBar - cm^3
material_parameters$G0      = 0.70     # MBar
#looks like this sets 90% of p dV goes into internal energy
#ptw fit above used 100% but difference should be small
material_parameters$chi     = 0.90     # -

# define material model
#   - notice we are using several default sub-component models
#   - example: constant_specific_heat, etc.

material_model = smdl$MaterialModel(
  parameters          = material_parameters,
  flow_stress_model   = smdl$ptw_stress)

# ----------------- Step 6: integrate/evaluate model over history ------------------
# evaluate state history from material model
  results_03 = smdl$compute_state_history(material_model, strain_history)

x.rr<-matrix(c(.0001,.05,
               .0001,5,
               .0001,.05,
               .0001,.005,
               .0001,.5,
               .000001,.0001,
               .0001,.005,
               .0001,.005
               ),ncol=2,byrow=T)

# material_parameters$theta   = 0.0183   # .0001,.05
# material_parameters$p       = 3.0      # .0001,5
# material_parameters$s0      = 0.019497 # .0001,.05
# material_parameters$sInf    = 0.002902 # .0001,.005
# material_parameters$kappa   = 0.3276   # .0001,.5
# material_parameters$gamma   = 1.0E-5   # .000001,.0001
# material_parameters$y0      = 0.002948 # .0001,.005
# material_parameters$yInf    = 0.001730 # .0001,.005






f<-function(x,tt,edot){ # x a vector
  strain_history = smdl$generate_strain_history(emax, edot, Nhist)
  material_model$state = smdl$MaterialState(T=tt)
  #material_model$state = smdl$MaterialState()
  x<-as.list(x)
  material_model$parameters$update_parameters(material_model$parameters,x)
  #material_model$parameters
  
  res<-smdl$compute_state_history(material_model,strain_history)
  return(res[,2:3])
}





unstandardize<-function(u){
  u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
}

p<-nrow(x.rr)

u<-runif(p)
plot(f(unstandardize(u),tt,edot))
# points(dat,col=2)






################################################################################
## Bayesian inversion - MCMC
################################################################################

library(mnormt)
library(parallel)

nmcmc<-10000
u<-matrix(nrow=nmcmc,ncol=p)
u[1,]<-runif(p)
s2<-sse<-NA
a<-100
b<-1e-5
hist(sqrt(1/rgamma(1000,a,b)))
summary(sqrt(1/rgamma(10000,a,b)))

mm<-f(unstandardize(u[1,]),tt,edot)
mm.dat<-approx(mm[,1],mm[,2],xout=dat[,1])$y
sse[1]<-sum((mm.dat-dat[,2])^2)

cc<-2.4^2/p
eps<-1e-5
S<-diag(p)*eps
n<-nrow(dat)
count=0
for(i in 2:nmcmc){
  
  s2[i]<-1/rgamma(1,n/2+a,b+.5*sse[i-1])
  
  u[i,]<-u[i-1,]
  sse[i]<-sse[i-1]
  
  if(i>200){
    mi<-max(1,i-300)
    S<-cov(u[mi:(i-1),])*cc+diag(eps*cc,p)
  }
  
  u.cand<-rmnorm(1,u[i-1,],S) # generate candidate
  if(any(u.cand<0 | u.cand>1)){ # constraint
    alpha<- -9999
  } else{
    mm<-f(unstandardize(u.cand),tt,edot)
    mm.dat<-approx(mm[,1],mm[,2],xout=dat[,1])$y
    
    sse.cand<-sum((mm.dat-dat[,2])^2)
    
    alpha<- -.5/s2[i]*(sse.cand-sse[i-1])
  }
  if(log(runif(1))<alpha){
    u[i,]<-u.cand
    sse[i]<-sse.cand
    count<-count+1
  }
  
  if(i%%100==0)
    cat('it:',i,'acc:',count,timestamp(quiet=T),'\n')
  
}

count/nmcmc



plot(sse,type='l')
plot(s2,type='l')
matplot(u,type='l')

################################################################################
## results
################################################################################

burn<-1:8000
pred<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),tt,edot)[,2],mc.cores = 3))

pdf('../Docs/pred_inSampPTW.pdf')
xx<-f(unstandardize(runif(p)),tt,edot)[,1]
matplot(xx,pred,type='l',col='lightgrey',ylim=c(0,max(pred)),ylab='stress',xlab='strain')
points(dat)
mult=1
ss<-mean(sqrt(s2[-burn]))
segments(x0=dat[,1],y0=dat[,2]-ss*mult,y1=dat[,2]+ss*mult,col=1,lwd=.5)
legend('topleft',c('measurements','posterior predictive samples'),pch=c(1,-1),lty=c(-1,1),col=c(1,'lightgrey'),bty='n')
legend('bottomright','-196 degrees C')
dev.off()

udf<-data.frame(u)
names(udf)<-c('A','B','C','n','m')

library(psych)
pdf('../Docs/calib1PTW.pdf')
pairs.panels(udf[-burn,],
             method = "pearson", # correlation method
             cor = F,
             hist.col = "#00AFBB",
             smooth=F,
             density = TRUE,  # show density plots
             ellipses = F, # show correlation ellipses
             xlim=c(0,1),ylim=c(0,1)
)
dev.off()
####################################################################################



dat.star<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T125C.csv')
dat.star[,2]<-dat.star[,2]*1e-5


temps<-c(-196,125,125,-196,25,100,100,25) + 273.15 # +273.15 takes from C to K
edots<-c(.001,.001,.1,2000,2500,3000,3500,7000)# * 1e-6 # strain per second, using time units of

temp.star<- 125 + 273.15 # +273.15 takes from C to K
edot.star<-.001

pred.star<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),temp.star,edot.star)[,2],mc.cores = 3))


pdf('../Docs/pred_outSampPTW.pdf')
matplot(xx,pred.star,type='l',col='lightgrey',ylim=c(0,max(pred)),ylab='stress',xlab='strain')
points(dat.star[-1,])
segments(x0=dat.star[,1],y0=dat.star[,2]-ss*mult,y1=dat.star[,2]+ss*mult,col=1,lwd=.5)
legend('topleft',c('measurements','posterior predictive samples'),pch=c(1,-1),lty=c(-1,1),col=c(1,'lightgrey'),bty='n')
legend('bottomright','125 degrees C')
dev.off()
