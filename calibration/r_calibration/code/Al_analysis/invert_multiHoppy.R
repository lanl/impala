################################################################################
## hoppy bar data
################################################################################
setwd('~/git/immpala/code')
dat<-list()
dat[[1]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T-196C.csv')
dat[[2]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_001T125C.csv')
dat[[3]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S0_1T125C.csv')
dat[[4]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S2000T-196C.csv')
dat[[5]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S2500T25C.csv')
dat[[6]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S3000T100C.csv')
dat[[7]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S3500T200C.csv')
dat[[8]]<-read.csv('../Al-5083/Stress-Strain_Data/Gray94_Al5083_S7000T25C.csv')
temps<-c(-196,125,125,-196,25,100,100,25) + 273.15 # +273.15 takes from C to K
edots<-c(.001,.001,.1,2000,2500,3000,3500,7000)# * 1e-6 # strain per second, using time units of microsecond
for(i in 1:length(dat)){
  dat[[i]][,2]<-dat[[i]][,2]*1e-5
  dat[[i]]<-dat[[i]][-1,] # can't get the zero with the models, might make sense to delete the first few sometimes
}


dat.star<-dat[[5]]
dat[[5]]<-NULL
temps<-temps[-5]
edots=edots[-5]

xlim<-range(unlist(lapply(dat,function(x) range(x[,1]))))
ylim<-range(unlist(lapply(dat,function(x) range(x[,2]))))

plot(1,type='n',ylim=ylim,xlim=xlim)
for(i in 1:length(dat))
  lines(dat[[i]],col=i)



################################################################################
## connect python and R
################################################################################
library(reticulate)
smdl <- import_from_path("strength_models_add_ptw")



emax   = 0.5
Nhist  = 100




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

model='JC'
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



model='PTW'
x.rr<-matrix(c(.001,.1, #theta
                0,10, # p
               .01,.1, # s0
               .001,.1, #sinf #max of s0
               .01,1, # kappa
               1e-6,.1, # gamma
               0,.1, # y0 # yinf, s0
               0,.1
               # , # yinf max of min(sinf,y0)
               # .01,.1, # y1
               # .1,1, # y2
               # .1,.35
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
plot(f(unstandardize(u),temps[1],edots[1]))
plot(f(unstandardize(u),temps[1],edots[3]))
plot(f(unstandardize(u),temps[2],edots[2]))
plot(f(unstandardize(u),temps[3],edots[3]))
plot(f(unstandardize(u),temps[4],edots[4]))
plot(f(unstandardize(u),temps[5],edots[5]))
# points(dat,col=2)


system.time(f(unstandardize(u),temps[1],edots[1]))


#uu<- c(300.0e-5, 400.0e-5, 0.0105, 0.6,  1.2)
#plot(f(uu,298,2500e-6),type='l')
#lines(f(uu,298,.001e-6),col=2)

uu<-u[30000,]
plot(f(unstandardize(uu),77.15,1.0e-09),type='l')
lines(f(unstandardize(uu),77.15,2.0e-03),col=2)

get.sse<-function(uu,tt,edot){
  mm<-mm.dat<-sse<-list()
  for(i in 1:length(tt)){
    mm[[i]]<-f(unstandardize(uu),tt[i],edot[i])
    mm.dat[[i]]<-approx(mm[[i]][,1],mm[[i]][,2],xout=dat[[i]][,1])$y
    sse[[i]]<-sum((mm.dat[[i]]-dat[[i]][,2])^2)
  }
  return(sum(unlist(sse)))
}

constraints<-function(model,uu){
  if(any(uu<0 | uu>1))
    return(T)
    
  if(model=='PTW'){
    if(uu[4]>uu[3] | uu[7]<uu[8] | uu[7]>uu[3] | uu[8]>min(uu[7],uu[4]) )#| uu[9]<uu[3] | uu[10]<uu[11])
      return(T)
  }
  return(F)
}


################################################################################
## Bayesian inversion - MCMC
################################################################################

library(mnormt)
library(parallel)

mcmc<-function(dummy){
nmcmc<-30000
u<-matrix(nrow=nmcmc,ncol=p)
start.u<-runif(p)
cont<-constraints(model,start.u)
while(cont){
  start.u<-runif(p)
  cont<-constraints(model,start.u)
}
  
u[1,]<-start.u
s2<-sse<-NA
#a<-0#100
#b<-0#1e-8
a<-100
b<-1e-5
#hist(sqrt(1/rgamma(1000,a,b)))
#summary(sqrt(1/rgamma(10000,a,b)))

sse[1]<-get.sse(u[1,],temps,edots)

cc<-2.4^2/p
eps<-1e-10
S<-diag(p)*eps
n<-sum(unlist(lapply(dat,nrow)))
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
  if(constraints(model,u.cand)){ # constraint
    alpha<- -9999
  } else{
    
    sse.cand<-get.sse(u.cand,temps,edots)
    
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

return(list(u=u,count=count,s2=s2,sse=sse))
}



ps<-mclapply(1:20,function(i) mcmc(i) ,mc.preschedule = F,mc.cores = 20)

count/nmcmc


plot(sse,type='l')
plot(s2,type='l')
matplot(u,type='l')

################################################################################
## results
################################################################################

burn<-1:20000
pred<-list()
for(ee in 1:length(dat)){
  pred[[ee]]<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),temps[ee],edots[ee])[,2],mc.cores = 3))
}

#pred[[4]]<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),temps[4],edots[4])[,2],mc.cores = 3))
#pred[[1]]<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),temps[1],edots[1])[,2],mc.cores = 3))


pred.star<-do.call(cbind,mclapply(1:nrow(u[-burn,]),function(i) f(unstandardize(u[-burn,][i,]),25+273.15,2500*1e-6)[,2],mc.cores = 3))








col<-c(6:1,'burlywood4','darkgoldenrod1')
xx<-f(unstandardize(runif(p)),temps[1],edots[1])[,1]
xx1<-f(unstandardize(runif(p)),temps[2],edots[2])[,1]
ymax<-.01#max(unlist(pred),dat[[4]][,2],na.rm = T)
ss<-mean(sqrt(s2[-burn]))


pdf('../Docs/hbfit_PTW_loo.pdf',height=5,width=9)
par(mfrow=c(2,4),mar=c(0,0,0,0),oma=c(5,5,5,5))
for(i in 1:4){
  plot(xx,xx,ylim=c(.002,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
  points(dat[[i]],lwd=1,col=col[i],cex=.5,pch=1)
  mult<-1
  segments(x0=dat[[i]][,1],y0=dat[[i]][,2]-ss*mult,y1=dat[[i]][,2]+ss*mult,col=col[i],lwd=.5)
  qq<-apply(pred[[i]],1,quantile,probs=c(.025,.975))
  mm<-rowMeans(pred[[i]])
  lines(xx,mm,lwd=2,col=col[i],lty=1)
  lines(xx,qq[1,],col=col[i],lty=1)
  lines(xx,qq[2,],col=col[i],lty=1)
  if(i==1)
    legend('topright',c(paste('temperature (C):',temps[i]-273.15),paste('strain rate (1/s):',edots[i])),bty='n')
  else
    legend('topright',legend=c(temps[i]-273.15,edots[i]),bty='n')
  if(i %in% c(1,5))
    axis(2)
  if(i %in% 5:8)
    axis(1)
}

plot(xx,xx,ylim=c(.002,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
points(dat.star,lwd=1,col=col[5],cex=.5,pch=1)
mult<-1
segments(x0=dat.star[,1],y0=dat.star[,2]-ss*mult,y1=dat.star[,2]+ss*mult,col=col[5],lwd=.5)
qq<-apply(pred.star,1,quantile,probs=c(.025,.975))
mm<-rowMeans(pred.star)
lines(xx,mm,lwd=2,col=col[5],lty=1)
lines(xx,qq[1,],col=col[5],lty=1)
lines(xx,qq[2,],col=col[5],lty=1)
legend('topright',legend=c(25,2500),bty='n')
axis(2)
axis(1)
text(.2,.008,'held out',cex=2)

for(i in 5:7){
  plot(xx,xx,ylim=c(.002,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
  points(dat[[i]],lwd=1,col=col[i+1],cex=.5,pch=1)
  mult<-1
  segments(x0=dat[[i]][,1],y0=dat[[i]][,2]-ss*mult,y1=dat[[i]][,2]+ss*mult,col=col[i+1],lwd=.5)
  qq<-apply(pred[[i]],1,quantile,probs=c(.025,.975))
  mm<-rowMeans(pred[[i]])
  lines(xx,mm,lwd=2,col=col[i+1],lty=1)
  lines(xx,qq[1,],col=col[i+1],lty=1)
  lines(xx,qq[2,],col=col[i+1],lty=1)
  if(i==1)
    legend('topright',c(paste('temperature (C):',temps[i]-273.15),paste('strain rate (1/s):',edots[i])),bty='n')
  else
    legend('topright',legend=c(temps[i]-273.15,edots[i]),bty='n')
  
    axis(1)
}

mtext('PTW fit',3,outer = T,cex=2,line=2)
mtext('strain',1,outer = T,cex=1,line=3)
mtext('stress',2,outer = T,cex=1,line=3)
dev.off()







udf<-data.frame(u)
names(udf)<-c('A','B','C','n','m')

names(udf)<-c('theta','p','s0','sInf','kappa','gamma','y0','yInf')


library(psych)
pdf('../Docs/caliball_JC_loo.pdf')
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

save.image('jc_fit.rda')
