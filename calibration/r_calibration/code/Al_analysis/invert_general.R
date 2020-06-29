######################################################
## get data

ndat.vel<-3
nfeatures<-10
nsim<-1000

setwd('~/git/realTime/AlAl/data_trial5/')

dat.vel<-matrix(nrow=ndat.vel,ncol=nfeatures)
dat.vel[1,]<-unlist(read.csv('features_cdf_obs104S.csv')) # odd are timings (micro seconds), even are velocities (m/s)
dat.vel[2,]<-unlist(read.csv('features_cdf_obs105S.csv'))
dat.vel[3,]<-unlist(read.csv('features_cdf_obs106S.csv'))

dat.vel.vec<-c(dat.vel)
#dat.vel[,c(2,4,6,8,10)]<-dat.vel[,c(2,4,6,8,10)]/10000

sim.vel<-array(dim=c(ndat.vel,nfeatures,nsim))
sim.vel[1,,]<-t(read.csv('features_cdf104S.csv'))
sim.vel[2,,]<-t(read.csv('features_cdf105S.csv'))
sim.vel[3,,]<-t(read.csv('features_cdf106S.csv'))

sim.vel[,c(2,4,6,8,10),]<-sim.vel[,c(2,4,6,8,10),]*10000 # get sims on m/s scale

sim.vel.mat<-array(sim.vel,dim=c(ndat.vel*nfeatures,nsim))

inputs.sim.vel<-read.table('../Al.trial5.design.txt',head=T)


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



######################################################
## formulate models/emulators


library(BASS)
ho<-sample(1:1000,size = 50)
n.pc<-10
nc<-parallel::detectCores()-1
mod<-bassPCA(xx=inputs.sim.vel[-ho,],y=t(sim.vel.mat[,-ho]),n.pc=n.pc,n.cores = min(nc,n.pc))




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
x.rr<-matrix(c(.0005,.01,
               .0005,.01,
               0,.03,
               0,1.5
               ,0,3,
               .0192,.0216,
               .0334,.0376,
               .0455,.0513,
               .2,.5,
               0,.1,
               0,.2
),ncol=2,byrow=T)




f<-function(x,exp.params){ # x a vector
  tt<-exp.params[1]
  edot<-exp.params[2]
  strain_history = smdl$generate_strain_history(emax, edot, Nhist)
  material_model$state = smdl$MaterialState(T=tt)
  #material_model$state = smdl$MaterialState()
  x<-as.list(x[1:5])
  material_model$parameters$update_parameters(material_model$parameters,x)
  res<-smdl$compute_state_history(material_model,strain_history)
  return(res[,2:3])
}

unstandardize<-function(u){
  u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
}

p<-nrow(x.rr)











get.sse<-function(uu,tt,edot){
  mm<-mm.dat<-sse<-list()
  for(i in 1:length(tt)){
    mm[[i]]<-f(unstandardize(uu),tt[i],edot[i])
    mm.dat[[i]]<-approx(mm[[i]][,1],mm[[i]][,2],xout=dat[[i]][,1])$y
    sse[[i]]<-sum((mm.dat[[i]]-dat[[i]][,2])^2)
  }
  return(sum(unlist(sse)))
}

get.sse.vel<-function(uu){
  pred<-predict(mod,unstandardize(uu),n.cores = 1,mcmc.use = 1)
  sum((pred-dat.vel.vec)^2)
}

get.alpha<-function(uu,s2.vel,s2.hop,w.vel,w.hop,sse.vel.curr,sse.hop.curr){
  #browser()
  sse.hop.cand<-get.sse(uu,temps,edots)
  sse.vel.cand<-get.sse.vel(uu)
  alpha<- -.5*(sse.hop.cand/s2.hop*w.hop + sse.vel.cand/s2.vel*w.vel - sse.hop.curr/s2.hop*w.hop - sse.vel.curr/s2.vel*w.vel)
  return(list(alpha=alpha,sse.hop=sse.hop.cand,sse.vel=sse.vel.cand))
}

getAlpha<-function(uu,curr,dat,prior){
  alpha<-0
  sse<-rep(NA,length(dat))
  for(i in 1:length(dat)){
    for(j in 1:length(dat[[i]]$data)){
      sse[i]<-sse[i]+sum((dat[[i]]$model(uu,dat[[i]]$exper.vars[[j]]) - dat[[i]]$data[[j]])^2) # just have one model, but evaluate it more if it is an emu
    }
    alpha<-alpha-.5*(sse[i]-curr$sse[i])/curr$s2[i]*dat$w[i]
  }
  return(list(sse=sse,alpha=alpha))
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







my.dat<-list()

my.dat[[1]]<-list()
my.dat[[1]]$has.emu<-F
my.dat[[1]]$model<-f
my.dat[[1]]$data<-list()
for(j in 1:8){
  my.dat[[1]]$data[[j]]<-dat[[j]][,2]
}

my.dat[[2]]<-list()
my.dat[[2]]$has.emu<-T
my.dat[[2]]$model<-function(x,exp.params){
  predict(mod,x,mcmc.use=sample.int(1000,1))
}
my.dat[[2]]$data<-list()
for(j in 1:3){
  my.dat[[2]]$data[[j]]<-dat.vel[j,]
}


prior<-list()
prior$s2<-list() # IG priors
prior$s2[[1]]<-list()
prior$s2[[2]]<-list()
prior$s2[[1]]$a<-1
prior$s2[[1]]$b<-1
prior$s2[[2]]$a<-1
prior$s2[[2]]$b<-1
prior$theta<-list()
for(i in 1:nrow(x.rr)){
  prior$theta[[i]]<-list()
  prior$theta[[i]]$ldist<-function(x){
    dunif(x,x.rr[i,1],x.rr[i,2],log=T)
  }
}

######################################################
## general inversion function

invert<-function(dat,prior,nmcmc=10000){
  nd<-length(dat)
  p<-length(prior$theta)
  
  
  u<-matrix(nrow=nmcmc,ncol=p)
  start.u<-runif(p)
  cont<-constraints(model,start.u)
  while(cont){
    start.u<-runif(p)
    cont<-prior$constraints(model,start.u)
  }
  
  u[1,]<-start.u
  
  s2<-matrix(nrow=nd,ncol=nmcmc)
  s2[1,]<-1
  
  cand<-getAlpha(t(u[1,]),1,1,.5,.5,1,1)
  sse.vel[1]<-cand$sse.vel
  sse.hop[1]<-cand$sse.hop
  
  
  n.vel<-ndat.vel*nfeatures
  n.hop<-sum(unlist(lapply(dat,nrow)))
  n.tot<-n.vel+n.hop
  w.vel<-1/n.vel * .5
  w.hop<-1/n.hop * .5 # each HB datapoint should have weight so that the sum are weighted .5?
  w.vel*n.vel + w.hop*n.hop
  w.vel<-1
  w.hop<-1
  
  
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  count=0
  for(m in 2:nmcmc){
    
    for(i in 1:nd)
      s2[m,i]<-1/rgamma(1,prior$s2$n[i]/2+prior$s2$a,prior$s2$b+.5*curr$sse[i])
    
    u[i,]<-u[i-1,]
    if(i>200){
      mi<-max(1,i-300)
      S<-cov(u[mi:(i-1),])*cc+diag(eps*cc,p)
    }
    
    u.cand<-rmnorm(1,u[i-1,],S) # generate candidate
    if(constraints(dat$model,u.cand)){ # constraint
      alpha<- -9999
    } else{
      
      cand<-getAlpha()
      alpha<-cand$alpha
    }
    if(log(runif(1)) < alpha){
      u[i,]<-u.cand
      curr<-cand
      count<-count+1
    }
    
    if(dat$emu...)
      sse.vel[i]<-get.sse.vel(t(u[i,]))
    
    if(i%%100==0)
      cat('it:',i,'acc:',count,timestamp(quiet=T),'\n')
    
  }
  
  return(list(u=u,count=count,s2=s2))
}