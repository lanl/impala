################################################################################
## get velocimetry data
################################################################################
ndat.vel<-1
nfeatures<-10
nsim<-1000

setwd('~/Desktop/immpala_data/PlateResultsTrial3//')


dat.vel<-matrix(nrow=ndat.vel,ncol=nfeatures)
dat.vel[1,]<-unlist(read.csv('~/git/immpala/code/features_cdf_obsCu-Cu.csv')) # odd are timings (micro seconds), even are velocities (m/s)



dat.vel.vec<-c(dat.vel)
#dat.vel[,c(2,4,6,8,10)]<-dat.vel[,c(2,4,6,8,10)]/10000

sim.vel<-array(dim=c(ndat.vel,nfeatures,nsim))
sim.vel[1,,]<-t(read.csv('features_cdfcu_ptw.csv'))


sim.vel[,c(2,4,6,8,10),]<-sim.vel[,c(2,4,6,8,10),]*10000 # get sims on m/s scale

sim.vel.mat<-array(sim.vel,dim=c(ndat.vel*nfeatures,nsim))

#inputs.sim.vel<-read.table('../Al.trial5.design.txt',head=T)
inputs.sim.vel<-read.table('OFHC-Cu.design.ptw.1000.txt',head=T)



## build emulator


 library(BASS)
# ho<-sample(1:1000,size = 50)
 n.pc<-8
 nc<-parallel::detectCores()-1
# mod.vel<-bassPCA(xx=inputs.sim.vel[-ho,],y=t(sim.vel.mat[,-ho]),n.pc=n.pc,n.cores = min(nc,n.pc),scale = T)
# 
# pmod<-predict(mod.vel,inputs.sim.vel[ho,],n.cores = 3,mcmc.use = 1:1000)
# 
# plot(apply(pmod,2:3,mean)[,c(1,3,5,7,9)],t(sim.vel.mat[c(1,3,5,7,9),ho]))
# abline(a=0,b=1,col=2)
# 
# plot(apply(pmod,2:3,mean)[,c(1,3,5,7,9)+1],t(sim.vel.mat[c(1,3,5,7,9)+1,ho]))
# abline(a=0,b=1,col=2)

mod.vel<-bassPCA(xx=inputs.sim.vel,y=t(sim.vel.mat),n.pc=n.pc,n.cores = min(nc,n.pc),scale = T)



################################################################################
## get taylor cylinder data
################################################################################

load('~/Desktop/immpala_data/TaylorTrial3/features.rda')



## build emulator
mod.tc<-bassPCA(inputs.sim.tc,sim.tc,scale=T,n.pc=3,n.cores = 3)


# pmod<-predict(mod.tc,inputs.sim.tc,n.cores = 3,mcmc.use = 1:1000)
# plot(apply(pmod,2:3,mean),sim.tc)
# abline(a=0,b=1,col=2)
# 
# pairs(cbind(t(sim.vel.mat)[1:nrow(sim.tc),],sim.tc))

################################################################################
## hoppy bar data
################################################################################
setwd('~/git/immpala/code')
dat<-list()
dat[[1]]<-read.table('../Cu-annealed/Cu20203.txt')
dat[[2]]<-read.table('../Cu-annealed/Cu40203.txt')
dat[[3]]<-read.table('../Cu-annealed/Cu60203.txt')
dat[[4]]<-read.table('../Cu-annealed/CuRT10-1.SRC.txt')
dat[[5]]<-read.table('../Cu-annealed/CuRT10-3.SRC.txt')
dat[[6]]<-read.table('../Cu-annealed/CuRT203.txt')


temps<-c(473,673,873,298,298,298) #+ 273.15 # +273.15 takes from C to K
edots<-c(2000,2000,2000,.1,.001,2000)# * 1e-6 # strain per second, using time units of microsecond
for(i in 1:length(dat)){
  dat[[i]][,2]<-dat[[i]][,2]*1e-5
  dat[[i]]<-dat[[i]][-1,] # can't get the zero with the models, might make sense to delete the first few sometimes
}

################################################################################
## connect python and R
################################################################################
# library(reticulate)
# smdl <- import_from_path("strength_models_add_ptw")
# 
# ##### JC
# 
# material_parameters = smdl$ModelParameters(update_parameters=smdl$update_parameters_JC)
# material_parameters$A      = 270.0e-5 # MBar
# material_parameters$B      = 470.0e-5 # MBar
# material_parameters$C      = 0.0105   # -
# material_parameters$n      = 0.600    # -
# material_parameters$m      = 1.200    # -
# material_parameters$Tref   = 298.0    # K
# material_parameters$Tmelt0 = 933.0    # K
# material_parameters$edot0  = 1.0e-6   # 1/mu-s
# material_parameters$rho0   = 2.683    # g/cc
# material_parameters$Cv0    = 0.900e-5 # MBar - cm^3
# material_parameters$G0     = 0.70     # MBar
# material_parameters$chi    = 0.90     # -
# 
# 
# material_model = smdl$MaterialModel(
#   parameters          = material_parameters,
#   flow_stress_model   = smdl$johnson_cook)
# 
# 
# f<-function(x,tt,edot){ # x a vector
#   strain_history = smdl$generate_strain_history(emax, edot, Nhist)
#   material_model$state = smdl$MaterialState(T=tt)
#   #material_model$state = smdl$MaterialState()
#   x<-as.list(x[1:5])
#   #x<-as.list(x[1:8])
#   material_model$parameters$update_parameters(material_model$parameters,x)
#   res<-smdl$compute_state_history(material_model,strain_history)
#   return(res[,2:3])
# }
# 
# 
# 
# 


################################################################################
## functions
################################################################################
unstandardize<-function(u){
  u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
}

get.se<-function(uu,tt,edot){
  mm<-mm.dat<-se<-list()
  for(i in 1:length(tt)){
    mm[[i]]<-f(unstandardize(uu),tt[i],edot[i])
    mm.dat[[i]]<-approx(mm[[i]][,1],mm[[i]][,2],xout=dat[[i]][,1])$y
    se[[i]]<-(mm.dat[[i]]-dat[[i]][,2])^2
  }
  unlist(se)
}


get.se.vel<-function(uu){
  uu<-unstandardize(uu)
  pred<-predict(mod.vel,uu[,c(1:11),drop=F],n.cores = 1,mcmc.use = sample(1000,1))
  (pred-dat.vel.vec)^2
}

get.se.tc<-function(uu){
  uu<-unstandardize(uu)
  pred<-predict(mod.tc,uu[,c(1:11),drop=F],n.cores = 1,mcmc.use = sample(1000,1))
  (pred-dat.tc)^2
}

get.alpha<-function(uu,s2.vel,s2.hop,s2.tc,w.vel,w.hop,w.tc,se.vel.curr,se.hop.curr,se.tc.curr){
  se.hop.cand<-get.se(uu,temps,edots)
  se.vel.cand<-get.se.vel(uu)
  se.tc.cand<-get.se.tc(uu)
  alpha<- -.5*(sum(se.hop.cand*w.hop/s2.hop) + sum(se.vel.cand*w.vel/s2.vel) + sum(se.tc.cand*w.tc/s2.tc) - sum(se.hop.curr*w.hop/s2.hop) - sum(se.vel.curr*w.vel/s2.vel) - sum(se.tc.curr*w.tc/s2.tc))
  return(list(alpha=alpha,se.hop=se.hop.cand,se.vel=se.vel.cand,se.tc=se.tc.cand))
}


constraints<-function(model,uu){
  if(any(uu<0 | uu>1))
    return(T)
  
  uu<-unstandardize(uu)
  if(model=='PTW'){
    if(uu[4]>uu[3] | uu[5]>uu[3] | uu[5]<max(uu[6],0) | uu[6]>min(uu[4],uu[5]) | uu[7]<uu[3])#if(uu[4]>uu[3] | uu[7]<uu[8] | uu[7]>uu[3] | uu[8]>min(uu[7],uu[4]) )#| uu[9]<uu[3] | uu[10]<uu[11])
      return(T)
  }
  return(F)
}




mcmc<-function(whop,wvel,wtc){
  require(BASS)
  require(mnormt)
  
  nmcmc=30000
  nburn=20000
  

  
  use.tc<-use.vel<-use.hop<-1
  if(wvel==0)
    use.vel<-0
  if(whop==0)
    use.hop<-0
  if(wtc==0)
    use.tc<-0
  
  n.tc<-3
  n.vel<-ndat.vel*nfeatures
  n.hop<-sum(unlist(lapply(dat,nrow)))
  n.tot<-n.vel+n.hop+n.tc
  n.tot.use<-n.vel*use.vel+n.hop*use.hop+n.tc*use.tc
  
  
  
  w.vel<-rep(1/n.vel,n.vel)*n.tot.use*wvel
  w.hop<-rep(1/n.hop,n.hop)*n.tot.use*whop
  w.tc<-rep(1/n.tc,n.tc)*n.tot.use*wtc
  
  #browser()
  
  u<-matrix(nrow=nmcmc,ncol=p)
  start.u<-runif(p)
  cont<-constraints(model,start.u)
  while(cont){
    start.u<-runif(p)
    cont<-constraints(model,start.u)
  }
  
  u[1,]<-start.u
  s2.vel<-matrix(nrow=nmcmc,ncol=n.vel)
  s2.hop<-matrix(nrow=nmcmc,ncol=n.hop)
  s2.tc<-matrix(nrow=nmcmc,ncol=n.tc)
  se.vel<-matrix(nrow=nmcmc,ncol=n.vel)
  se.hop<-matrix(nrow=nmcmc,ncol=n.hop)
  se.tc<-matrix(nrow=nmcmc,ncol=n.tc)
  
  
  #hist(sqrt(1/rgamma(1000,a.vel,b.vel)))
  #summary(sqrt(1/rgamma(10000,a.hop,b.hop)))
  
  a.vel<-rep(0,n.vel)
  a.hop<-rep(0,n.hop)
  a.tc<-rep(0,n.tc)
  b.vel<-rep(0,n.vel)
  b.hop<-rep(0,n.hop)
  b.tc<-rep(0,n.tc)
  
  
  
  # # vel sd: xvals mean: .01^2, r: (4*.03)^2
  # mm=.01 # mean in sd space
  # ss=.03 # sd in sd space
  # hist(sqrt(1/rgamma(1000000,2.5,.00025)),breaks=100,freq=F)
  # curve(dnorm(x,mm,ss),col=2,add=T)
  # 
  # # vel variance: yvals mean: 20, sd: 20 => a=3, b=40
  # mm=20
  # ss=20
  # hist(sqrt(1/rgamma(1000000,6,2000)),breaks=100,freq=F)
  # curve(dnorm(x,mm,ss),col=2,add=T)
  # 
  # # hop variance: mean: .0003, sd=.0003 a<-100, b<-1e-5
  # hist(sqrt(1/rgamma(100000,8,1e-6)))
  # 
  # # tc variance top: mean 1, sd 1
  # mm=1
  # ss=1
  # hist(sqrt(1/rgamma(1000000,2.5,2)),breaks=100,freq=F)
  # curve(dnorm(x,mm,ss),col=2,add=T)
  # 
  # # tc variance foot: mean .1, sd .2
  # mm=.1
  # ss=.2
  # hist(sqrt(1/rgamma(1000000,2,.1)),breaks=100,freq=F)
  # curve(dnorm(x,mm,ss),col=2,add=T)
  # 
  # # tc variance int: mean 5, sd 10
  # mm=5
  # ss=10
  # hist(sqrt(1/rgamma(1000000,2,100)),breaks=100,freq=F)
  # curve(dnorm(x,mm,ss),col=2,add=T)
  
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ make these priors informative, then debug, verify for all singular weight combinations
  
  ind.same.var.hop<-rep(1,n.hop) # all have same variance
  ind.same.var.vel<-rep(1:2,n.vel/2) # all x features have one variance and y features have another variance
  ind.same.var.tc<-1:n.tc # each feature has own variance
  
  a.vel<-rep(c(4,6),n.vel/2)
  b.vel<-rep(c(.00025,200),n.vel/2)
  
  a.hop<-rep(8,n.hop)
  b.hop<-rep(1e-6,n.hop)
  
  a.tc<-c(2,2.5,2.5)
  b.tc<-c(.01,.2,.2)
  
  
  # w.vel<-1/n.vel /3
  # w.hop<-1/n.hop /3 # each HB datapoint should have weight so that the sum are weighted .5?
  # w.tc<-1/n.tc /3
  # #w.vel*n.vel + w.hop*n.hop + w.tc*n.tc
  # w.vel<-0
  # w.hop<-0
  # w.tc<-1
  

  
  cand<-get.alpha(t(u[1,]),rep(1,n.vel),rep(1,n.hop),rep(1,n.tc),w.vel,w.hop,w.tc,rep(1,n.vel),rep(1,n.hop),rep(1,n.tc))
  se.vel[1,]<-cand$se.vel
  se.hop[1,]<-cand$se.hop
  se.tc[1,]<-cand$se.tc
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  count=0
  for(i in 2:nmcmc){
    
    for(j in 1:max(ind.same.var.vel)){
      indj<-which(ind.same.var.vel==j)
      if(any(w.vel[indj]>0))
        s2.vel[i,indj]<-1/rgamma(1,1/2+a.vel[indj[1]]+sum(w.vel[indj])/2,b.vel[indj[1]]+.5*sum(se.vel[i-1,indj]*w.vel[indj]))
      else
        s2.vel[i,indj]<-1
    }
    
    #browser()
    for(j in 1:max(ind.same.var.hop)){
      indj<-which(ind.same.var.hop==j)
      if(any(w.hop[indj]>0))
        s2.hop[i,indj]<-1/rgamma(1,1/2+a.hop[indj[1]]+sum(w.hop[indj])/2,b.hop[indj[1]]+.5*sum(se.hop[i-1,indj]*w.hop[indj]))
      else
        s2.hop[i,indj]<-1
    }
    
    for(j in 1:max(ind.same.var.tc)){
      indj<-which(ind.same.var.tc==j)
      if(any(w.tc[indj]>0))
        s2.tc[i,indj]<-1/rgamma(1,1/2+a.tc[indj[1]]+sum(w.tc[indj])/2,b.tc[indj[1]]+.5*sum(se.tc[i-1,indj]*w.tc[indj]))
      else
        s2.tc[i,indj]<-1
    }
    
    u[i,]<-u[i-1,]
    se.hop[i,]<-se.hop[i-1,]
    se.vel[i,]<-se.vel[i-1,]
    se.tc[i,]<-se.tc[i-1,]
    
    if(i>200 & i<nburn){
      mi<-max(1,i-300)
      S<-cov(u[mi:(i-1),])*cc+diag(eps*cc,p)
    }
    
    u.cand<-rmnorm(1,u[i-1,],S) # generate candidate
    if(constraints(model,u.cand)){ # constraint
      alpha<- -9999
    } else{
      
      cand<-get.alpha(t(u.cand),s2.vel[i,],s2.hop[i,],s2.tc[i,],w.vel,w.hop,w.tc,se.vel[i-1,],se.hop[i-1,],se.tc[i-1,])
      alpha<-cand$alpha
    }
    if(log(runif(1)) < alpha){
      u[i,]<-u.cand
      se.hop[i,]<-cand$se.hop
      se.vel[i,]<-cand$se.vel
      se.tc[i,]<-cand$se.tc
      count<-count+1
    }
    
    se.vel[i,]<-get.se.vel(t(u[i,]))
    se.tc[i,]<-get.se.tc(t(u[i,]))
    
    if(i%%100==0)
      cat('it:',i,'acc:',count,timestamp(quiet=T),'\n')
    
  }
  
  return(list(u=u,count=count,s2.vel=s2.vel,s2.hop=s2.hop,s2.tc=s2.tc,se.vel=se.vel,se.hop=se.hop,se.tc=se.tc))
}

################################################################################
## other
################################################################################

model='PTW'
x.rr<-matrix(c(.001,.1,
               0,10,
               .001,.05,
               .001,.05,
               0,.05,
               0,.05, 
               0,.1, 
               0,1,
               .01,1,
               10^-6,10^-1,
               0.030184,0.031316 # vel
),ncol=2,byrow=T)
p=nrow(x.rr)

emax   = 0.65
Nhist  = 100


save.image('~/Desktop/immpala_data/calib_start_ptw.rda')



################################################################################
## Bayesian inversion - MCMC
################################################################################

ps<-mclapply(1:8,function(i) mcmc(whop=1/3,wvel=1/3,wtc=1/3),mc.preschedule = F,mc.cores = 8) # should be done by 1:30


matplot(ps[[2]]$u,type='l')

j=0
j=j+1
plot(1,ylim=c(0,1),xlim=c(0,nrow(ps[[1]]$u)),main=j)
for(i in 1:length(ps)){
  lines(ps[[i]]$u[,j],col=i)
}

################################################################################
## results
################################################################################
burn<-1:29000

j=2
pred<-list()
for(ee in 1:length(dat)){
  pred[[ee]]<-do.call(cbind,mclapply(1:nrow(ps[[j]]$u[-burn,]),function(i) f(unstandardize(ps[[j]]$u[-burn,][i,]),temps[ee],edots[ee])[,2],mc.cores = nc))
}


col<-c(6:1,'burlywood4','darkgoldenrod1')
xx<-f(unstandardize(runif(p)),temps[1],edots[1])[,1]
xx1<-f(unstandardize(runif(p)),temps[2],edots[2])[,1]
ymax<-.004#max(unlist(pred),dat[[4]][,2],na.rm = T)
ss<-colMeans(sqrt(ps[[j]]$s2.hop[-burn,]))[1]


pdf('../Docs/hbfit_JC_both_cu.pdf',height=5,width=9)
par(mfrow=c(2,3),mar=c(0,0,0,0),oma=c(5,5,5,5))
for(i in 1:length(dat)){
  plot(xx,xx,ylim=c(0,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
  points(dat[[i]],lwd=1,col=col[i],cex=.5,pch=1)
  mult<-1
  segments(x0=dat[[i]][,1],y0=dat[[i]][,2]-ss*mult,y1=dat[[i]][,2]+ss*mult,col=col[i],lwd=.5)
  qq<-apply(pred[[i]],1,quantile,probs=c(.025,.975))
  mm<-rowMeans(pred[[i]])
  lines(xx,mm,lwd=2,col=col[i],lty=1)
  lines(xx,qq[1,],col=col[i],lty=1)
  lines(xx,qq[2,],col=col[i],lty=1)
  if(i==1)
    legend('topright',c(paste('temperature (C):',temps[i]-273),paste('strain rate (1/s):',edots[i])),bty='n')
  else
    legend('topright',legend=c(temps[i]-273,edots[i]),bty='n')
  if(i %in% c(1,4))
    axis(2)
  if(i %in% 4:8)
    axis(1)
}
mtext('JC fit',3,outer = T,cex=2,line=2)
mtext('strain',1,outer = T,cex=1,line=3)
mtext('stress',2,outer = T,cex=1,line=3)
dev.off()


load('~/Desktop/immpala_data/calib_hop.rda')
load('~/Desktop/immpala_data/calib_vel.rda')
load('~/Desktop/immpala_data/calib_tc.rda')
load('~/Desktop/immpala_data/calib_hopvel.rda')
load('~/Desktop/immpala_data/calib_equalWeight.rda')

udf<-ps[[1]]$u[-burn,]
for(j in 1:length(ps))
  udf<-rbind(udf,ps[[j]]$u[-burn,])

udf<-data.frame(udf)
names(udf)<-c('A','B','C','n','m')

#names(udf)<-c('theta','p','s0','sInf','kappa','gamma','y0','yInf')

library(psych)
#pdf('../Docs/caliball_JC_both_cu.pdf')
pairs.panels(udf,
             method = "pearson", # correlation method
             cor = F,
             hist.col = "#00AFBB",
             smooth=F,
             density = TRUE,  # show density plots
             ellipses = F, # show correlation ellipses
             xlim=c(0,1),ylim=c(0,1)
)
#dev.off()






dat.vel.pred<-matrix(c(predict(mod.vel,t(unstandardize(ps[[j]]$u[20000])[c(1:5,6,8)]),n.cores = 1,mcmc.use = 1)),nrow=1)
matplot(t(t(1:5)),t(t(dat.vel[,c(1,3,7,9,5)+1])),type='l',col=1:3,lty=1)
matplot(t(t(1:5)),t(t(dat.vel.pred[,c(1,3,7,9,5)+1])),type='l',col=1:3,lty=2,add=T)

dat.vel.arr<-array(dim=c(1,10,1000))
k<-0
for(i in (1:nrow(ps[[j]]$u))[-burn]){
  k<-k+1
  dat.vel.arr[,,k]<-matrix(c(predict(mod.vel,t(unstandardize(ps[[j]]$u[i,])[c(1:5,6,8)]),n.cores = 1,mcmc.use = 1)),nrow=1)
}

#pdf('../Docs/vfit_JC_both_cu.pdf',height=4,width=8)
obs<-read.table('~/Desktop/Cu-Cu_Thomas_PlateImpact/ofhc-cu-symmetric-impact.txt')
plot(obs,type='b',xlim=c(.8,1.3),xlab='time',ylab='velocity',lwd=2,main='Cu Flyer Plate')

use<-apply(dat.vel.arr[1,c(1,3,7,9,5),],2,function(x) all(diff(x)>0) & all(x>0))
use<-apply(dat.vel.arr[1,c(1,3,7,9,5)+1,],2,function(x) all(x>0)) & use

matplot(dat.vel.arr[1,c(1,3,7,9,5),use],dat.vel.arr[1,c(1,3,7,9,5)+1,use],type='l',col='lightgrey',add=T)
lines(obs,type='b',lwd=2)
legend('bottomright',c('velocimetry measurements','velocimetry features: posterior predictive samples'),pch=c(1,NA),lty=c(3,1),lwd=c(3,1),col=c(1,'lightgrey'),bty='n')
#dev.off()



pred.tc.mat<-matrix(nrow=1000,ncol=3)
k<-0
for(i in (1:nrow(ps[[j]]$u))[-burn]){
  k<-k+1
  pred.tc.mat[k,]<-c(predict(mod.tc,t(unstandardize(ps[[j]]$u[i,])[c(1:5,7,8)]),mcmc.use=sample(1000,1)))
}

hist(sim.tc[,1],freq=F)
abline(v=dat.tc[1],col=2)
abline(v=mean(pred.tc.mat[,1]),col=3)
lines(density(pred.tc.mat[,1]),col=3)

hist(sim.tc[,2],freq=F)
abline(v=dat.tc[2],col=2)
abline(v=mean(pred.tc.mat[,2]),col=3)
lines(density(pred.tc.mat[,2]),col=3)

hist(sim.tc[,3],freq=F)
abline(v=dat.tc[3],col=2)
abline(v=mean(pred.tc.mat[,3]),col=3)
lines(density(pred.tc.mat[,3]),col=3)

sqrt(colMeans(ps[[j]]$s2.tc[-burn,]))

pairs(pred.tc.mat)
library(rgl)
plot3d(pred.tc.mat)
rgl.spheres(dat.tc, col = 'red',radius=.1)


pdf('../Docs/vfit_JC_both_cu.pdf',height=4,width=8)
par(mfrow=c(1,1),mar=c(0,0,0,0),oma=c(5,5,5,5))
matplot(dat.vel.arr[1,c(1,3,7,9,5)+1,seq(1,1000,10)],type='l',col='lightgrey')
lines(dat.vel[1,c(1,3,7,9,5)+1],lwd=3,lty=3)
legend(1,400,c('velocimetry features','posterior predictive samples'),lty=c(3,1),lwd=c(3,1),col=c(1,'lightgrey'),bty='n')
legend('top',c('impact velocity (km/s): 0.2040','Flyer/target thickness (mm): 3.950/5.965'),bty='n')
dev.off()


par(mfrow=c(2,4),mar=c(0,0,0,0),oma=c(5,5,5,5))
for(i in 1:length(dat)){
  plot(xx,xx,ylim=c(.002,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
  points(dat[[i]],lwd=1,col=col[i],cex=.5,pch=1)
  #mult<-1
  #segments(x0=dat[[i]][,1],y0=dat[[i]][,2]-ss*mult,y1=dat[[i]][,2]+ss*mult,col=col[i],lwd=.5)
  #qq<-apply(pred[[i]],1,quantile,probs=c(.025,.975))
  mm<-rowMeans(pred[[i]])
  lines(xx,mm,lwd=2,col=col[i],lty=1)
  #lines(xx,qq[1,],col=col[i],lty=1)
  #lines(xx,qq[2,],col=col[i],lty=1)
  if(i %in% c(1,5))
    axis(2)
  if(i %in% 5:8)
    axis(1)
}



save.image('jc_fit.rda')
