################################################################################
## get velocimetry data
################################################################################
ndat.vel<-1
nfeatures<-10
nsim<-1000

setwd('~/Desktop/Cu-Cu_Thomas_PlateImpact/')


dat.vel<-matrix(nrow=ndat.vel,ncol=nfeatures)
dat.vel[1,]<-unlist(read.csv('features_cdf_obsCu-Cu.csv')) # odd are timings (micro seconds), even are velocities (m/s)

dat.vel.vec<-c(dat.vel)
#dat.vel[,c(2,4,6,8,10)]<-dat.vel[,c(2,4,6,8,10)]/10000

sim.vel<-array(dim=c(ndat.vel,nfeatures,nsim))
sim.vel[1,,]<-t(read.csv('features_cdfCu-Cu.csv'))


sim.vel[,c(2,4,6,8,10),]<-sim.vel[,c(2,4,6,8,10),]*10000 # get sims on m/s scale

sim.vel.mat<-array(sim.vel,dim=c(ndat.vel*nfeatures,nsim))

#inputs.sim.vel<-read.table('../Al.trial5.design.txt',head=T)
inputs.sim.vel<-read.table('Cu-Cu_Thomas.design.yjc.1000.txt',head=T)


################################################################################
## build emulator
################################################################################

library(BASS)
ho<-sample(1:1000,size = 50)
n.pc<-5
nc<-parallel::detectCores()-1
mod<-bassPCA(xx=inputs.sim.vel[-ho,],y=t(sim.vel.mat[,-ho]),n.pc=n.pc,n.cores = min(nc,n.pc))

mod2<-bassPCA(xx=inputs.sim.vel[-ho,],y=t(sim.vel.mat[c(1,3,5,7,9),-ho]),n.pc=n.pc,n.cores = min(nc,n.pc))

BASS:::bassPCAsetup

pmod<-predict(mod,inputs.sim.vel[ho,],n.cores = 3,mcmc.use = 1:1000)

plot(apply(pmod,2:3,mean)[,c(1,3,5,7,9)],t(sim.vel.mat[c(1,3,5,7,9),ho]))
abline(a=0,b=1,col=2)





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
library(reticulate)
smdl <- import_from_path("strength_models_add_ptw")



emax   = 0.65
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
x.rr<-matrix(c(.0001,.01,
               .0001,.01,
               .0002,.03,
               .0001,1.5,
               .002,3,
               .03,.0315,
               .3,.6
               ),ncol=2,byrow=T)











f<-function(x,tt,edot){ # x a vector
  strain_history = smdl$generate_strain_history(emax, edot, Nhist)
  material_model$state = smdl$MaterialState(T=tt)
  #material_model$state = smdl$MaterialState()
  x<-as.list(x[1:5])
  #x<-as.list(x[1:8])
  material_model$parameters$update_parameters(material_model$parameters,x)
  res<-smdl$compute_state_history(material_model,strain_history)
  return(res[,2:3])
}



unstandardize<-function(u){
  u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
}

p<-nrow(x.rr)

u<-runif(p)
plot(f(unstandardize(u),temps[1],edots[1]))




################################################################################
## Bayesian inversion - MCMC
################################################################################

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


constraints<-function(model,uu){
  if(any(uu<0 | uu>1))
    return(T)
  
  if(model=='PTW'){
    if(uu[4]>uu[3] | uu[7]<uu[8] | uu[7]>uu[3] | uu[8]>min(uu[7],uu[4]) )#| uu[9]<uu[3] | uu[10]<uu[11])
      return(T)
  }
  return(F)
}


library(mnormt)
library(parallel)

mcmc<-function(nmcmc=30000,a.vel=0,b.vel=0,a.hop=0,b.hop=0){
  
  u<-matrix(nrow=nmcmc,ncol=p)
  start.u<-runif(p)
  cont<-constraints(model,start.u)
  while(cont){
    start.u<-runif(p)
    cont<-constraints(model,start.u)
  }
    
  u[1,]<-start.u
  s2.vel<-s2.hop<-sse.vel<-sse.hop<-NA

  
  #hist(sqrt(1/rgamma(1000,a.vel,b.vel)))
  #summary(sqrt(1/rgamma(10000,a.hop,b.hop)))
  
  cand<-get.alpha(t(u[1,]),1,1,.5,.5,1,1)
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
  for(i in 2:nmcmc){
    
    s2.vel[i]<-278#1/rgamma(1,n.vel/2+a.vel,b.vel+.5*sse.vel[i-1])
    s2.hop[i]<-2.7e-8#1/rgamma(1,n.hop/2+a.hop,b.hop+.5*sse.hop[i-1])
    
    u[i,]<-u[i-1,]
    sse.hop[i]<-sse.hop[i-1]
    sse.vel[i]<-sse.vel[i-1]
    
    if(i>200){
      mi<-max(1,i-300)
      S<-cov(u[mi:(i-1),])*cc+diag(eps*cc,p)
    }
    
    u.cand<-rmnorm(1,u[i-1,],S) # generate candidate
    if(constraints(model,u.cand)){ # constraint
      alpha<- -9999
    } else{
      
      cand<-get.alpha(t(u.cand),s2.vel[i],s2.hop[i],w.vel,w.hop,sse.vel[i-1],sse.hop[i-1])
      alpha<-cand$alpha
    }
    if(log(runif(1)) < alpha){
      u[i,]<-u.cand
      sse.hop[i]<-cand$sse.hop
      sse.vel[i]<-cand$sse.vel
      count<-count+1
    }
    
    sse.vel[i]<-get.sse.vel(t(u[i,]))
    
    if(i%%100==0)
      cat('it:',i,'acc:',count,timestamp(quiet=T),'\n')
    
  }

  return(list(u=u,count=count,s2.vel=s2.vel,s2.hop=s2.hop,sse.vel=sse.vel,sse.hop=sse.hop))
}


model="JC"
ps<-mclapply(1:10,mcmc,nmcmc=30000,mc.preschedule = F,mc.cores = nc)


matplot(ps[[1]]$u,type='l')


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

j=1
pred<-list()
for(ee in 1:length(dat)){
  pred[[ee]]<-do.call(cbind,mclapply(1:nrow(ps[[j]]$u[-burn,]),function(i) f(unstandardize(ps[[j]]$u[-burn,][i,]),temps[ee],edots[ee])[,2],mc.cores = 3))
}


col<-c(6:1,'burlywood4','darkgoldenrod1')
xx<-f(unstandardize(runif(p)),temps[1],edots[1])[,1]
xx1<-f(unstandardize(runif(p)),temps[2],edots[2])[,1]
ymax<-.004#max(unlist(pred),dat[[4]][,2],na.rm = T)
ss<-mean(sqrt(ps[[1]]$s2.hop[-burn]))


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

udf<-data.frame(ps[[1]]$u)
names(udf)<-c('A','B','C','n','m')

#names(udf)<-c('theta','p','s0','sInf','kappa','gamma','y0','yInf')


library(psych)
pdf('../Docs/caliball_JC_both_cu.pdf')
pairs.panels(udf[-burn,1:5],
             method = "pearson", # correlation method
             cor = F,
             hist.col = "#00AFBB",
             smooth=F,
             density = TRUE,  # show density plots
             ellipses = F, # show correlation ellipses
             xlim=c(0,1),ylim=c(0,1)
)
dev.off()





plot(dat.vel.vec,c(predict(mod,t(unstandardize(a$u[30000,])),n.cores = 1,mcmc.use = 1))); abline(a=0,b=1,col=2)

dat.vel.pred<-matrix(c(predict(mod,t(unstandardize(ps[[1]]$u[20000,])),n.cores = 1,mcmc.use = 1)),nrow=1)
matplot(t(t(1:5)),t(t(dat.vel[,c(1,3,7,9,5)+1])),type='l',col=1:3,lty=1)
matplot(t(t(1:5)),t(t(dat.vel.pred[,c(1,3,7,9,5)+1])),type='l',col=1:3,lty=2,add=T)

dat.vel.arr<-array(dim=c(1,10,1000))
k<-0
for(i in 29001:30000){
  k<-k+1
  dat.vel.arr[,,k]<-matrix(c(predict(mod,t(unstandardize(ps[[1]]$u[i,])),n.cores = 1,mcmc.use = 1)),nrow=1)
}

pdf('../Docs/vfit_JC_both_cu.pdf',height=4,width=8)
obs<-read.table('~/Desktop/Cu-Cu_Thomas_PlateImpact/ofhc-cu-symmetric-impact.txt')
plot(obs,type='b',xlim=c(.8,1.3),xlab='time',ylab='velocity',lwd=2,main='Cu Flyer Plate')

use<-apply(dat.vel.arr[1,c(1,3,7,9,5),],2,function(x) all(diff(x)>0) & all(x>0))
use<-apply(dat.vel.arr[1,c(1,3,7,9,5)+1,],2,function(x) all(x>0)) & use

matplot(dat.vel.arr[1,c(1,3,7,9,5),use],dat.vel.arr[1,c(1,3,7,9,5)+1,use],type='l',col='lightgrey',add=T)
lines(obs,type='b',lwd=2)
legend('bottomright',c('velocimetry measurements','velocimetry features: posterior predictive samples'),pch=c(1,NA),lty=c(3,1),lwd=c(3,1),col=c(1,'lightgrey'),bty='n')
dev.off()



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
