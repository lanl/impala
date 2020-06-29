


astep<-function(sz,ct){
  for(s in 1:length(sz)){
    if(ct[s]<20)
      sz[s]<-sz[s]*.5
    if(ct[s]>30)
      sz[s]<-sz[s]*1.1
  }
  return(sz)
}


get.se<-function(uu){
  se<-NA
  unlist(mclapply(1:length(dat.plastic),function(i){
    #for(i in 1:length(dat.plastic)){
    mm<-f(unstandardize(uu),as.numeric(dat.info.plastic$temp[i]),as.numeric(dat.info.plastic$strainRate[i]))
    mm.dat<-approx(mm[,1],mm[,2],xout=dat.plastic[[i]][,1])$y
    se[i]<-sum((mm.dat-dat.plastic[[i]][,2])^2)
  },mc.cores=4))
}



get.se<-function(uu){
  #se<-NA
  #  unlist(mclapply(1:length(dat.plastic),function(i){
  #for(i in 1:length(dat.plastic)){
  mm<-f(unstandardize(uu),temp.calib,strainRate.calib)
  mm.dat<-approx(mm[,1],mm[,2],xout=dat.calib[,1],rule=c(2,2))$y
  sum((mm.dat-dat.calib[,2])^2)
  # },mc.cores=4))
}

# R<-exp(-as.matrix(dist(dat.calib[,1]))^2/.01^2)+diag(length(dat.calib[,1]))*1e-10
# R[1:10,1:10]
# Rinv<-solve(R)
# get.se<-function(uu){
#   #se<-NA
#   #  unlist(mclapply(1:length(dat.plastic),function(i){
#   #for(i in 1:length(dat.plastic)){
#   mm<-f(unstandardize(uu),temp.calib,strainRate.calib)
#   mm.dat<-approx(mm[,1],mm[,2],xout=dat.calib[,1],rule=c(2,2))$y
#   c(t(mm.dat-dat.calib[,2])%*%Rinv%*%(mm.dat-dat.calib[,2]))
#   # },mc.cores=4))
# }

system.time(aa<-get.se(rep(.5,11)))

get.alpha<-function(uu,s2.hop,se.hop.curr){
  se.hop.cand<-get.se(uu)
  alpha<- -.5*(sum(se.hop.cand/s2.hop) - sum(se.hop.curr/s2.hop) )
  return(list(alpha=alpha,se.hop=se.hop.cand))
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



p=12
mcmc<-function(){
  require(BASS)
  require(mnormt)
  
  nmcmc=10000
  nburn=5000
  
  u<-matrix(nrow=nmcmc,ncol=p)
  start.u<-runif(p)
  cont<-constraints(model,start.u)
  while(cont){
    start.u<-runif(p)
    cont<-constraints(model,start.u)
  }
  
  n<-length(dat.plastic[[ii]][,1])
  n.hop=1
  u[1,]<-start.u
  s2.hop<-matrix(nrow=nmcmc,ncol=n.hop)
  se.hop<-matrix(nrow=nmcmc,ncol=n.hop)
  
  a.hop<-rep(0,n.hop)
  b.hop<-rep(0,n.hop)
  
  # # hop variance: mean: .0003, sd=.0003 a<-100, b<-1e-5
  # hist(sqrt(1/rgamma(100000,8,1e-7)))
  a.hop<-rep(8,n.hop)
  b.hop<-rep(1e-7,n.hop)
  
  cand<-get.alpha(t(u[1,]),rep(1,n.hop),rep(1,n.hop))
  se.hop[1,]<-cand$se.hop
  
  cc<-2.4^2/p
  eps<-1e-5^2
  S<-diag(p)*.00001
  S<-SS*cc
  
  stepsize<-rep(.5,p)
  
  count<-count100<-rep(0,p)
  
  
  for(i in 2:nmcmc){
    
    s2.hop[i,]<-1/rgamma(1,n/2+a.hop,b.hop+.5*sum(se.hop[i-1,]))
    
    u[i,]<-u[i-1,]
    se.hop[i,]<-se.hop[i-1,]
    
    
    
    for(jj in 1:p){
      u.cand<-u[i,]
      u.cand[jj]<-runif(1,max(u.cand[jj]-stepsize[jj],0),min(u.cand[jj]+stepsize[jj],1))
      
      if(constraints("PTW",unstandardize(u.cand)))
        alpha<- -9999
      else{
        cand<-get.alpha(t(u.cand),s2.hop[i,],se.hop[i-1,])
        alpha<-cand$alpha
        if(is.na(alpha))
          alpha<- -9999
      }
      
      if(log(runif(1)) < alpha){
        u[i,]<-u.cand
        se.hop[i,]<-cand$se.hop
        count[jj]<-count[jj]+1
        count100[jj]<-count100[jj]+1
      }
    }
    

    
    
    
    if(i%%100==0){
      if(i<nburn & i>1000)
        stepsize<-astep(stepsize,count100)
      cat('it:',i,'acc:',count100,timestamp(quiet=T),'\n')
      count100<-rep(0,p)
    }
    
    
    
  }
  
  return(list(u=u,count=count,s2.hop=s2.hop,se.hop=se.hop))
}

library(parallel)
ps<-mclapply(1:3,function(x) mcmc(),mc.cores = 3)

#SS<-cov(ps[[2]]$u[450000:500000,])

# probably need to do some tempering here, since there appear to be modes in PTW parameter space that are difficult to get between?  **But why is parameter 12 stuck when it contributes nothing** (if stuck in a mode, that would still be able to move)?  Perhaps I should try single site moves first.  Or maybe I need a covariance matrix in the likelihood.

nburn<-10000
jj<-5000
nthin<-10



kk<-4
par(mfrow=c(6,2),mar=c(1,3,1,3))
for(kk in 1:12){
  plot(ps[[1]]$u[,kk],type='l',ylim=c(0,1))
  lines(ps[[2]]$u[,kk],col=2)
  lines(ps[[3]]$u[,kk],col=3)
}

pairs(rbind(ps[[1]]$u[seq(nburn,jj,nthin),],ps[[2]]$u[seq(nburn,jj,nthin),],ps[[3]]$u[seq(nburn,jj,nthin),]),xlim=c(0,1),ylim=c(0,1),cex=.1,col=rep(1:3,each=length(seq(nburn,jj,nthin))))


plot(dat.calib[,1],dat.calib[,2])
lines(f(unstandardize(ps[[1]]$u[jj,]),temp.calib,strainRate.calib))
lines(f(unstandardize(ps[[2]]$u[jj,]),temp.calib,strainRate.calib),col=2)
lines(f(unstandardize(ps[[3]]$u[jj,]),temp.calib,strainRate.calib),col=3)

lines(f(unstandardize(c(ps[[2]]$u[jj,1:9],ps[[3]]$u[jj,10],ps[[2]]$u[jj,11:11])),temp.calib,strainRate.calib),col=4)

lines(f(unstandardize(rep(.5,11)),temp.calib,strainRate.calib),col=5)


get.se(ps[[1]]$u[jj,])/ps[[1]]$s2.hop[jj,]
get.se(ps[[2]]$u[jj,])/ps[[2]]$s2.hop[jj,]
get.se(ps[[3]]$u[jj,])/ps[[3]]$s2.hop[jj,]

get.alpha(ps[[1]]$u[jj,],ps[[2]]$s2.hop[jj,],ps[[2]]$se.hop[jj,])

get.alpha(ps[[2]]$u[jj,],ps[[1]]$s2.hop[jj,],ps[[1]]$se.hop[jj,])


get.alpha(c(ps[[2]]$u[jj,1:8],ps[[1]]$u[jj,9],ps[[2]]$u[jj,10:11]),ps[[2]]$s2.hop[jj,],ps[[2]]$se.hop[jj,])
