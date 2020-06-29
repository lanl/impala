calib.u<-runif(p)
cont<-constraints(model,calib.u)
while(cont){
  calib.u<-runif(p)
  cont<-constraints(model,calib.u)
}

temp.calib<-293
strainRate.calib<-10
dat.calib<-f(unstandardize(calib.u),temp.calib,strainRate.calib)
dat.calib[,2]<-(dat.calib[,2]+rnorm(100,0,.00005))#*1e5



astep.mat<-function(sz.mat,ct.mat){
  for(s in 1:ncol(sz.mat)){
    sz.mat[,s]<-astep(sz.mat[,s],ct.mat[,s])
  }
  sz.mat
}

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

get.alpha<-function(uu,s2.hop,se.hop.curr,itemp){
  se.hop.cand<-get.se(uu)
  alpha<- -.5*(sum(se.hop.cand/s2.hop) - sum(se.hop.curr/s2.hop) )*itemp
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




mcmc<-function(){
  require(BASS)
  require(mnormt)
  
  nmcmc=1000
  nburn=500
  
  ntemp<-ntemps<-length(itemp.ladder)
  
  u<-array(dim=c(nmcmc,p,ntemp))
  
  for(tt in 1:ntemps){
    start.u<-runif(p)
    cont<-constraints(model,start.u)
    while(cont){
      start.u<-runif(p)
      cont<-constraints(model,start.u)
    }
    u[1,,tt]<-start.u
  }
  
  n<-length(dat.plastic[[ii]][,1])
  n.hop=1
  s2.hop<-matrix(nrow=nmcmc,ncol=tt)
  se.hop<-matrix(nrow=nmcmc,ncol=tt)
  
  a.hop<-rep(0,n.hop)
  b.hop<-rep(0,n.hop)
  
  # # hop variance: mean: .0003, sd=.0003 a<-100, b<-1e-5
  # hist(sqrt(1/rgamma(100000,8,1e-7)))
  a.hop<-rep(8,n.hop)
  b.hop<-rep(1e-7,n.hop)
  
  for(tt in 1:ntemp){
    cand<-get.alpha(t(u[1,,tt]),rep(1,n.hop),rep(1,n.hop),itemp.ladder[tt])
    se.hop[1,tt]<-cand$se.hop
  }

  
  stepsize<-matrix(.5,nrow=p,ncol=ntemps)
  
  count<-count100<-matrix(0,nrow=p,ncol=ntemps)
  count.swap<-0
  swap.vals<-matrix(nrow=nmcmc,ncol=2)
  
  lpost<-function(s2.hop,se.hop){
    (-n/2-a.hop-1)*log(s2.hop) +
      -1/s2.hop*(se.hop/2 + b.hop)
  }
  
  for(i in 2:nmcmc){
    
    for(tt in 1:ntemps)
      s2.hop[i,tt]<-BASS:::rigammaTemper(1,n/2+a.hop,b.hop+.5*sum(se.hop[i-1,tt]),itemper = itemp.ladder[tt])
    
    u[i,,]<-u[i-1,,]
    se.hop[i,]<-se.hop[i-1,]
    
    
    for(tt in 1:ntemps){
    
      for(jj in 1:p){
        u.cand<-u[i,,tt]
        u.cand[jj]<-runif(1,max(u.cand[jj]-stepsize[jj,tt],0),min(u.cand[jj]+stepsize[jj,tt],1))
        
        if(constraints("PTW",u.cand))
          alpha<- -9999
        else{
          cand<-get.alpha(t(u.cand),s2.hop[i,tt],se.hop[i-1,tt],itemp.ladder[tt])
          alpha<-cand$alpha
          if(is.na(alpha))
            alpha<- -9999
        }
        
        if(log(runif(1)) < alpha){
          u[i,,tt]<-u.cand
          se.hop[i,tt]<-cand$se.hop
          count[jj,tt]<-count[jj,tt]+1
          count100[jj,tt]<-count100[jj,tt]+1
        }
      }
    
    }

    if(i>start.temper & ntemp>1){
      #browser()
      swap<-sample(1:ntemp,size=2) # candidate swap
      alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*(lpost(s2.hop[i,swap[1]],se.hop[i,swap[1]])-lpost(s2.hop[i,swap[2]],se.hop[i,swap[2]]))
      if(log(runif(1))<alpha){ # swap states
        temp<-u[i,,swap[1]]
        u[i,,swap[1]]<-u[i,,swap[2]]
        u[i,,swap[2]]<-temp
        
        temp<-se.hop[i,swap[1]]
        se.hop[i,swap[1]]<-se.hop[i,swap[2]]
        se.hop[i,swap[2]]<-temp
        
        temp<-s2.hop[i,swap[1]]
        s2.hop[i,swap[1]]<-s2.hop[i,swap[2]]
        s2.hop[i,swap[2]]<-temp

        
        count.swap<-count.swap+1
        swap.vals[i,]<-sort(swap)
      }
    }
    
    
    
    if(i%%100==0){
      if(i<nburn & i>100)
        stepsize<-astep(stepsize,count100)
      cat('it:',i,'acc:',count100,timestamp(quiet=T),'\n')
      count100<-matrix(0,nrow=p,ncol=ntemps)
    }
    
    
  }
  
  return(list(u=u,count=count,s2.hop=s2.hop,se.hop=se.hop,count=count,count.swap=count.swap,swap.vals=swap.vals))
}

itemp.ladder<-1/(1.5^(0:8))#c(1,.8,.6,.4,.2,.1,.05,1/20)
start.temper=100
ps<-mcmc()

library(parallel)
ps<-mclapply(1:3,function(x) mcmc(),mc.cores = 3)

#SS<-cov(ps[[2]]$u[450000:500000,])

# probably need to do some tempering here, since there appear to be modes in PTW parameter space that are difficult to get between?  **But why is parameter 12 stuck when it contributes nothing** (if stuck in a mode, that would still be able to move)?  Perhaps I should try single site moves first.  Or maybe I need a covariance matrix in the likelihood.

dev.off()
plot(jitter(ps$swap.vals))
ps$count.swap

nburn<-500
jj<-1000
nthin<-1


par(mfrow=c(6,2),mar=c(1,3,1,3))
for(kk in 1:11){
  plot(ps$u[,kk,9],type='l',ylim=c(0,1))
  abline(h=calib.u[kk],col=2)
}

plot(dat.calib[,1],dat.calib[,2])
lines(f(unstandardize(ps$u[1000,,7]),temp.calib,strainRate.calib))

dev.off()
matplot(ps$s2.hop[-c(1:5000),],type='l')



par(mfrow=c(6,2),mar=c(1,3,1,3))
for(kk in 1:p){
  plot(ps[[1]]$u[,kk,1],type='l',ylim=c(0,1))
  lines(ps[[2]]$u[,kk,1],col=2)
  lines(ps[[3]]$u[,kk,1],col=3)
  abline(h=calib.u[kk],col=2)
}

pairs(rbind(ps[[1]]$u[seq(nburn,jj,nthin),],ps[[2]]$u[seq(nburn,jj,nthin),],ps[[3]]$u[seq(nburn,jj,nthin),]),xlim=c(0,1),ylim=c(0,1),cex=.1,col=rep(1:3,each=length(seq(nburn,jj,nthin))))


plot(dat.calib[,1],dat.calib[,2])
lines(f(unstandardize(ps[[1]]$u[jj,,1]),temp.calib,strainRate.calib))
lines(f(unstandardize(ps[[2]]$u[jj,,1]),temp.calib,strainRate.calib),col=2)
lines(f(unstandardize(ps[[3]]$u[jj,,1]),temp.calib,strainRate.calib),col=3)

lines(f(unstandardize(c(ps[[2]]$u[jj,1:9,1],ps[[3]]$u[jj,10,1],ps[[2]]$u[jj,11:11,1])),temp.calib,strainRate.calib),col=4)

lines(f(unstandardize(rep(.5,11)),temp.calib,strainRate.calib),col=5)


get.se(ps[[1]]$u[jj,,1])/ps[[1]]$s2.hop[jj,1]
get.se(ps[[2]]$u[jj,,1])/ps[[2]]$s2.hop[jj,1]
get.se(ps[[3]]$u[jj,,1])/ps[[3]]$s2.hop[jj,1]

get.alpha(ps[[1]]$u[jj,,1],ps[[2]]$s2.hop[jj,1],ps[[2]]$se.hop[jj,1],1)

get.alpha(ps[[2]]$u[jj,,1],ps[[1]]$s2.hop[jj,1],ps[[1]]$se.hop[jj,1],1)


get.alpha(c(ps[[2]]$u[jj,1:8],ps[[1]]$u[jj,9],ps[[2]]$u[jj,10:11]),ps[[2]]$s2.hop[jj,],ps[[2]]$se.hop[jj,])
