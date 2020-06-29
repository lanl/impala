rm(list=ls())
gc()

library(BASS)
library(parallel)
library(mnormt)

source('~/git/immpala/code/strength_setup.R')
load('~/Desktop/immpala_data/calib_start.rda')

get.alpha<-function(uu,s2.vel,s2.hop,s2.tc,w.vel,w.hop,w.tc,se.vel.curr,se.hop.curr,se.tc.curr){
  se.hop.cand<-se.vel.cand<-se.tc.cand<-0
  if(sum(w.hop)>0)
    se.hop.cand<-get.se(uu,temps,edots)
  if(sum(w.vel)>0)
    se.vel.cand<-get.se.vel(uu)
  if(sum(w.tc)>0)
    se.tc.cand<-get.se.tc(uu)
  alpha<- -.5*(sum(se.hop.cand*w.hop/s2.hop) + sum(se.vel.cand*w.vel/s2.vel) + sum(se.tc.cand*w.tc/s2.tc) - sum(se.hop.curr*w.hop/s2.hop) - sum(se.vel.curr*w.vel/s2.vel) - sum(se.tc.curr*w.tc/s2.tc))
  return(list(alpha=alpha,se.hop=se.hop.cand,se.vel=se.vel.cand,se.tc=se.tc.cand))
}

mcmc<-function(whop,wvel,wtc){
  require(BASS)
  require(mnormt)
  
  nmcmc=10000
  nburn=9000
  
  
  
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
  
  a.vel<-rep(c(2.5,6),n.vel/2)
  a.hop<-rep(8,n.hop)
  a.tc<-c(2,2.5,2)
  b.vel<-rep(c(.00025,2000),n.vel/2)
  b.hop<-rep(1e-6,n.hop)
  b.tc<-c(.1,2,100)
  
  
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
  
  count=rep(0,p)
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
    

    for(jj in 1:p){
      u.cand<-u[i,]
      u.cand[jj]<-runif(1)
    
      
      cand<-get.alpha(t(u.cand),s2.vel[i,],s2.hop[i,],s2.tc[i,],w.vel,w.hop,w.tc,se.vel[i-1,],se.hop[i-1,],se.tc[i-1,])
      alpha<-cand$alpha
    
      if(log(runif(1)) < alpha){
        u[i,]<-u.cand
        se.hop[i,]<-cand$se.hop
        se.vel[i,]<-cand$se.vel
        se.tc[i,]<-cand$se.tc
        count[jj]<-count[jj]+1
      }
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

model='JC'
x.rr<-matrix(c(.0001,.01,
               .0001,.01,
               .0002,.3,
               .0001,1.5,
               .002,3,
               .03,.0315, # flyer plate vel
               .03,.0315, # tc vel
               .3,.6 # sm_0
),ncol=2,byrow=T)


ps<-mclapply(1:8,function(i) mcmc(whop=1,wvel=0,wtc=0),mc.preschedule = F,mc.cores = 8) 
save(ps,file='~/Desktop/immpala_data/calib_hop_indepSampler.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=1,wtc=0),mc.preschedule = F,mc.cores = 8) # started 8:31
save(ps,file='~/Desktop/immpala_data/calib_vel_indepSampler.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=0,wtc=1),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_tc_indepSampler.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/2,wvel=1/2,wtc=0),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_hopvel_indepSampler.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/3,wvel=1/3,wtc=1/3),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_equalWeight_indepSampler.rda')
rm(ps)
gc()