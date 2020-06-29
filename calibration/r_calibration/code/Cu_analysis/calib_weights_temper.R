rm(list=ls())
gc()

library(BASS)
library(parallel)
library(mnormt)

source('~/git/immpala/code/strength_setup.R')
load('~/Desktop/immpala_data/calib_start.rda')




get.alpha<-function(uu,s2.vel,s2.hop,s2.tc,w.vel,w.hop,w.tc,se.vel.curr,se.hop.curr,se.tc.curr,itemperature){
  se.hop.cand<-get.se(uu,temps,edots)
  se.vel.cand<-get.se.vel(uu)
  se.tc.cand<-get.se.tc(uu)
  alpha<- -.5*(sum(se.hop.cand*w.hop/s2.hop) + sum(se.vel.cand*w.vel/s2.vel) + sum(se.tc.cand*w.tc/s2.tc) - sum(se.hop.curr*w.hop/s2.hop) - sum(se.vel.curr*w.vel/s2.vel) - sum(se.tc.curr*w.tc/s2.tc))*itemperature
  return(list(alpha=alpha,se.hop=se.hop.cand,se.vel=se.vel.cand,se.tc=se.tc.cand))
}






mcmc<-function(whop,wvel,wtc,itemp.ladder,start.temper){
  require(BASS)
  require(mnormt)
  
  nmcmc=30000
  nburn=20000
  #start.temper<-5000
  
  
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
  
  

  cand<-get.alpha(t(u[1,]),rep(1,n.vel),rep(1,n.hop),rep(1,n.tc),w.vel,w.hop,w.tc,rep(1,n.vel),rep(1,n.hop),rep(1,n.tc),1)
  se.vel[1,]<-cand$se.vel
  se.hop[1,]<-cand$se.hop
  se.tc[1,]<-cand$se.tc
  
  ntemp<-length(itemp.ladder)
  u.curr<-s2.vel.curr<-s2.hop.curr<-s2.tc.curr<-se.vel.curr<-se.hop.curr<-se.tc.curr<-list()
  for(tt in 1:ntemp){
    u.curr[[tt]]<-start.u
    s2.vel.curr[[tt]]<-rep(NA,n.vel)
    s2.hop.curr[[tt]]<-rep(NA,n.hop)
    s2.tc.curr[[tt]]<-rep(NA,n.tc)
    se.vel.curr[[tt]]<-cand$se.vel
    se.hop.curr[[tt]]<-cand$se.hop
    se.tc.curr[[tt]]<-cand$se.tc
  }
  
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
  
  
  count.swap<-0
  swap.vals<-matrix(nrow=nmcmc,ncol=2)
  
  lpost<-function(s2.vel,s2.hop,s2.tc,se.vel,se.hop,se.tc){
    sum((-1.5-a.vel-w.vel/2)*log(s2.vel)) +
      sum((-1.5-a.hop-w.hop/2)*log(s2.hop)) +
      sum((-1.5-a.tc-w.tc/2)*log(s2.tc)) +
      sum(-1/s2.vel*(.5*se.vel*w.vel/s2.vel + b.vel)) +
      sum(-1/s2.hop*(.5*se.hop*w.hop/s2.hop + b.hop)) +
      sum(-1/s2.tc*(.5*se.tc*w.tc/s2.tc + b.tc))
  }

  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  count=rep(0,ntemp)
  for(i in 2:nmcmc){
    
    for(tt in 1:ntemp){
    
    
      for(j in 1:max(ind.same.var.vel)){
        indj<-which(ind.same.var.vel==j)
        if(any(w.vel[indj]>0))
          s2.vel.curr[[tt]][indj]<-BASS:::rigammaTemper(1,1/2+a.vel[indj[1]]+sum(w.vel[indj])/2,b.vel[indj[1]]+.5*sum(se.vel.curr[[tt]][indj]*w.vel[indj]),itemp.ladder[tt])
        else
          s2.vel.curr[[tt]][indj]<-1
      }
      
      #browser()
      for(j in 1:max(ind.same.var.hop)){
        indj<-which(ind.same.var.hop==j)
        if(any(w.hop[indj]>0))
          s2.hop.curr[[tt]][indj]<-BASS:::rigammaTemper(1,1/2+a.hop[indj[1]]+sum(w.hop[indj])/2,b.hop[indj[1]]+.5*sum(se.hop.curr[[tt]][indj]*w.hop[indj]),itemp.ladder[tt])
        else
          s2.hop.curr[[tt]][indj]<-1
      }
      
      for(j in 1:max(ind.same.var.tc)){
        indj<-which(ind.same.var.tc==j)
        if(any(w.tc[indj]>0))
          s2.tc.curr[[tt]][indj]<-BASS:::rigammaTemper(1,1/2+a.tc[indj[1]]+sum(w.tc[indj])/2,b.tc[indj[1]]+.5*sum(se.tc.curr[[tt]][indj]*w.tc[indj]),itemp.ladder[tt])
        else
          s2.tc.curr[[tt]][indj]<-1
      }
      
      
      if(i>200 & i<nburn){
        mi<-max(1,i-300)
        S<-cov(u[mi:(i-1),])*cc+diag(eps*cc,p)
      }
      
      u.cand<-rmnorm(1,u[i-1,],S) # generate candidate
      if(constraints(model,u.cand)){ # constraint
        alpha<- -9999
      } else{
        
        cand<-get.alpha(t(u.cand),s2.vel.curr[[tt]],s2.hop.curr[[tt]],s2.tc.curr[[tt]],w.vel,w.hop,w.tc,se.vel.curr[[tt]],se.hop.curr[[tt]],se.tc.curr[[tt]],itemp.ladder[tt])
        alpha<-cand$alpha
      }
      if(log(runif(1)) < alpha){
        u.curr[[tt]]<-u.cand
        se.hop.curr[[tt]]<-cand$se.hop
        se.vel.curr[[tt]]<-cand$se.vel
        se.tc.curr[[tt]]<-cand$se.tc
        count[tt]<-count[tt]+1
      }
      
      se.vel.curr[[tt]]<-get.se.vel(t(u.curr[[tt]]))
      se.tc.curr[[tt]]<-get.se.tc(t(u.curr[[tt]]))
    
    }
    
    
    if(i>start.temper & ntemp>1){
      #browser()
      swap<-sample(1:ntemp,size=2) # candidate swap
      alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*(lpost(s2.vel.curr[[swap[1]]],s2.hop.curr[[swap[1]]],s2.tc.curr[[swap[1]]] ,se.vel.curr[[swap[1]]],se.hop.curr[[swap[1]]],se.tc.curr[[swap[1]]])-lpost(s2.vel.curr[[swap[2]]],s2.hop.curr[[swap[2]]],s2.tc.curr[[swap[2]]] ,se.vel.curr[[swap[2]]],se.hop.curr[[swap[2]]],se.tc.curr[[swap[2]]]))
      if(log(runif(1))<alpha){ # swap states
        temp<-u.curr[[swap[1]]]
        u.curr[[swap[1]]]<-u.curr[[swap[2]]]
        u.curr[[swap[2]]]<-temp
        
        temp<-se.vel.curr[[swap[1]]]
        se.vel.curr[[swap[1]]]<-se.vel.curr[[swap[2]]]
        se.vel.curr[[swap[2]]]<-temp
        
        temp<-se.hop.curr[[swap[1]]]
        se.hop.curr[[swap[1]]]<-se.hop.curr[[swap[2]]]
        se.hop.curr[[swap[2]]]<-temp
        
        temp<-se.tc.curr[[swap[1]]]
        se.tc.curr[[swap[1]]]<-se.tc.curr[[swap[2]]]
        se.tc.curr[[swap[2]]]<-temp

        temp<-s2.vel.curr[[swap[1]]]
        s2.vel.curr[[swap[1]]]<-s2.vel.curr[[swap[2]]]
        s2.vel.curr[[swap[2]]]<-temp
        
        temp<-s2.hop.curr[[swap[1]]]
        s2.hop.curr[[swap[1]]]<-s2.hop.curr[[swap[2]]]
        s2.hop.curr[[swap[2]]]<-temp
        
        temp<-s2.tc.curr[[swap[1]]]
        s2.tc.curr[[swap[1]]]<-s2.tc.curr[[swap[2]]]
        s2.tc.curr[[swap[2]]]<-temp
        
        count.swap<-count.swap+1
        swap.vals[i,]<-sort(swap)
      }
    }
    
    
    
    
    s2.vel[i,]<-s2.vel.curr[[1]]
    s2.hop[i,]<-s2.hop.curr[[1]]
    s2.tc[i,]<-s2.tc.curr[[1]]
    u[i,]<-u.curr[[1]]
    
    
    if(i%%100==0)
      cat('it:',i,'acc:',count,'swap:',count.swap,timestamp(quiet=T),'\n')
    
  }
  
  return(list(u=u,count=count,s2.vel=s2.vel,s2.hop=s2.hop,s2.tc=s2.tc,se.vel=se.vel,se.hop=se.hop,se.tc=se.tc,count.swap=count.swap,swap.vals=swap.vals))
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

#mcmc(whop=0,wvel=1,wtc=0,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000)


ps<-mclapply(1:8,function(i) mcmc(whop=1,wvel=0,wtc=0,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_hop_temper.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=1,wtc=0,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_vel_temper.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=0,wtc=1,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_tc_temper.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/2,wvel=1/2,wtc=0,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_hopvel_temper.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/3,wvel=1/3,wtc=1/3,itemp.ladder = c(1,.9,.8,.6,.5),start.temper=20000),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_equalWeight_temper.rda')
rm(ps)
gc()