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
save(ps,file='~/Desktop/immpala_data/calib_hop_flatprior.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=1,wtc=0),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_vel_flatprior.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=0,wvel=0,wtc=1),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_tc_flatprior.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/2,wvel=1/2,wtc=0),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_hopvel_flatprior.rda')
rm(ps)
gc()

ps<-mclapply(1:8,function(i) mcmc(whop=1/3,wvel=1/3,wtc=1/3),mc.preschedule = F,mc.cores = 8)
save(ps,file='~/Desktop/immpala_data/calib_equalWeight_flatprior.rda')
rm(ps)
gc()