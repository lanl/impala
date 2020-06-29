library(reticulate)
setwd('~/git/immpala/code/')
smdl <- import_from_path("strength_models_add_ptw")

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


material_model = smdl$MaterialModel(
  parameters          = material_parameters,
  flow_stress_model   = smdl$johnson_cook)


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


# x.rr<-matrix(c(.0001,.01,
#                .0001,.01,
#                .0002,.3,
#                .0001,1.5,
#                .002,3,
#                .03,.0315, # flyer plate vel
#                .03,.0315, # tc vel
#                .3,.6 # sm_0
# ),ncol=2,byrow=T)
# u<-runif(8)
# unstandardize2<-function(u){
#   u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
# }
# 
# emax   = 0.65
# Nhist  = 100
# plot(f(unstandardize(u)[1:5],473,2000))
# 
# eps<-seq(0,.65,length.out=100)
# plot(jc(unstandardize(u)[1:5],473,2000,eps))
# 
# chi<-.9
# Cv0<-0.900e-5
# Cv<-Cv0
# rho0<-2.683 
# rho<-rho0
# TTseq<-TT
# for(i in 2:length(eps)){
#   TTseq[i]<-TTseq[i]+chi*stress*edot*dt/(Cv*rho)
# }
# jc<-function(pp,edot,TT,eps,edot0=1.0e-6,Tref=298,Tmelt0=933){
#   Tstar<-max((TT-Tref)/(Tmelt0-Tref),0)
#   browser()
#   (pp[1]+pp[2]*eps^pp[4])*(1+pp[3]*log(edot/edot0))*(1-Tstar^pp[5])
# }
# Th    = nmp.max( [(T-mp.Tref) / (Tmelt-mp.Tref), 0.0])
# Y = (mp.A+mp.B*eps**mp.n)*(1.0+mp.C*nmp.log(edot/mp.edot0) )*(1.0-Th**mp.m)
# 
# chi = self.parameters.chi
# Cv  = self.specific_heat(self)
# rho = self.density(self)
# 
# self.state.T      += chi*self.state.stress*edot*dt / (Cv*rho)