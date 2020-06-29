library(reticulate)
setwd('~/git/immpala/calibration/py_calibration/')
smdl <- import_from_path("strength_models_add_ptw")
smdl <- import_from_path("physical_models")

smdl$ModelParameters$update_parameters()
##### PTW


# ----------------- Step 5: define another material model -------------------------
#    here we set up a PTW model
#xnames <- c('theta0', 'p', 's0','sinf',       'y0',      'yinf','y1', 'y2','kappa', 'gamma', 'vel')
material_parameters=smdl$ModelParameters$update_parameters(smdl$)
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

material_model = smdl$MaterialModel()
material_model$update_parameters(material_model,material_parameters)
smdl$ModelParameters$update_parameters()



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
  x<-as.list(x[1:10])
  #x<-as.list(x[1:8])
  material_model$parameters$update_parameters(material_model$parameters,x)
  res<-smdl$compute_state_history(material_model,strain_history)
  return(res[,2:3])
}
