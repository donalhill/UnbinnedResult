def RunMinuit(sess, nll, feed_dict = None, call_limit = 50000, useGradient = True,
              gradient = None, printout = 50, tmpFile = "tmp_result.txt",
              runHesse = False, runMinos = False,
              options = None, run_metadata = None ) :
  """
    Perform MINUIT minimisation of the negative likelihood.
      sess         : TF session
      nll          : graph for negitive likelihood to be minimised
      feed_dict    : Dictionary of feeds for placeholders (or None if data is already loaded by LoadData)
      call_limit   : limit on number of calls for MINUIT
      gradient     : external gradient graph. If None and useGradient is not False, will be
                     calculated internally
      useGradient  : flag to control the use of analytic gradient while fitting:
                     None or False   : gradient is not used
                     True or "CHECK" : analytic gradient will be checked with finite elements,
                                       and will be used if they match
                     "FORCE"         : analytic gradient will be used regardless.
      printout     : Printout frequency
      tmpFile      : Name of the file with temporary results (updated every time printout is called)
      runHesse     ; Run HESSE after minimisation
      runMinos     : Run MINOS after minimisation
      options      : additional options to pass to TF session run
      run_metadata : metadata to pass to TF session run
  """

  global cacheable_tensors

  tfpars = tf.trainable_variables()                      # Create TF variables
  float_tfpars = [ p for p in tfpars if p.floating() ]   # List of floating parameters

  if useGradient and gradient is None :
    gradient = tf.gradients(nll, float_tfpars)            # Get analytic gradient

  cached_data = {}

  fetch_list = []
  for i in cacheable_tensors :
    if i not in cached_data : fetch_list += [ i ]
  if feed_dict :
    feeds = dict(feed_dict)
  else :
    feeds = None
  for i in cacheable_tensors :
    if i in cached_data : feeds[i] = cached_data[i]

  fetch_data = sess.run(fetch_list, feed_dict = feeds ) # Calculate tensors to be cached

  for i,d in zip(fetch_list, fetch_data) :
    cached_data[i] = d

  if feed_dict :
    feeds = dict(feed_dict)
  else :
    feeds = None
  for i in cacheable_tensors :
    if i in cached_data : feeds[i] = cached_data[i]

  def fcn(npar, gin, f, par, istatus) :                  # MINUIT fit function
    for i in range(len(float_tfpars)) : float_tfpars[i].update(sess, par[i])

    f[0] = sess.run(nll, feed_dict = feeds, options = options, run_metadata = run_metadata ) # Calculate log likelihood

    if istatus == 2 :            # If gradient calculation is needed
      dnll = sess.run(gradient, feed_dict = feeds, options = options, run_metadata = run_metadata )  # Calculate analytic gradient
      for i in range(len(float_tfpars)) : gin[i] = dnll[i] # Pass gradient to MINUIT
    fcn.n += 1
    if fcn.n % printout == 0 :
      print "  Iteration ", fcn.n, ", Flag=", istatus, " NLL=", f[0], ", pars=", sess.run(float_tfpars)
      tmp_results = { 'loglh' : f[0], "status" : -1 }
      for n,p in enumerate(float_tfpars) :
        tmp_results[p.par_name] = (p.prev_value, 0.)
      WriteFitResults(tmp_results, tmpFile)

  fcn.n = 0
  minuit = TVirtualFitter.Fitter(0, len(tfpars))        # Create MINUIT instance
  minuit.Clear()
  minuit.SetFCN(fcn)
  arglist = array.array('d', 10*[0])    # Auxiliary array for MINUIT parameters

  for n,p in enumerate(float_tfpars) :  # Declare fit parameters in MINUIT

#    print "passing parameter %s to Minuit" % p.par_name
    step_size = p.step_size
    lower_limit = p.lower_limit
    upper_limit = p.upper_limit
    if not step_size : step_size = 1e-6
    if not lower_limit : lower_limit = 0.
    if not upper_limit : upper_limit = 0.
    minuit.SetParameter(n, p.par_name, p.init_value, step_size, lower_limit, upper_limit)

  arglist[0] = 0.5
  minuit.ExecuteCommand("SET ERR", arglist, 1)  # Set error definition for neg. likelihood fit
  if useGradient == True or useGradient == "CHECK" :
    minuit.ExecuteCommand("SET GRA", arglist, 0)  # Ask analytic gradient
  elif useGradient == "FORCE" :
    arglist[0] = 1
    minuit.ExecuteCommand("SET GRA", arglist, 1)  # Ask analytic gradient
  arglist[0] = 1
  minuit.ExecuteCommand("SET STR", arglist, 1)
  arglist[0] = call_limit                       # Set call limit
  arglist[1] = 0.001
  minuit.ExecuteCommand("MIGRAD", arglist, 1)   # Perform minimisation
#  minuit.ExecuteCommand("SHO EIG", arglist, 1)
#  minuit.ExecuteCommand("IMP", arglist, 1)   # Perform minimisation
#  minuit.ExecuteCommand("SIMPLEX", arglist, 1)   # Perform minimisation

  minuit.ExecuteCommand("SET NOG", arglist, 0)  # Ask no analytic gradient

  if runHesse :
    minuit.ExecuteCommand("HESSE", arglist, 1)

  if runMinos :
    minuit.ExecuteCommand("MINOS", arglist, 1)

  results = {}                                  # Get fit results and update parameters

  for n,p in enumerate(float_tfpars) :
    p.update(sess, minuit.GetParameter(n) )
    p.fitted_value = minuit.GetParameter(n)
    p.error = minuit.GetParError(n)
    covmat=np.zeros((len(float_tfpars),len(float_tfpars)))
    for i in range(len(float_tfpars)):
      for j in range(len(float_tfpars)):
        covmat[i,j]=minuit.GetCovarianceMatrixElement(i,j)

####
    if runMinos :
      eplus  = array.array("d", [0.])
      eminus = array.array("d", [0.])
      eparab = array.array("d", [0.])
      globcc = array.array("d", [0.])
      minuit.GetErrors(n, eplus, eminus, eparab, globcc)
      p.positive_error = eplus[0]
      p.negative_error = eminus[0]
      results[p.par_name] = ( p.fitted_value, p.error, p.positive_error, p.negative_error )
    else :
      results[p.par_name] = ( p.fitted_value, p.error)

  # Get status of minimisation and NLL at the minimum
  maxlh = array.array("d", [0.])
  edm = array.array("d", [0.])
  errdef = array.array("d", [0.])
  nvpar = array.array("i", [0])
  nparx = array.array("i", [0])
  fitstatus = minuit.GetStats(maxlh, edm, errdef, nvpar, nparx)

  # return fit results
  results["loglh"] = maxlh[0]
  results["status"] = fitstatus
  results["iterations"] = fcn.n
  return results,covmat
