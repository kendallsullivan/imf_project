specfit is a routine contained in model_fit_tools_vx.py, where the version numbers currently correspond to:
v1) standard emcee implementation of the fitting routine, as well as a Metropolis-Hastings MCMC function -- This has been updated LEAST recently, and thus currently isn't public.
v2) parallel tempering emcee implementation, as well as updated versions of the functions contained in v1. This is the core code for anything one might want to do with this package.
v3) uses a simulated annealing algorithm (which is essentially a modified Metropolis-Hastings) for the MCMC, and is dependent on v2 being in the same directory, as it imports its basic functions from v2. 

Any/all of these should be fairly useable. The file paths are currently hardcoded, so they would need to be modified. There are lots of additional functionalities to come, once I've picked an MCMC method to use. Leaning toward either a standard MH MCMC or All of these depend on phoenix models being in the same directory in a subdirectory called 'phoenix/*'. 

test_models.py shows a standard implementation of most of the tools in model_fit_tools_v2.py, except for currently showing a useable fitting routine (that will be rectified soon!). Other utilities shown in test_models include: -reduce resolution by smoothing with a Gaussian kernel -select a fitting region -add spectra and scale them -plot nice looking results panels.

The preprocessing scripts set up the requirements and systems for running the emcee implementations of the code. For most other requirements I would suggesting writing your own initialization code and simply using the parts of the code that are useful to you. 