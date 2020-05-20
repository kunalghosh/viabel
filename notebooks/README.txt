This file contains information about the types of experiments performed.
All the scripts containing expts are contained in this folder.
The file stopping_rule_Workflow.py contains experiments on different models 
using the recently proposed workflow for optimisation. The file 
stopping_rule_expts.py contains code for running expts where the averaging
was being done since beginning and hence contains bad performance. 
Currently, there are two stopping ruls: the default used in Stan, which works
on absolute relative difference of successive elbo values in a window of certain
length,which is set by setting the parameter stopping_rule=1 in the code. The proposed
stopping rule is set by setting the parameter stopping_rule=2 in the code. The optimisers
are present in the file optimizers_workflow.py contained in one folder above.

MORE DETAILS TO FOLLOW ...
 

