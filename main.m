% -------------------------------------------------------------------------

warning('off','all');
display("--------------------------------------------------------------");
display(" Reinforcement Learning (RL) for Valve Control V.3.0 Apr-2021");
display("--------------------------------------------------------------");
display(" - This is a 'test' run demonstrating the software capabilities");
display(" - Initial load takes some time, please wait for the Episode Manager to show up");
display(" - The agent is then trained for 100 episodes");
display(" - This model is saved in the RESULTS folder");
display(" - Stability analysis is then demonstrated using a pre-trained agent");
display("--------------------------------------------------------------");
display("");

display("Begin training a RL agent for control of non-linear valve");
code_DDPG_Training;
display("Verify the RL agent works. Compare against PID control");
code_Experimental_Setup;
display("Stability Analysis: Estimate TF");
code_SA_TF_Estimator;
display("Stability Analysis: Plots");
code_SA_Utilities;
display("RL controller model saved and Stability Analysis plots saved in Results folder.");
display("----- End run -----"); 

