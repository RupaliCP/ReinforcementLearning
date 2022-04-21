%--------------------------------------------------------------------------

clear all;

%% Set paths
MODELS_PATH = 'results/';
VALVE_SIMULATION_MODEL = 'sm_Experimental_Setup'; % Simulink experimentation circuit
RL_AGENT = strcat(VALVE_SIMULATION_MODEL, '/RL Sub-System/RL Agent');

%% GRADED LEARNING models
PRE_TRAINED_MODEL_FILE = 'Grade_I.mat';
%PRE_TRAINED_MODEL_FILE = 'Grade_II.mat';
%PRE_TRAINED_MODEL_FILE = 'Grade_III.mat';
%PRE_TRAINED_MODEL_FILE = 'Grade_IV.mat';

% Physical system parameters. Use iteratively. Suceessively increase
%  difficulty of training task and apply Graded Learning to train the agent
TIME_DELAY = 2.5/2;   % Time delay for process controlled by valve
fS = 8.4000/2;        % Valve dynamic friction
fD = 3.5243/2;        % Valve static friction

% Agent stage to be tested
RL_MODEL_FILE = strcat(MODELS_PATH, PRE_TRAINED_MODEL_FILE);

% Time step. Tf/Ts gives Simulink's simulation time
Ts = 1.0;   % Ts: Sample time (secs)
Tf = 200;   % Tf: Simulation length (secs)
ACCEPTABLE_DELTA = 0.05;

% Load experiences from pre-trained agent    
sprintf('- Load model: %s', PRE_TRAINED_MODEL_FILE)
load(RL_MODEL_FILE,'agent');

% ----------------------------------------------------------------
% Validate the learned agent against the model by simulation
% ----------------------------------------------------------------
% Define observation and action space
NUMBER_OBSERVATIONS = 3;

% Observation Vector 
%  (1) U(k)
%  (2) Error signal
%  (3) Error integral

obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -inf 0]',...
    'UpperLimit',[ inf  inf inf]');
obsInfo.Name = 'observations';
obsInfo.Description = 'controlled flow, error, integral of error';
numObservations = obsInfo.Dimension(1);

% obsInfo = rlNumericSpec([NUMBER_OBSERVATIONS 1],...
%     'LowerLimit',[0    -inf -inf]',...             % Actual-flow is limited to 0 on the lower-side
%     'UpperLimit',[inf   inf  inf]');
% obsInfo.Name = 'observations';
% obsInfo.Description = '[actual-signal, error, integrated error]';
% numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1]);
actInfo.Name = 'flow';
numActions = numel(actInfo);

% Intialise the environment with the serialised agent and run the test
sprintf ('\n\n ==== RL for control of valves V.5.1 ====================')
sprintf (' ---- Testing model: %s', PRE_TRAINED_MODEL_FILE)
sprintf (' ---- Parameters: Time-Delay: %3.2f, fS: %3.2f, fD: %3.2f', TIME_DELAY, fS, fD)
        
env = rlSimulinkEnv(VALVE_SIMULATION_MODEL, RL_AGENT, obsInfo, actInfo);
simOpts = rlSimulationOptions('MaxSteps', 2000);
experiences = sim(env, agent, simOpts);
    
% ------------------------------------------------------------------------
% Environment Reset function 
% Randomize Reference_Signal between 0 and 100
% Reset if the controlled speed drops below zero or exceeds 100 
% ------------------------------------------------------------------------
function in = localResetFcn(in, RL_System)
    block_Reference_Signal = strcat (RL_System, '/Reference_Signal');
    Reference_Signal = 20+randi(80) + rand;
    in = setBlockParameter(in, block_Reference_Signal, ...
        'Value', num2str(Reference_Signal));

    % Randomize initial condition of the flow (0 and 100) 
    block_Actual_Flow = strcat (RL_System, '/Plant/Process/FLOW');    
    Actual_Flow = 20 + randi(80) + rand;
    in = setBlockParameter(in, block_Actual_Flow, 'Bias', num2str(Actual_Flow));
end