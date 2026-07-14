%% Parameters for the Chaplygin Sleigh with servo actuated 3 link tail

% dq = [u_t; dtheta_1; theta_1; dtheta_2; theta_2; dtheta_h; theta_h]

rho =997;   %  density
AM = 1;

% For Head link 
Const.b = 0.075;                                    % length major axis (m)
Const.bs = 0.035;                                    % length minor axis (m)
m_h = 0.54;                                   % mass (kg)
I_h = m_h*(Const.b^2+Const.bs^2)/4;     % m.o.i (Kg/m^2) 
Const.ma_hx = m_h + AM*pi*rho*(Const.bs)^2*Const.b;          % added mass (kg)
Const.ma_hy = m_h + AM*pi*rho*(Const.b)^3;         % added mass (kg)
Const.Ia_hz = I_h + AM*(1/8)*pi*rho*Const.b*(Const.b^2-Const.bs^2)^2;% added m.o.i (Kg/m^2)

% For link1 
Const.l1 = 0.048;                      % length (m)
m_l1 = 0.01/2;                        % mass (kg)
I_l1 = (1/12)*m_l1*Const.l1^2;  % m.o.i (Kg/m^2)
Const.ma_l1x = m_l1 + AM*0;                           % added mass (kg)
Const.ma_l1y = m_l1 + AM*pi*rho*0.075*(Const.l1/2)^2;        % added mass (kg)
Const.Ia_l1z = I_l1 + (1/8)*pi*rho*0.075*(Const.l1/2)^4; % added m.o.i (Kg/m^2)

% For link2 
Const.l2 = 0.048;                       % length (m)
m_l2 = 0.01/2;                        % mass (kg)
I_l2 = (1/12)*m_l2*Const.l2^2;  % m.o.i (Kg/m^2) 
Const.ma_l2x = m_l2 + AM*0;                           % added mass (kg)
Const.ma_l2y = m_l2 + AM*pi*rho*0.075*(Const.l2/2)^2;      % added mass (kg)
Const.Ia_l2z = I_l2 + (1/8)*pi*rho*0.075*(Const.l2/2)^4; % added m.o.i (Kg/m^2)

% For short link 
Const.ls = 0.015;                    % length (m)
m_ls = 0.01;                        % mass (kg)
I_ls = (1/12)*m_ls*Const.ls^2;  % m.o.i (Kg/m^2) 
Const.ma_lsx = m_ls + AM*0;                           % added mass (kg)
Const.ma_lsy = m_ls + AM*pi*rho*0.075*(Const.ls/2)^2;       % added mass (kg)
Const.Ia_lsz = I_ls + AM*(1/8)*pi*rho*0.075*(Const.ls/2)^4; % added m.o.i (Kg/m^2)

% For rotor 
Const.c = 0.03;                             % length (m)
Const.m_r = 0.1;                            % mass (kg)
Const.I_r = Const.m_r*0.027^2;              % m.o.i (Kg/m^2) 

clearvars rho AM I_ls m_ls I_l2 m_l2 I_l1 m_l1 I_h m_h  
% Constant Rayleigh dissipation 
% Const.C_hx = 0.46;
% Const.C_hy = 10;
% Const.C_lx = 1;
% Const.C_ly = 10;
% Const.C_wl = 0.001;
% Const.C_wh = 0.017;
% Const.K_1 = 4;      % stiffness (Nm/rad)
% Const.K_2 = 7;           % stiffness (Nm/rad)
