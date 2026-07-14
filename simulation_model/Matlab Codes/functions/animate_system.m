
function animate_system(rdN, Const, tht, rot_tht,step,video)
    
    if nargin <5
        step =10;
    end
    a = 0.03;
    figure; clf
    set(gcf,'Renderer','opengl')
    
    axis equal
    grid on
    xlabel('x'); ylabel('y')
    hold on
    
    t = linspace(0,2*pi,120);
    % -------- Precreate graphics objects --------
    % Links + joints
    hLinks  = gobjects(3,1);
    hJoints = gobjects(3,1);
    for jj = 1:3
        hLinks(jj)  = plot(nan,nan,'r-','LineWidth',3);
        hJoints(jj) = scatter(nan,nan,20,'k','filled');
    end
    
    % Base
    hBase = scatter(nan,nan,25,'b','filled');
    % COM trajectory
    hTraj = plot(nan,nan,'k-','LineWidth',1);
    % -------- BODY (transparent red) --------
    hBody = patch(nan,nan,'r', ...
        'FaceAlpha',0.25, ...      % faint transparency
        'EdgeColor','r', ...
        'LineWidth',2);
    % -------- ROTOR (solid black filled disk) --------
    hRotor = patch(nan,nan,'k', ...
        'EdgeColor','k', ...
        'FaceAlpha',1);
    % Rotor arm
    hRotorArm = plot(nan,nan,'g-','LineWidth',2);
    
    % -------- Animation loop --------
    for ii = 1:step:length(rdN(1,1,:))
        cen = [rdN(1,5,ii), rdN(2,5,ii)];
        phi = tht(ii,4);
        cen_rotor = [cen(1) + Const.c*cos(phi), ...
                     cen(2) + Const.c*sin(phi)];
        rotor_end = [cen_rotor(1) + a*cos(rot_tht(ii)), ...
                     cen_rotor(2) + a*sin(rot_tht(ii))];
    
        % ---- Update BODY (ellipse) ----
        ex = Const.b  * cos(t);
        ey = Const.bs * sin(t);
        R  = [cos(phi) -sin(phi); 
              sin(phi)  cos(phi)];
        xy = R * [ex; ey];
        set(hBody, ...
            'XData', cen(1) + xy(1,:), ...
            'YData', cen(2) + xy(2,:));
        % ---- Update links + joints ----
        for jj = 1:3
            set(hLinks(jj), ...
                'XData', rdN(1,jj:jj+1,ii), ...
                'YData', rdN(2,jj:jj+1,ii));
    
            set(hJoints(jj), ...
                'XData', rdN(1,jj+1,ii), ...
                'YData', rdN(2,jj+1,ii));
        end
        set(hBase, ...
            'XData', rdN(1,1,ii), ...
            'YData', rdN(2,1,ii));
        % ---- Update trajectory ----
        set(hTraj, ...
            'XData', squeeze(rdN(1,5,1:ii)), ...
            'YData', squeeze(rdN(2,5,1:ii)));
        % ---- Update rotor disk (black filled circle) ----
        set(hRotor, ...
            'XData', cen_rotor(1) + a*cos(t), ...
            'YData', cen_rotor(2) + a*sin(t));
        % ---- Update rotor arm ----
        set(hRotorArm, ...
            'XData', [cen_rotor(1), rotor_end(1)], ...
            'YData', [cen_rotor(2), rotor_end(2)]);
    
        axis([cen(1)-1 cen(1)+1 cen(2)-1 cen(2)+1])
        % axis([-1 1 -1 1])
        drawnow limitrate

        if nargin > 5
            if video ==1
                filan = "Act_balancedCS_wtail1";
                filangif = filan+".gif";
                exportgraphics(ff,filangif,"Append",true)
                % sgtitle(sprintf(['Inputs: $A = %.2f$, $\\Omega = %.2f$, Parameters: $C_hx = %.2f$,' ...
                % '$C_hy = %.2f$, $C_lsx = %.2f$, $C_lsy = %.2f$, $C_l2x = %.2f$, $C_l2y = %.2f$,'...
                % '$C_l1x = %.2f$, $C_l1y = %.2f$, $K_1 = %.2f$, $K_2 = %.2f$, '],...
                % A, omega, Const.C_hx, Const.C_hy, Const.C_lsx, Const.C_lsy, Const.C_l2x, Const.C_l2y, ...
                % Const.C_l1x, Const.C_l1y, Const.K_1, Const.K_2), 'Interpreter', 'latex');
            end
        end
        pause(0.1)
    end
end
