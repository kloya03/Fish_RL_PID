function [M, C, G] = extract_dynamics(B, Qd, Qdd)
% Extracts Mass matrix (M), Coriolis matrix (C), and Gravity vector (G)
% from symbolic dynamics equations of the form: M*Qdd + C*Qd + G = 0

nq = length(Qdd);
M = sym(zeros(nq,nq));
C = sym(zeros(nq,nq));

for i = 1:nq
    for j = 1:nq
         % Extract M(i,j): Coefficient of Qdd(j)
        [cf,tf] = coeffs(B(i),Qdd(j));
        cf =formula(cf); tf = formula(tf);
        if isempty(cf(tf==Qdd(j)))
            M(i,j)=0;
        else
            M(i,j) = cf(tf==Qdd(j));
        end
        B(i) = simplify(expand(B(i)-M(i,j)*Qdd(j)));
        
        % Extract C(i,j): Coefficient of Qd(j) from remaining terms
        [cfc,tfc] = coeffs(B(i),Qd(j));
        cfc =formula(cfc); tfc = formula(tfc);
        if isempty(tfc)
            C(i,j) = 0;
        else
            num = size(tfc);
            for k = 1:num(1,2)  
                if tfc(1,k)~=1
                    C(i,j) = simplify(expand(C(i,j) + cfc(1,k)*tfc(1,k)/Qd(j)));
                elseif tfc(1,k)==1
                    C(i,j) = C(i,j) + 0;
                end
            end
        end
        B(i) = simplify(B(i)-C(i,j)*Qd(j));
    end
end

% Remaining terms are treated as gravity or external forces
G=B;