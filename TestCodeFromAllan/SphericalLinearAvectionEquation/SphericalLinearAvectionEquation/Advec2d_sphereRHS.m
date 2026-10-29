function [rhsu] = Advec2d_sphereRHS(u, c_phi, c_theta)
Globals2D;
df = zeros(Nfp*Nfaces, K);

%Upwind flux
cn = c_phi* nx(:) + c_theta * ny(:);
kappa = 0.5 * (1 + cn ./ abs(cn));
ustar = kappa .* u(vmapM) + (1 - kappa) .* u(vmapP);
df(:) = cn .* (u(vmapM) - ustar);

%Gradient
[ux, uy] = Grad2D(u);
% rhsu = -((c_phi* ux) + c_theta * uy) + LIFT * (Fscale .* df);
rhsu = -((c_phi* ux)./sin(y) + c_theta * uy) + LIFT * (Fscale .* df);
end

