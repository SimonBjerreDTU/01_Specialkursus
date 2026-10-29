function [Kk] = directional_advection_matrix2Dsimple(k, Np, J, MassMatrix, rX, sX, Dr, Ds)
DXk = diag(rX(:,k))*Dr + diag(sX(:,k))*Ds;
Kk = diag(J(:,k))*MassMatrix*DXk;
return