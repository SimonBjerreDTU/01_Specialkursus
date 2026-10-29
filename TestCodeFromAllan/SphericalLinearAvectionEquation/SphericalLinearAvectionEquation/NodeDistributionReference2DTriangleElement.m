function [r,s] = NodeDistributionReference2DTriangleElement(N,r,s,Np)
% Purpose: Reorder nodes to enable easy lookups later and match 2D and 3D
%          configurations.
%
% By Allan P. Engsig-Karup.

Nfp = N+1;
tol = 0.2/N^2;
% Faces
fid1 = find( abs(s+1) < tol)'; % south
fid2 = find( abs(r+s) < tol)'; % northeast
fid3 = find( abs(r+1) < tol)'; % west
% Interior
fint = setdiff(1:Np,[fid1 fid2 fid3]);
% Reorder local nodes to match the chosen ordering in terms of vertices,
% edges, interior nodes.
% LocalReorder odering must match with chosen ordering scheme: vertices,
% edges, interior nodes...
LocalReorder = [find(r==-1 & s==-1) find(r==1 & s==-1)...
    find(r==-1 & s==1) fid1(2:Nfp-1) ...
    fid2(2:Nfp-1) fid3(Nfp-1:-1:2) fint];
%return
r = r(LocalReorder);
s = s(LocalReorder);
return
