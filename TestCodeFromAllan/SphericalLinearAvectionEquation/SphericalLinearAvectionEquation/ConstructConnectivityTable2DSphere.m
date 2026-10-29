function [c,gidx,X3D,Y3D,Z3D] = ConstructConnectivityTable2DSphere(VX,VY,VZ,EToV,N,x,y,z,varargin);
%
%function [c,gidx,X3D,Y3D,Z3D] = ConstructConnectivityTable3DHexahedral(VX,VY,VZ,EToV,P,Ps,x,y,z,varargin)
%
% Construct local-to-global connectivity tables (Algorithm 4.1).
%
% HOW TO ORDER AND ADD NODES
%
% SHOULD MATCH the Localreorder mapping for the element nodes.
%
% Ordering used is: [Vertice interioredgenodes verticalEdges Facenodes Interiornodes]
%
% THIS IS A BRUTE FORCE APPROACH
%
% By Allan P. Engsig-Karup, apek@dtu.dk.
%
Nk = size(EToV,1);
Np = size(x,1);
c  = zeros(Nk,Np); % allocate connectivity table
Nfaces = 3; % triangle
[EToE,EToF]= tiConnect2D(EToV,Nfaces);%  tiConnect3DgeneralHexahedral(EToV);
%% global vertices defining elements 
c(:,1:3) = EToV;
gidx     = length(VX);   % no assumption on vertices and numbering
exitflag = 0;
X3D = zeros(1,length(Nk*Np));
Y3D = zeros(1,length(Nk*Np));
Z3D = zeros(1,length(Nk*Np));
X3D(1:gidx) = VX(unique(EToV));
Y3D(1:gidx) = VY(unique(EToV));
Z3D(1:gidx) = VZ(unique(EToV));
%gidx = Np;
NODETOL = 1e-8;
lidx = 4:Np;
c(1,lidx) = gidx + (1:length(lidx)); % all nodes in first element must be unique
k = 1;
tmpx = x(lidx,k); X3D(gidx+(1:length(lidx))) = tmpx(:);
tmpy = y(lidx,k); Y3D(gidx+(1:length(lidx))) = tmpy(:);
tmpz = z(lidx,k); Z3D(gidx+(1:length(lidx))) = tmpz(:);
gidx = gidx + length(lidx);
for k = 2 : Nk
    for i = 4 : Np
%         close all
%         figure
%         plot3(x(:,k),y(:,k),z(:,k),'r.')
%         hold on
        for m = setdiff(EToE(k,:),k:Nk) % 1 : (k-1) % go through all connected and previous elements
%            setdiff(EToE(k,:),k:Nk)
%            plot3(x(:,m),y(:,m),z(:,m),'k.')
            % compute distance between known nodes to find ones that are
            % identical
            d = (x(i,k)-x(:,m)).^2+(y(i,k)-y(:,m)).^2+(z(i,k)-z(:,m)).^2;
            % find the node that are the same at the current one with 
            % index i
            lidx = find(abs(d) < NODETOL);
            if isempty(lidx) == 0 % not empty
                lidx = lidx(1); % only one node can be unique
                c(k,i) = c(m,lidx); % same global index as one node that exist already
%                 idd = lidx; %c(m,lidx) %c(k,i)
%                 plot3(x(idd,m),y(idd,m),z(idd,m),'bo')
%                 idd = i;% c(k,i);
%                 plot3(x(idd,k),y(idd,k),z(idd,k),'g*')
%                 drawnow
%                 pause
                exitflag = 1;
                break;
            end
        end
        if exitflag == 0 % add new node
            lidx = i;
            tmpx = x(lidx,k); X3D(gidx+1) = tmpx(:);
            tmpy = y(lidx,k); Y3D(gidx+1) = tmpy(:);
            tmpz = z(lidx,k); Z3D(gidx+1) = tmpz(:);
            gidx = gidx + 1;
            c(k,i) = gidx;
        else
            exitflag = 0; % we updated the index of the current node
        end
    end
end
return
