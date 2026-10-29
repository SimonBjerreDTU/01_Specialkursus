% Nodal DG-FEM for the spherical linear advection equation
%
% By Allan Peter Engsig-Karup, apek@dtu.dk
clear; clc;

% Driver script
Globals2D;

% Polynomial order
N = 4;

generation = 2;     
type = 'ico'; 

[ P, tri ] = generateSphereMesh( generation, type );


%Cartesian coordinates of the sphere mesh vertices
x_coor=P(1,:)';
y_coor=P(2,:)';
z_coor=P(3,:)';


%Spherical coodinates
r1=sqrt(x_coor.^2+y_coor.^2+z_coor.^2);
theta=acos(z_coor./r1);
phi=atan2(y_coor,x_coor);


VX=phi';
VY=theta';
EToV=tri;
K=size(EToV,1);
Nv=length(VX);

StartUp2D;

% % Build periodic maps
% phiperiod = 2*pi;
% thetaperiod = pi;
% BuildPeriodicMaps2D(phiperiod, thetaperiod);


%Parameters
phi0 = pi; theta0 = pi/2; sigma = 0.3;
c_phi = 0.5; c_theta = 0.2;
FinalTime = 0;

%Initial condition
u = exp(-((phi- phi0).^2 + (theta- theta0).^2) / sigma^2);

%Solve
[u, time] = Advec2d_Spherical(u, FinalTime, c_phi, c_theta);



Ltheta = pi; %Length of the domain theta
Lphi=2*pi;    %Length of the domain phi

%Exact soltuion
phi_shift = mod(phi - c_phi * FinalTime - phi0 + Lphi/2, Lphi) - Lphi/2;
theta_shift = mod(theta - c_theta * FinalTime - theta0 + Ltheta/2, Ltheta) - Ltheta/2;
uexact = exp(-(phi_shift.^2 + theta_shift.^2) / sigma^2);
u=u(:);

%Numerical solution
figure;
trisurf(tri, P(1,:), P(2,:), P(3,:), u, 'EdgeColor', 'none');
title('Numerical');
axis equal tight;
view(3); 
shading interp; 
colormap jet;
colorbar;


%Exact solution
figure;
trisurf(tri, P(1,:), P(2,:), P(3,:), uexact, 'EdgeColor', 'none');
title('Exact');
axis equal tight;
view(3); 
shading interp; 
colormap jet;
colorbar;


% Plot mesh
figure;
trisurf(tri, P(1,:), P(2,:), P(3,:), 'FaceColor', 'none', 'EdgeColor', 'k');
hold on;
scatter3(P(1,:), P(2,:), P(3,:), 20, 'r', 'filled');  % Plot vertices
view(3); 
axis equal tight;
title('Mesh');


%Internal nodes populated
figure;
X_nodes=NodesOnSphere(P, tri, N);
trisurf(tri, P(1,:), P(2,:), P(3,:), 'FaceColor', '[0.8,0.8,0.8]', 'EdgeColor', 'k');
hold on;
scatter3(X_nodes(1,:), X_nodes(2,:), X_nodes(3,:), 10, 'r', 'filled');
title(['Nodes on Sphere (N = ' num2str(N) ')']);
axis equal tight; 
view(3);
xlabel('x'); ylabel('y'); zlabel('z');







