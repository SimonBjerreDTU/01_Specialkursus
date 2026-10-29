% Nodal DG-FEM for the spherical linear advection equation
%
% By Allan Peter Engsig-Karup, apek@dtu.dk
clear; 
clc;
Globals2D;

%Polynomial order
N = 4;

%Icosahedral type
generation = 2;
type = 'ico';

% Generate spherical mesh 
[P, tri] = generateSphereMesh(generation, type);
EToV = tri;
K = size(EToV,1);

%Cartesian coordinates
x_coor = P(1,:)'; 
y_coor = P(2,:)'; 
z_coor = P(3,:)';

%spherical coordinates
r1 = sqrt(x_coor.^2 + y_coor.^2 + z_coor.^2);
theta = acos(z_coor ./ r1);
phi = atan2(y_coor, x_coor);

VX = phi';                      
VY = theta';

% Initialize reference element
StartUp2D;

MassMatrix2D = MassMatrix;

%Initialize area
total_area = 0;

x3D = zeros(length(r(:)),K);
y3D = zeros(length(r(:)),K);
z3D = zeros(length(r(:)),K);

% Cartesian metrix coefficients
rx2D = zeros(length(r(:)),K);
ry2D = zeros(length(r(:)),K);
sx2D = zeros(length(r(:)),K);
sy2D = zeros(length(r(:)),K);
rz2D = zeros(length(r(:)),K);
sz2D = zeros(length(r(:)),K);

% Contravariant metric coefficients
g11_2D = zeros(length(r(:)),K);
g12_2D = zeros(length(r(:)),K);
g22_2D = zeros(length(r(:)),K);

% Spherical surface Jacobian
J2D = zeros(length(r(:)),K);

xr2D = zeros(length(r(:)),K);
xs2D = zeros(length(r(:)),K);
yr2D = zeros(length(r(:)),K);
ys2D = zeros(length(r(:)),K);
zr2D = zeros(length(r(:)),K);
zs2D = zeros(length(r(:)),K);

rDirLon2D = zeros(length(r(:)),K);
sDirLon2D = zeros(length(r(:)),K);

rDirLat2D = zeros(length(r(:)),K);
sDirLat2D = zeros(length(r(:)),K);

rv2D = zeros(length(r(:)),K);
sv2D = zeros(length(r(:)),K);

for k = 1:K
    % Get vertices of triangle k
    v1 = P(:, EToV(k,1));
    v2 = P(:, EToV(k,2));
    v3 = P(:, EToV(k,3));

    % Map from reference to physical triangle
    x = 0.5*(-(r+s)*v1(1) + (1+r)*v2(1) + (1+s)*v3(1));
    y = 0.5*(-(r+s)*v1(2) + (1+r)*v2(2) + (1+s)*v3(2));
    z = 0.5*(-(r+s)*v1(3) + (1+r)*v2(3) + (1+s)*v3(3));

    % Project to sphere
    norms = sqrt(x.^2 + y.^2 + z.^2);
    x = x ./ norms;
    y = y ./ norms;
    z = z ./ norms;

    % Store nodes in array
    x3D(:,k) = x(:);
    y3D(:,k) = y(:);
    z3D(:,k) = z(:);

    % Geometric factors
    xr = Dr*x; xs = Ds*x; 
    yr = Dr*y; ys = Ds*y; 
    zr = Dr*z; zs = Ds*z;

    %Metric tensor
    g11 = xr.^2 + yr.^2 + zr.^2;
    g12 = xr.*xs + yr.*ys + zr.*zs;
    g21 = xr.*xs + yr.*ys + zr.*zs;
    g22 = xs.^2 + ys.^2 + zs.^2;
    
    % Metric tensor components
    g11_2D(:,k) = g11(:);
    g12_2D(:,k) = g12(:);
    g22_2D(:,k) = g22(:);

    %Surface Jacobian
    J   = sqrt(g11.*g22 - g12.^2);
    J2D(:,k) = J(:);

    lambda_nodes = atan2(y, x);
    theta_nodes = acos(z);
    
    sin_theta = sin(theta_nodes);
    cos_theta = cos(theta_nodes);
    cos_lambda = cos(lambda_nodes);
    sin_lambda = sin(lambda_nodes);
    
    tLatX = -cos_lambda .* sin_theta;
    tLatY = -sin_lambda .* sin_theta;
    tLatZ = cos_theta;    
    
    tLambdaX = -sin_theta .* sin_lambda;
    tLambdaY =  sin_theta .* cos_lambda;

    %tLambdaX = -sin(lambda_nodes);
    %tLambdaY = cos(lambda_nodes);
    tLambdaZ = zeros(size(lambda_nodes));
    alpha_r_lambda = tLambdaX .* xr + tLambdaY .* yr + tLambdaZ .* zr;
    alpha_s_lambda = tLambdaX .* xs + tLambdaY .* ys + tLambdaZ .* zs;    

    invDetG = 1 ./ (J.^2);
    g11c =  g22 .* invDetG;
    g22c =  g11 .* invDetG;
    g12c = -g12 .* invDetG;
    g21c = g12c;

    rDirLon = alpha_r_lambda .* g11c + alpha_s_lambda .* g12c;
    sDirLon = alpha_r_lambda .* g12c + alpha_s_lambda .* g22c;

    rDirLon2D(:,k) = rDirLon(:);
    sDirLon2D(:,k) = sDirLon(:);    

    alpha_r_lat = tLatX .* xr + tLatY .* yr + tLatZ .* zr;
    alpha_s_lat = tLatX .* xs + tLatY .* ys + tLatZ .* zs;
    
    rDirLat = alpha_r_lat .* g11c + alpha_s_lat .* g12c;
    sDirLat = alpha_r_lat .* g12c + alpha_s_lat .* g22c;

    rDirLat2D(:,k) = rDirLat(:);
    sDirLat2D(:,k) = sDirLat(:);    

    % directional metric coefficients
    % https://chatgpt.com/c/68741472-c864-800c-bcc2-99025a2c0e40
    v_lambda = 1; % longitudinal advection speed
    v_phi    = 0.1; % lattitudinal advection speed
    %alpha_r = v_lambda*xr;
    %alpha_s = v_phi*xs;
    % compute the cartesian speeds v_x, v_y, v_z from the speeds in the longitudinal and
    % lattitudeinal directions
    %v_x = v_lambda * cos(theta_nodes); 
    %v_y = v_phi * sin(theta_nodes);
    %v_z = zeros(size(theta_nodes));

    % compute by scaling the unit vectors in the cartesian directions
    v_x = v_lambda * (-sin_lambda) + v_phi * cos_theta .* cos_lambda;
    v_y = v_lambda * (+cos_lambda) + v_phi * cos_theta .* sin_lambda;
    v_z = v_lambda * (0)           + v_phi * (-sin_theta);

    alpha_r = v_x .* xr + v_y .* yr + v_z .* zr;
    alpha_s = v_x .* xs + v_y .* ys + v_z .* zs;
    % form the advection speed in the directional operator
    rv = alpha_r.*g11c + alpha_s.*g21c;
    sv = alpha_r.*g12c + alpha_s.*g22c;
    rv2D(:,k) = rv(:);
    sv2D(:,k) = sv(:);

    %GeometricFactors3D
%    [rx,sx,tx,ry,sy,ty,rz,sz,tz,J] = GeometricFactors3D(x,y,z,Dr,Ds,Dt)
    rx = ys./J; 
    sx =-yr./J; 
    ry =-xs./J; 
    sy = xr./J;

    rx2D(:,k) = rx(:);
    ry2D(:,k) = ry(:);
    %rz2D(:,k) = rz(:);
    sx2D(:,k) = sx(:);
    sy2D(:,k) = sy(:);   
    %sz2D(:,k) = sz(:);

    % Scale mass matrix
    M_k = diag(J)*MassMatrix;

     %Total area
    total_area = total_area +sum(sum(M_k));
end

% --- Output ---
disp('Check mass matrix and jacobians by integrating at the sphere surface')
fprintf('Exact area is 4*pi : %.8f\n', 4*pi);
fprintf('Computed total area: %.8f\n', total_area);

%% SPECTRAL ELEMENT OPERATORS

%% GENERATE CONNECTIVITY TABLE
% create connectivity list with global numbers for each local degree of freedom
VX = P(1,:)';
VY = P(2,:)';
VZ = P(3,:)';
EToV = tri;
% [c,gidx,X2D,Y2D,FaceValues1D] = ConstructConnectivityTable2D(VX,VY,EToV,P,x,y,varargin)
[c2D,gidx2D,X2D,Y2D,Z2D] = ConstructConnectivityTable2DSphere(VX,VY,VZ,EToV,N,x3D,y3D,z3D);
Nk2D = K;
Np2D = length(r(:));
Dr2D = Dr;
Ds2D = Ds;
[M2D]  = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@massmatrixsimple,J2D,MassMatrix2D);
[Sx2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,rx2D,sx2D,Dr2D,Ds2D);
[Sy2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,ry2D,sy2D,Dr2D,Ds2D);
[Sz2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,zr2D,zs2D,Dr2D,Ds2D);
%[Sx3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,rx3D,sx3D,tx3D,Dr3D,Ds3D,Dt3D);
%[Sy3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,ry3D,sy3D,ty3D,Dr3D,Ds3D,Dt3D);
%[Sz3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,rz3D,sz3D,tz3D,Dr3D,Ds3D,Dt3D);
DMx2D = @(xxx) M2D \ ( Sx2D * xxx(:) );
DMy2D = @(xxx) M2D \ ( Sy2D * xxx(:) );
DMz2D = @(xxx) M2D \ ( Sz2D * xxx(:) );
%DMx3D = @(xxx) M3D \ ( Sx3D * xxx(:) );
%DMy3D = @(xxx) M3D \ ( Sy3D * xxx(:) );
%DMz3D = @(xxx) M3D \ ( Sz3D * xxx(:) );

% Check if mass matrix is correct
u = X2D*0+1;
disp(sprintf('SEM, sphere area   : %.8f',sum(M2D*u(:))))


%% Differentiation in Longitude and Latitude Directions

% Directional derivative operator
[Sdir2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,...
  @directional_advection_matrix2Dsimple,...
  J2D, MassMatrix2D, rv2D, sv2D, Dr2D, Ds2D);

Ddir2D = @(f) M2D \ (Sdir2D * f(:));

%% === Test directional derivative operator ===

% Compute spherical coordinates at global nodes
lambda_nodes2D = atan2(Y2D(:), X2D(:));
theta_nodes2D = acos(Z2D(:));

sin_theta2D = sin(theta_nodes2D);
cos_theta2D = cos(theta_nodes2D);
sin_lambda2D = sin(lambda_nodes2D);
cos_lambda2D = cos(lambda_nodes2D);

% Define the test function f = sin(theta) * cos(lambda)
f_test = sin_theta2D .* cos_lambda2D;

% Apply directional derivative operator
df_numeric = Ddir2D(f_test);

% Compute the exact directional derivative:
% For v = e_lambda: df = sin(theta) * d/dlambda f
% d/dlambda f = -sin(theta) * sin(lambda)
% So the directional derivative is:
%   -sin^2(theta) * sin(lambda)
df_exact = - (sin_theta2D).^2 .* sin_lambda2D;

% Compute norms and L2 error
norm_num = norm(df_numeric);
norm_exact = norm(df_exact);
l2_error = sqrt(mean((df_numeric - df_exact).^2));

fprintf('\n--- Directional Derivative Test ---\n');
fprintf('Norm of numerical derivative: %.6e\n', norm_num);
fprintf('Norm of exact derivative:     %.6e\n', norm_exact);
fprintf('Relative norm difference:     %.2f%%\n', 100 * abs(norm_num - norm_exact)/norm_exact);
fprintf('L2 error:                     %.3e\n', l2_error);

% Visualize the numerical and exact derivatives and their error
figure;
subplot(1,3,1);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric, 'filled');
axis equal; colorbar;
title('Numerical derivative');

subplot(1,3,2);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_exact, 'filled');
axis equal; colorbar;
title('Exact derivative');

subplot(1,3,3);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric - df_exact, 'filled');
axis equal; colorbar;
title('Error (numerical - exact)');

return
