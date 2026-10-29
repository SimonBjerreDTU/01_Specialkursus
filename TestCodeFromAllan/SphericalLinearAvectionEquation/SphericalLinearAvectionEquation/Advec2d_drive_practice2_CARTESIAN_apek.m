% Nodal DG-FEM for the spherical linear advection equation
%
% By Allan Peter Engsig-Karup, apek@dtu.dk
close all
clear; 
clc;
Globals2D;

%Polynomial order
N = 4;

IC = 4; % IC=1: f_test=sin(theta)*cos(lambda), IC=2: f_test=X2D, IC=4: Gaussian
%Icosahedral type
generation = 3;
type = 'ico';

% Define Gaussian center in spherical coordinates
lambda0 = deg2rad(225)%295); % center longitude in radians
theta0 = deg2rad(65);    % center colatitude (0 at north pole)
% Sigma controls spread (radians) of guassian IC
sigma = deg2rad(10);    % e.g., 15 degrees standard deviation
v_phi = 0.4;
v_lambda = 0.4;

% Generate spherical mesh 
[P, tri] = generateSphereMesh(generation, type);
EToV = tri;
K = size(EToV,1);
VX = P(1,:);
VY = P(2,:);
VZ = P(3,:);

% Initialize reference element
StartUp2D;
MassMatrix2D = MassMatrix;
Nk2D = K;
Np2D = length(r(:));
Dr2D = Dr;
Ds2D = Ds;

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

% Coefficients for directional derivative
rv2D = zeros(length(r(:)),K);
sv2D = zeros(length(r(:)),K);

% Contravariant metric coefficients
g11_2D = zeros(length(r(:)),K);
g12_2D = zeros(length(r(:)),K);
g22_2D = zeros(length(r(:)),K);

% Spherical surface Jacobian
J2D = zeros(length(r(:)),K);

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

    xr_vec = [xr, yr, zr];
    xs_vec = [xs, ys, zs];

    n_vec = cross(xr_vec, xs_vec, 2);  % normal
    J = sqrt(sum(n_vec.^2, 2));        % surface Jacobian
    n_unit = n_vec ./ J;
    ar = cross(xs_vec, n_unit, 2) ./ J;
    as = cross(n_unit, xr_vec, 2) ./ J;
    rx = ar(:,1); 
    ry = ar(:,2); 
    rz = ar(:,3);
    sx = as(:,1); 
    sy = as(:,2); 
    sz = as(:,3);

    if 0
        %Metric tensor
        g11 = xr.^2  + yr.^2  + zr.^2;
        g12 = xr.*xs + yr.*ys + zr.*zs;
        g21 = xr.*xs + yr.*ys + zr.*zs;
        g22 = xs.^2  + ys.^2  + zs.^2;

        % Metric tensor components
        g11_2D(:,k) = g11(:);
        g12_2D(:,k) = g12(:);
        g22_2D(:,k) = g22(:);

        %Surface Jacobian
        J2   = sqrt(g11.*g22 - g12.^2);
    end
    
    J2D(:,k) = J(:);

    if 0
        invDetG = 1 ./ (J.^2);
        g11c =  g22 .* invDetG;
        g22c =  g11 .* invDetG;
        g12c = -g12 .* invDetG;
        g21c = g12c;

        % Example: rotation about Z axis with angular speed omega
        omega = 0.4;
        d_x = -omega * y;
        d_y = +omega * x;
        d_z = zeros(size(x));

        %% compute velocity vector, d X x
        %D = [d_x(:), d_y(:), d_z(:)];
        %X = [x(:), y(:), z(:)];
        %U = cross(D,X,2);
        %vx = x*0;
        %vy = x*0;
        %vz = x*0;
        %vx(:) = U(:,1);
        %vy(:) = U(:,2);
        %vz(:) = U(:,3);

        % Compute

        %GeometricFactors3D
        rx = ys./J;
        sx =-yr./J;
        ry =-xs./J;
        sy = xr./J;

        rz = (ys .* zr - zs .* yr) ./ J;
        sz = (zr .* xs - xr .* zs) ./ J;
    end
    rx2D(:,k) = rx(:);
    ry2D(:,k) = ry(:);
    rz2D(:,k) = rz(:);
    sx2D(:,k) = sx(:);
    sy2D(:,k) = sy(:);
    sz2D(:,k) = sz(:);

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
% [c,gidx,X2D,Y2D,FaceValues1D] = ConstructConnectivityTable2D(VX,VY,EToV,P,x,y,varargin)
[c2D,gidx2D,X2D,Y2D,Z2D] = ConstructConnectivityTable2DSphere(VX,VY,VZ,EToV,N,x3D,y3D,z3D);
[M2D]  = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@massmatrixsimple,J2D,MassMatrix2D);
[Sx2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,rx2D,sx2D,Dr2D,Ds2D);
[Sy2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,ry2D,sy2D,Dr2D,Ds2D);
[Sz2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,@advectionmatrix2Dsimple,J2D,MassMatrix2D,rz2D,sz2D,Dr2D,Ds2D);
%[Sx3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,rx3D,sx3D,tx3D,Dr3D,Ds3D,Dt3D);
%[Sy3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,ry3D,sy3D,ty3D,Dr3D,Ds3D,Dt3D);
%[Sz3D] = GlobalAssemblySEMnDFULLsimple(Nk3D,Np3D,c3D,@advectionmatrix3Dsimple,J3D,MassMatrix3D,rz3D,sz3D,tz3D,Dr3D,Ds3D,Dt3D);

% Numerical deriatives using SEM
DMx2D = @(xxx) M2D \ ( Sx2D * xxx(:) );
DMy2D = @(xxx) M2D \ ( Sy2D * xxx(:) );
DMz2D = @(xxx) M2D \ ( Sz2D * xxx(:) );

SMx2D = @(xxx) ( Sx2D * xxx(:) );
SMy2D = @(xxx) ( Sy2D * xxx(:) );
SMz2D = @(xxx) ( Sz2D * xxx(:) );
%DMx3D = @(xxx) M3D \ ( Sx3D * xxx(:) );
%DMy3D = @(xxx) M3D \ ( Sy3D * xxx(:) );
%DMz3D = @(xxx) M3D \ ( Sz3D * xxx(:) );

% Check if mass matrix is correct
u = X2D*0+1;
disp(sprintf('SEM, sphere area   : %.8f',sum(M2D*u(:))))

%% Compose vector field in cartesian coordinates to avoid singularies
% Compute unit normals
n_x = X2D(:);
n_y = Y2D(:);
n_z = Z2D(:);
% normalize to unit vectors
norm_n = sqrt(n_x.^2 + n_y.^2 + n_z.^2);
n_x = n_x ./ norm_n;
n_y = n_y ./ norm_n;
n_z = n_z ./ norm_n;

%% COMPUTE ANALYTICAL SOLUTION

switch IC
    case 1
        % Compute spherical coordinates at global nodes
        lambda_nodes2D = atan2(Y2D(:), X2D(:));
        theta_nodes2D  = acos(Z2D(:));

        sin_theta2D  = sin(theta_nodes2D);
        cos_theta2D  = cos(theta_nodes2D);
        sin_lambda2D = sin(lambda_nodes2D);
        cos_lambda2D = cos(lambda_nodes2D);

        f_test = sin_theta2D .* cos_lambda2D;

        % Partial derivatives of f with respect to theta and lambda
        df_dtheta  =  cos_theta2D .* cos_lambda2D;    % df/dtheta
        df_dlambda = -sin_theta2D .* sin_lambda2D;    % df/dlambda

        % THIS CASE IS THE SAME AS IC=2 !!!
        f_test2 = X2D(:); %sin_theta2D .* cos_lambda2D;
        norm(f_test(:)-f_test2(:),inf)

    case 2
        f_test = X2D(:); %sin_theta2D .* cos_lambda2D;
        % Dummies
        df_dtheta = X2D(:)*0;
        df_dlambda = X2D(:)*0;
    case 3

        % Precompute angles
        lambda = atan2(Y2D(:), X2D(:));
        theta = acos(Z2D(:));

        d_lambda = lambda - lambda0;
        A = cos(d_lambda);
        B = sin(theta0);
        C = cos(theta0);

        cosG = sin(theta) .* A .* B + cos(theta) .* C;

        % Main function
        f_test = exp((cosG - 1) / sigma^2);

        % Derivatives of cosG
        dcosG_dtheta  = cos(theta) .* A .* B - sin(theta) .* C;
        dcosG_dlambda = -sin(theta) .* sin(d_lambda) .* B;

        % Final derivatives of f
        df_dtheta  = (1 / sigma^2) * f_test .* dcosG_dtheta;
        df_dlambda = (1 / sigma^2) * f_test .* dcosG_dlambda;
    case 4 % Gaussian
        %% === Define Gaussian function on the sphere ===

        % Define Gaussian center in spherical coordinates
        %lambda0 = deg2rad(245);   % center longitude in radians
        %theta0 = deg2rad(60);    % center colatitude (0 at north pole)

        % Convert center to Cartesian coordinates
        xc = [ sin(theta0)*cos(lambda0);
               sin(theta0)*sin(lambda0);
               cos(theta0) ];

        % Sigma controls spread (radians)
        % sigma = deg2rad(10);  % e.g., 15 degrees standard deviation

        % Compute dot product of all nodes with center
        x_nodes = [X2D(:), Y2D(:), Z2D(:)];
        dotp = x_nodes * xc;

        % Ensure within [-1,1] to avoid acos issues
        dotp = min(max(dotp, -1), 1);

        % Compute angular distance to center
        alpha = acos(dotp);

        % Evaluate Gaussian
        f_test = exp( -0.5 * (alpha./sigma).^2 );

        % Compute spherical coordinates at global nodes
        lambda_nodes2D = atan2(Y2D(:), X2D(:));
        theta_nodes2D  = acos(Z2D(:));

        sin_theta2D  = sin(theta_nodes2D);
        cos_theta2D  = cos(theta_nodes2D);
        sin_lambda2D = sin(lambda_nodes2D);
        cos_lambda2D = cos(lambda_nodes2D);

        % For derivatives
        C = sin(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) + cos(theta_nodes2D).*cos(theta0);
        C = min(max(C, -1), 1);
        alpha = acos(C);
        sqrt1mC2 = sqrt(1 - C.^2);
        % Handle near-zero sqrt(1 - C^2)
        eps_safe = 1e-10;
        sqrt1mC2(sqrt1mC2 < eps_safe) = eps_safe;

        prefactor = (alpha ./ sigma^2) .* exp(-0.5*(alpha./sigma).^2) ./ sqrt1mC2;

        df_dtheta = prefactor .* ( ...
            cos(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) ...
            - sin(theta_nodes2D).*cos(theta0) );

        df_dlambda = ...
            -prefactor .* (sin(theta_nodes2D).*sin(theta0).*sin(lambda_nodes2D - lambda0));

        epsilon = 1e-6;

        % Perturb lambda
        lambda_plus  = atan2(Y2D(:), X2D(:)) + epsilon;
        lambda_minus = atan2(Y2D(:), X2D(:)) - epsilon;

        % Update
        % Recompute C and Gaussian values
        C_plus  = sin(theta_nodes2D).*sin(theta0).*cos(lambda_plus - lambda0) + cos(theta_nodes2D).*cos(theta0);
        C_minus = sin(theta_nodes2D).*sin(theta0).*cos(lambda_minus - lambda0) + cos(theta_nodes2D).*cos(theta0);

        C_plus  = min(max(C_plus, -1), 1);
        C_minus = min(max(C_minus, -1), 1);

        f_plus  = exp(-0.5 * (acos(C_plus)./sigma).^2);
        f_minus = exp(-0.5 * (acos(C_minus)./sigma).^2);

        df_dlambda_fd = (f_plus - f_minus) / (2 * epsilon);

        % Compare
        fprintf('Max diff in df_dlambda: %.3e\n', norm(df_dlambda - df_dlambda_fd, inf));

        % perturbed theta
        theta_plus = theta_nodes2D + epsilon;
        theta_minus = theta_nodes2D - epsilon;

        % Upper and lower C values
        C_plus = sin(theta_plus).*sin(theta0).*cos(lambda_nodes2D - lambda0) + cos(theta_plus)*cos(theta0);
        C_minus = sin(theta_minus).*sin(theta0).*cos(lambda_nodes2D - lambda0) + cos(theta_minus)*cos(theta0);

        % Clamp C to [-1,1]
        C_plus = min(max(C_plus, -1), 1);
        C_minus = min(max(C_minus, -1), 1);

        % Angular distance alpha
        alpha_plus = acos(C_plus);
        alpha_minus = acos(C_minus);

        % Function values
        f_plus = exp(-0.5 * (alpha_plus ./ sigma).^2);
        f_minus = exp(-0.5 * (alpha_minus ./ sigma).^2);

        % FD derivative
        df_dtheta_fd = (f_plus - f_minus) / (2 * epsilon);
        
        error_theta = norm(df_dtheta - df_dtheta_fd, inf);
        fprintf('Max diff in df_dtheta (FD check): %.3e\n', error_theta);

%        wer 
end

% Geometry terms
r2_xy = X2D(:).^2 + Y2D(:).^2;
r2_xy(r2_xy < 1e-12) = 1e-12;  % prevent division by zero

dlambda_dx = -Y2D(:) ./ r2_xy;
dlambda_dy =  X2D(:) ./ r2_xy;
dlambda_dz =  zeros(size(X2D(:)));

dtheta_dx = zeros(size(X2D(:)));
dtheta_dy = zeros(size(X2D(:)));
Zsafe = min(max(Z2D(:), -1 + 1e-12), 1 - 1e-12);
dtheta_dz = -1 ./ sqrt(1 - Zsafe.^2);
%dtheta_dz = -1 ./ sqrt(1 - Z2D(:).^2);

% Chain rule for Cartesian derivatives
grad_fx_exact = df_dlambda .* dlambda_dx + df_dtheta .* dtheta_dx;
grad_fy_exact = df_dlambda .* dlambda_dy + df_dtheta .* dtheta_dy;
grad_fz_exact = df_dlambda .* dlambda_dz + df_dtheta .* dtheta_dz;

% Dot product of exact gradient and normal
dot_gradn_exact = grad_fx_exact .* n_x + grad_fy_exact .* n_y + grad_fz_exact .* n_z;

% Projected exact gradient
grad_fx_exact_tan = grad_fx_exact - dot_gradn_exact .* n_x;
grad_fy_exact_tan = grad_fy_exact - dot_gradn_exact .* n_y;
grad_fz_exact_tan = grad_fz_exact - dot_gradn_exact .* n_z;


%% Compute numerical derivatives
% Compute raw gradient
grad_fx = DMx2D(f_test);
grad_fy = DMy2D(f_test);
grad_fz = DMz2D(f_test);

% Compute dot product of grad_f with normal
dot_gradn = grad_fx .* n_x + grad_fy .* n_y + grad_fz .* n_z;

% Project onto tangent plane
grad_fx_tan = grad_fx - dot_gradn .* n_x;
grad_fy_tan = grad_fy - dot_gradn .* n_y;
grad_fz_tan = grad_fz - dot_gradn .* n_z;

%% EVALUATE

%% for f_test = X2D(:);
switch IC
    case 1
        df_numeric = grad_fx_tan;
        df_exact   = grad_fx_exact_tan;
    case 2
        df_numeric   = grad_fx_tan;
        df_exact_x   = 1-X2D(:).^2 ; %grad_fx_exact;
        df_exact_y   = -X2D(:).*Y2D(:); %grad_fx_exact;
        df_exact_z   = -X2D(:).*Z2D(:); %grad_fx_exact;
        df_exact     = df_exact_x; % plot this one
    case 4
        df_numeric = grad_fy_tan;
        df_exact   = grad_fy_exact_tan;
end

% Visualize the numerical and exact derivatives and their error
figure;
subplot(1,4,1);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, f_test, 'filled');
axis equal; colorbar;
title('Test function');

subplot(1,4,2);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric, 'filled');
axis equal; colorbar;
title('Numerical derivative');

subplot(1,4,3);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_exact, 'filled');
axis equal; colorbar;
title('Exact derivative');

subplot(1,4,4);
scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric - df_exact, 'filled');
axis equal; colorbar;
title('Error (numerical - exact)');

%return
close all

%Globals2D;
time=0;

%runge-Kutta resiual storage
resu=X2D(:)*0; %;zeros(Np,K);
rhsu=X2D(:)*0;

%Compute time step size
%rLGL=JacobiGQ(0,0,N);
%rmin=abs(rLGL(1)-rLGL(2));
%dtscale=dtscale2D;
%dt=min(dtscale)*rmin*2/3;

u = X2D(:)*0;
u(:) = f_test(:);
%dt = 0.01;
dt = 0.01;
%FinalTime = 2.0;
%FinalTime = 2*pi / v_lambda;% v_phi;
FinalTime = 2*pi / v_phi;% v_phi;
Nsteps = round(FinalTime/dt);
dt = FinalTime/Nsteps;

%div_v = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);

frames = {};   % cell array to collect frames
cmin = min(u(:));
cmax = max(u(:));
padding = 0.05*(cmax - cmin);
cmin = cmin - padding;
cmax = cmax + padding;

omega=0.4;
v_x = omega*(-Y2D(:));
v_y = omega*(X2D(:));
v_z = omega*(X2D(:)*0);

% Projected directional derivative operator
Ddir2D = @(ff) M2D \ ( ...
    v_x .* (SMx2D(ff) - n_x .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) + ...
    v_y .* (SMy2D(ff) - n_y .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) + ...
    v_z .* (SMz2D(ff) - n_z .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) );

% Compute spherical coordinates of the global nodes
lambda_nodes = atan2(Y2D(:), X2D(:));         % longitude in radians [-pi, pi]
theta_nodes = acos(Z2D(:));                   % colatitude in radians [0, pi]
latitude_nodes = pi/2 - theta_nodes;          % latitude in radians [-pi/2, +pi/2]

% Create regular grid in longitude and latitude
n_lon = 360;      % number of longitude samples
n_lat = 180;      % number of latitude samples
lon_grid = linspace(-pi, pi, n_lon);            % longitude grid
lat_grid = linspace(-pi/2, pi/2, n_lat);        % latitude grid
[Lon, Lat] = meshgrid(lon_grid, lat_grid);

% Interpolate solution onto the grid
F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, f_test(:), 'linear', 'nearest');
F_grid = F_interp(Lon, Lat);

% Convert grid to Cartesian coordinates for 3D surface plot
X_surf = cos(Lat).*cos(Lon);
Y_surf = cos(Lat).*sin(Lon);
Z_surf = sin(Lat);

%outer time step loop
tstep=0;
while(time<FinalTime)
    tstep=tstep+1;
    if(time+dt>FinalTime)
        dt=FinalTime-time;
    end

    for INTRK=1:5
        % rhsu = Advec2d_sphereRHS(u, c_phi, c_theta);
        %rhsu = Ddir2D(u);

        % Cartesian formulation
        div_cart = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);
        dot_vn = v_x .* n_x + v_y .* n_y + v_z .* n_z;
        div_v = div_cart - 2 * dot_vn;

        rhsu(:) = -( Ddir2D(u(:)) + u(:) .* div_v(:) );
        resu = rk4a(INTRK) * resu + dt * rhsu;
        u = u + rk4b(INTRK) * resu;

    end

    %increment time
    time=time+dt;

    %if tstep==0 || mod(tstep,5)==0
    if mod(tstep,5)==0
        disp(sprintf('tstep %d/%d',tstep,Nsteps))
        figure(98)
        set(gcf,'position',[25     300   661   552])
        set(gcf,'color', 'white')
        if 0
            scatter3(X2D(:), Y2D(:), Z2D(:), 15, u(:), 'filled');
            axis equal;
            xlim([-1 1]);
            ylim([-1 1]);
            zlim([-1 1]);
            xlabel('X');
            ylabel('Y');
            zlabel('Z');
            colorbar;
            title(sprintf('Solution, time: %.4f',time));
            drawnow
        else
            % Interpolate solution onto the grid
            F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, u(:), 'linear', 'nearest');
            u_grid = F_interp(Lon, Lat);
            surf(X_surf, Y_surf, Z_surf, u_grid, 'EdgeColor', 'none');
%            surf(X_surget(gcf,'f, Y_surf, Z_surf, F_grid, 'EdgeColor', 'none');
            axis equal off;
            xlabel('X'); ylabel('Y'); zlabel('Z');
            %title(sprintf('Solution, time: %.4f, step:%d/%d',time,tstep,Nsteps));
            %title('Solution on sphere surface');
            %colorbar;
            shading interp;
            lighting gouraud;
            %camlight headlight;
            axis tight
            caxis([cmin cmax]);   % <-- fix the color range here
%colorbar;            
            drawnow
        end

        % Capture the frame
        frame = getframe(gcf);
        img = frame2im(frame);
        frames{end+1} = img;

    end
end

gif_filename = 'SEMsphere2Dadvection_simulation.gif';

for k = 1:length(frames)
    [A,map] = rgb2ind(frames{k},256);
    if k==1
        imwrite(A,map,gif_filename,'gif','LoopCount',Inf,'DelayTime',0.025);
    else
        imwrite(A,map,gif_filename,'gif','WriteMode','append','DelayTime',0.025);
    end
end

fprintf('Animated GIF saved to: %s\n', gif_filename);

return













% return
% % compute velocity vector, d X x
% d_x = -omega * Y2D;
% d_y = +omega * X2D;
% d_z = zeros(size(X2D));
% D = [d_x(:), d_y(:), d_z(:)];
% X = [X2D(:), Y2D(:), Z2D(:)];
% U = cross(D,X,2);
% v_x = X2D'*0;
% v_y = X2D'*0;
% v_z = X2D'*0;
% v_x(:) = U(:,1);
% v_y(:) = U(:,2);
% v_z(:) = U(:,3);
% 
% % Projected directional derivative operator
% Ddir2D = @(ff) M2D \ ( ...
%     v_x .* (SMx2D(ff) - n_x .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) + ...
%     v_y .* (SMy2D(ff) - n_y .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) + ...
%     v_z .* (SMz2D(ff) - n_z .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff))) );
% 
% % Projected directional derivative operator
% Ddir2D = @(ff) ...
%     v_x .* (DMx2D(ff) - n_x .* (n_x .* DMx2D(ff) + n_y .* DMy2D(ff) + n_z .* DMz2D(ff))) + ...
%     v_y .* (DMy2D(ff) - n_y .* (n_x .* DMx2D(ff) + n_y .* DMy2D(ff) + n_z .* DMz2D(ff))) + ...
%     v_z .* (DMz2D(ff) - n_z .* (n_x .* DMx2D(ff) + n_y .* DMy2D(ff) + n_z .* DMz2D(ff))) ;
% 
% % Divergence on sphere
% div_cart = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);
% dot_vn = v_x .* n_x + v_y .* n_y + v_z .* n_z;
% div_v = div_cart - 2 * dot_vn;
% 
% %dot_vn = v_x .* n_x + v_y .* n_y + v_z .* n_z;
% 
% %Ddir2D = @(ff) M2D \ ( ...
% %    v_x .* SMx2D(ff) + v_y .* SMy2D(ff) + v_z .* SMz2D(ff) ...
% %    - dot_vn .* (n_x .* SMx2D(ff) + n_y .* SMy2D(ff) + n_z .* SMz2D(ff)) );
% 
% % Define the test function f = sin(theta) * cos(lambda)
% if IC
% 
%     % Compute spherical coordinates at global nodes
%     lambda_nodes2D = atan2(Y2D(:), X2D(:));
%     theta_nodes2D  = acos(Z2D(:));
% 
%     sin_theta2D  = sin(theta_nodes2D);
%     cos_theta2D  = cos(theta_nodes2D);
%     sin_lambda2D = sin(lambda_nodes2D);
%     cos_lambda2D = cos(lambda_nodes2D);
% 
%     f_test = 1+0*X2D(:);% sin_theta2D .* cos_lambda2D;
%     % Partial derivatives of f with respect to theta and lambda
%     df_dtheta  =  cos_theta2D .* cos_lambda2D;    % df/dtheta
%     df_dlambda = -sin_theta2D .* sin_lambda2D;    % df/dlambda
% 
% else
%     %% === Define Gaussian function on the sphere ===
% 
%     % Define Gaussian center in spherical coordinates
%     %lambda0 = deg2rad(245);   % center longitude in radians
%     %theta0 = deg2rad(60);    % center colatitude (0 at north pole)
% 
%     % Convert center to Cartesian coordinates
%     xc = [ sin(theta0)*cos(lambda0);
%            sin(theta0)*sin(lambda0);
%            cos(theta0) ];
% 
%     % Sigma controls spread (radians)
%     % sigma = deg2rad(10);  % e.g., 15 degrees standard deviation
% 
%     % Compute dot product of all nodes with center
%     x_nodes = [X2D(:), Y2D(:), Z2D(:)];
%     dotp = x_nodes * xc;
% 
%     % Ensure within [-1,1] to avoid acos issues
%     dotp = min(max(dotp, -1), 1);
% 
%     % Compute angular distance to center
%     alpha = acos(dotp);
% 
%     % Evaluate Gaussian
%     f_test = exp( -0.5 * (alpha./sigma).^2 );
% 
%     % Compute spherical coordinates at global nodes
%     lambda_nodes2D = atan2(Y2D(:), X2D(:));
%     theta_nodes2D  = acos(Z2D(:));
% 
%     sin_theta2D  = sin(theta_nodes2D);
%     cos_theta2D  = cos(theta_nodes2D);
%     sin_lambda2D = sin(lambda_nodes2D);
%     cos_lambda2D = cos(lambda_nodes2D);
% 
%     % For derivatives
%     C = sin(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) + cos(theta_nodes2D).*cos(theta0);
%     C = min(max(C, -1), 1);
%     alpha = acos(C);
%     sqrt1mC2 = sqrt(1 - C.^2);
% 
%     prefactor = (alpha ./ sigma^2) .* exp(-0.5*(alpha./sigma).^2) ./ sqrt1mC2;
% 
%     df_dtheta = prefactor .* ( ...
%         cos(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) ...
%         - sin(theta_nodes2D).*cos(theta0) );
% 
%     df_dlambda = ...
%     -prefactor .* (sin(theta_nodes2D).*sin(theta0).*sin(lambda_nodes2D - lambda0));
% 
% end
% % Apply directional derivative operator
% df_numeric = Ddir2D(f_test);
% 
% % Gradients of theta and lambda in Cartesian coordinates
% % grad(theta):
% %grad_theta_x = -X2D(:).*Z2D(:)./sqrt(1 - Z2D(:).^2);
% %grad_theta_y = -Y2D(:).*Z2D(:)./sqrt(1 - Z2D(:).^2);
% %grad_theta_z = -sqrt(1 - Z2D(:).^2);
% 
% % Correct grad(theta):
% grad_theta_x = zeros(size(X2D(:)));
% grad_theta_y = zeros(size(Y2D(:)));
% grad_theta_z = -1 ./ sqrt(1 - Z2D(:).^2);
% 
% % Correct grad(lambda):
% denomXY = X2D(:).^2 + Y2D(:).^2;
% denomXY(denomXY<1e-12) = 1e-12;   % avoid zero division near poles
% grad_lambda_x = -Y2D(:) ./ denomXY;
% grad_lambda_y = +X2D(:) ./ denomXY;
% grad_lambda_z = zeros(size(X2D(:)));
% 
% % Combine via chain rule:
% grad_f_x = df_dtheta .* grad_theta_x + df_dlambda .* grad_lambda_x;
% grad_f_y = df_dtheta .* grad_theta_y + df_dlambda .* grad_lambda_y;
% grad_f_z = df_dtheta .* grad_theta_z + df_dlambda .* grad_lambda_z;
% 
% df_exact = grad_f_x .* v_x + grad_f_y .* v_y + grad_f_z .* v_z;
% 
% % Raw gradient
% grad_f_x = DMx2D(f_test);
% grad_f_y = DMy2D(f_test);
% grad_f_z = DMz2D(f_test);
% 
% % Compute projected gradient
% dot_grad_n = grad_f_x .* n_x + grad_f_y .* n_y + grad_f_z .* n_z;
% grad_proj_x = grad_f_x - dot_grad_n .* n_x;
% grad_proj_y = grad_f_y - dot_grad_n .* n_y;
% grad_proj_z = grad_f_z - dot_grad_n .* n_z;
% 
% % Now dot with v
% %df_proj = M2D \ (v_x .* grad_proj_x + v_y .* grad_proj_y + v_z .* grad_proj_z);
% df_proj = v_x .* grad_proj_x + v_y .* grad_proj_y + v_z .* grad_proj_z;
% 
% dot_gradn_exact = grad_f_x .* n_x + grad_f_y .* n_y + grad_f_z .* n_z;
% grad_fx_proj = grad_f_x - dot_gradn_exact .* n_x;
% grad_fy_proj = grad_f_y - dot_gradn_exact .* n_y;
% grad_fz_proj = grad_f_z - dot_gradn_exact .* n_z;
% df_exact_proj = grad_fx_proj .* v_x + grad_fy_proj .* v_y + grad_fz_proj .* v_z;
% 
% %df_exact = df_exact_proj;
% 
% % Compute norms and L2 error
% norm_num   = norm(df_numeric);
% norm_exact = norm(df_exact);
% norm_proj  = norm(df_proj);
% l2_error = sqrt(mean((df_numeric - df_exact).^2));
% l2_error_proj = sqrt(mean((df_proj - df_exact).^2));
% 
% fprintf('\n--- Directional Derivative Test ---\n');
% fprintf('Norm of numerical derivative: %.6e\n', norm_num);
% fprintf('Norm of numerical derivative (proj): %.6e\n', norm_proj);
% fprintf('Norm of exact derivative:     %.6e\n', norm_exact);
% fprintf('Relative norm difference:     %.2f%%\n', 100 * abs(norm_num - norm_exact)/norm_exact);
% fprintf('L2 error:                     %.3e\n', l2_error);
% fprintf('L2 error:                     %.3e\n', l2_error_proj);
% 
% % Visualize the numerical and exact derivatives and their error
% figure;
% subplot(1,3,1);
% scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric, 'filled');
% axis equal; colorbar;
% title('Numerical derivative');
% 
% subplot(1,3,2);
% scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_exact, 'filled');
% axis equal; colorbar;
% title('Exact derivative');
% 
% subplot(1,3,3);
% scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric - df_exact, 'filled');
% axis equal; colorbar;
% title('Error (numerical - exact)');
% 
% figure
% scatter3(X2D(:), Y2D(:), Z2D(:), 15, f_test, 'filled');
% axis equal; colorbar;
% title('Solution')
% 
% %% === Visualize solution as surface and lat-lon plot ===
% 
% % Compute spherical coordinates of the global nodes
% lambda_nodes = atan2(Y2D(:), X2D(:));         % longitude in radians [-pi, pi]
% theta_nodes = acos(Z2D(:));                   % colatitude in radians [0, pi]
% latitude_nodes = pi/2 - theta_nodes;          % latitude in radians [-pi/2, +pi/2]
% 
% % Create regular grid in longitude and latitude
% n_lon = 360;      % number of longitude samples
% n_lat = 180;      % number of latitude samples
% lon_grid = linspace(-pi, pi, n_lon);            % longitude grid
% lat_grid = linspace(-pi/2, pi/2, n_lat);        % latitude grid
% [Lon, Lat] = meshgrid(lon_grid, lat_grid);
% 
% % Interpolate solution onto the grid
% F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, f_test(:), 'linear', 'nearest');
% F_grid = F_interp(Lon, Lat);
% 
% % Convert grid to Cartesian coordinates for 3D surface plot
% X_surf = cos(Lat).*cos(Lon);
% Y_surf = cos(Lat).*sin(Lon);
% Z_surf = sin(Lat);
% 
% %% 3D surface plot
% figure;
% surf(X_surf, Y_surf, Z_surf, F_grid, 'EdgeColor', 'none');
% axis equal off;
% xlabel('X'); ylabel('Y'); zlabel('Z');
% title('Solution on sphere surface');
% colorbar;
% shading interp;
% lighting gouraud;
% camlight headlight;
% 
% %% 2D latitude–longitude projection
% figure;
% imagesc(lon_grid*180/pi, lat_grid*180/pi, F_grid);
% axis xy;
% xlabel('Longitude (deg)');
% ylabel('Latitude (deg)');
% title('Solution in latitude-longitude projection');
% colorbar;
% return
% 
% 
% 
% 
% pause
% close all
% 
% %Globals2D;
% time=0;
% 
% %runge-Kutta resiual storage
% resu=X2D(:)*0; %;zeros(Np,K);
% rhsu=X2D(:)*0;
% 
% %Compute time step size
% %rLGL=JacobiGQ(0,0,N);
% %rmin=abs(rLGL(1)-rLGL(2));
% %dtscale=dtscale2D;
% %dt=min(dtscale)*rmin*2/3;
% 
% u = X2D(:)*0;
% u(:) = f_test(:);
% %dt = 0.01;
% dt = 0.001;
% %FinalTime = 2.0;
% %FinalTime = 2*pi / v_lambda;% v_phi;
% FinalTime = 2*pi / v_phi;% v_phi;
% Nsteps = round(FinalTime/dt);
% dt = FinalTime/Nsteps;
% 
% %div_v = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);
% 
% frames = {};   % cell array to collect frames
% cmin = min(u(:));
% cmax = max(u(:));
% padding = 0.05*(cmax - cmin);
% cmin = cmin - padding;
% cmax = cmax + padding;
% 
% %outer time step loop
% tstep=0;
% while(time<FinalTime)
%     tstep=tstep+1;
%     if(time+dt>FinalTime)
%         dt=FinalTime-time;
%     end
% 
%     for INTRK=1:5
%         % rhsu = Advec2d_sphereRHS(u, c_phi, c_theta);
%         %rhsu = Ddir2D(u);
% 
%         div_cart = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);
%         dot_vn = v_x .* n_x + v_y .* n_y + v_z .* n_z;
%         div_v = div_cart - 2 * dot_vn;
% 
%         rhsu(:) = -( Ddir2D(u(:)) + u(:) .* div_v(:) );
%         resu = rk4a(INTRK) * resu + dt * rhsu;
%         u = u + rk4b(INTRK) * resu;
% 
%     end
% 
%     %increment time
%     time=time+dt;
%     if mod(tstep,5)==0
%         disp(sprintf('tstep %d/%d',tstep,Nsteps))
%         figure(98)
%         if 0
%             scatter3(X2D(:), Y2D(:), Z2D(:), 15, u(:), 'filled');
%             axis equal;
%             xlim([-1 1]);
%             ylim([-1 1]);
%             zlim([-1 1]);
%             xlabel('X');
%             ylabel('Y');
%             zlabel('Z');
%             colorbar;
%             title(sprintf('Solution, time: %.4f',time));
%             drawnow
%         else
%             % Interpolate solution onto the grid
%             F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, u(:), 'linear', 'nearest');
%             u_grid = F_interp(Lon, Lat);
%             surf(X_surf, Y_surf, Z_surf, u_grid, 'EdgeColor', 'none');
% %            surf(X_surf, Y_surf, Z_surf, F_grid, 'EdgeColor', 'none');
%             axis equal off;
%             xlabel('X'); ylabel('Y'); zlabel('Z');
%             title(sprintf('Solution, time: %.4f, step:%d/%d',time,tstep,Nsteps));
%             %title('Solution on sphere surface');
%             %colorbar;
%             shading interp;
%             lighting gouraud;
%             %camlight headlight;
%             axis tight
%             caxis([cmin cmax]);   % <-- fix the color range here
% %colorbar;            
%             drawnow
%         end
% 
%         % Capture the frame
%         frame = getframe(gcf);
%         img = frame2im(frame);
%         frames{end+1} = img;
% 
%     end
% end
% 
% gif_filename = 'SEMsphere2Dadvection_simulation.gif';
% 
% for k = 1:length(frames)
%     [A,map] = rgb2ind(frames{k},256);
%     if k==1
%         imwrite(A,map,gif_filename,'gif','LoopCount',Inf,'DelayTime',0.2);
%     else
%         imwrite(A,map,gif_filename,'gif','WriteMode','append','DelayTime',0.2);
%     end
% end
% 
% fprintf('Animated GIF saved to: %s\n', gif_filename);
% 
% return
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % return
% % % 
% % % %% Differentiation in Longitude and Latitude Directions
% % % 
% % % % Directional derivative operator
% % % [Sdir2D] = GlobalAssemblySEMnDFULLsimple(Nk2D,Np2D,c2D,...
% % %   @directional_advection_matrix2Dsimple,...
% % %   J2D, MassMatrix2D, rv2D, sv2D, Dr2D, Ds2D);
% % % 
% % % Ddir2D = @(ff) M2D \ (Sdir2D * ff(:));
% % % 
% % % %        omega = 0.4;
% % % %v_x = -omega * Y2D(:);
% % % %v_y = +omega * X2D(:);
% % % %v_z = zeros(size(X2D(:)));
% % % %Ddir2D = @(u) v_x(:).*DMx2D(u(:)) + v_y(:).*DMy2D(u(:)) + v_z(:).*DMz2D(u(:));
% % % 
% % % %% === Test directional derivative operator ===
% % % 
% % % % Compute spherical coordinates at global nodes
% % % lambda_nodes2D = atan2(Y2D(:), X2D(:));
% % % theta_nodes2D  = acos(Z2D(:));
% % % 
% % % sin_theta2D  = sin(theta_nodes2D);
% % % cos_theta2D  = cos(theta_nodes2D);
% % % sin_lambda2D = sin(lambda_nodes2D);
% % % cos_lambda2D = cos(lambda_nodes2D);
% % % 
% % % % Convert to Cartesian
% % % v_x = v_lambda * (-sin_lambda2D) + v_phi * cos_theta2D .* cos_lambda2D;
% % % v_y = v_lambda * (+cos_lambda2D) + v_phi * cos_theta2D .* sin_lambda2D;
% % % v_z = v_lambda * (0)             + v_phi * (-sin_theta2D);
% % % 
% % % % Define the test function f = sin(theta) * cos(lambda)
% % % if IC
% % %     f_test = sin_theta2D .* cos_lambda2D;
% % %     % Partial derivatives of f with respect to theta and lambda
% % %     df_dtheta = cos_theta2D .* cos_lambda2D;      % df/dtheta
% % %     df_dlambda = -sin_theta2D .* sin_lambda2D;    % df/dlambda
% % % 
% % % else
% % %     %% === Define Gaussian function on the sphere ===
% % % 
% % %     % Define Gaussian center in spherical coordinates
% % %     %lambda0 = deg2rad(245);   % center longitude in radians
% % %     %theta0 = deg2rad(60);    % center colatitude (0 at north pole)
% % % 
% % %     % Convert center to Cartesian coordinates
% % %     xc = [ sin(theta0)*cos(lambda0);
% % %            sin(theta0)*sin(lambda0);
% % %            cos(theta0) ];
% % % 
% % %     % Sigma controls spread (radians)
% % %     % sigma = deg2rad(10);  % e.g., 15 degrees standard deviation
% % % 
% % %     % Compute dot product of all nodes with center
% % %     x_nodes = [X2D(:), Y2D(:), Z2D(:)];
% % %     dotp = x_nodes * xc;
% % % 
% % %     % Ensure within [-1,1] to avoid acos issues
% % %     dotp = min(max(dotp, -1), 1);
% % % 
% % %     % Compute angular distance to center
% % %     alpha = acos(dotp);
% % % 
% % %     % Evaluate Gaussian
% % %     f_test = exp( -0.5 * (alpha./sigma).^2 );
% % % 
% % %     % For derivatives
% % %     C = sin(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) + cos(theta_nodes2D).*cos(theta0);
% % %     C = min(max(C, -1), 1);
% % %     alpha = acos(C);
% % %     sqrt1mC2 = sqrt(1 - C.^2);
% % % 
% % %     prefactor = (alpha ./ sigma^2) .* exp(-0.5*(alpha./sigma).^2) ./ sqrt1mC2;
% % % 
% % %     df_dtheta = prefactor .* ( ...
% % %         cos(theta_nodes2D).*sin(theta0).*cos(lambda_nodes2D - lambda0) ...
% % %         - sin(theta_nodes2D).*cos(theta0) );
% % % 
% % %     df_dlambda = ...
% % %     -prefactor .* (sin(theta_nodes2D).*sin(theta0).*sin(lambda_nodes2D - lambda0));
% % % 
% % % end
% % % % Apply directional derivative operator
% % % df_numeric = Ddir2D(f_test);
% % % 
% % % % Compute the exact directional derivative:
% % % % For v = e_lambda: df = sin(theta) * d/dlambda f
% % % % d/dlambda f = -sin(theta) * sin(lambda)
% % % % So the directional derivative is:
% % % %   -sin^2(theta) * sin(lambda)
% % % % df_exact = - (sin_theta2D).^2 .* sin_lambda2D;
% % % % Compute gradient of f in Cartesian coordinates
% % % % Recall:
% % % % f = sin(theta)*cos(lambda)
% % % % where:
% % % % theta = acos(z)
% % % % lambda = atan2(y,x)
% % % 
% % % 
% % % % Gradients of theta and lambda in Cartesian coordinates
% % % % grad(theta):
% % % %grad_theta_x = -X2D(:).*Z2D(:)./sqrt(1 - Z2D(:).^2);
% % % %grad_theta_y = -Y2D(:).*Z2D(:)./sqrt(1 - Z2D(:).^2);
% % % %grad_theta_z = -sqrt(1 - Z2D(:).^2);
% % % 
% % % % Correct grad(theta):
% % % grad_theta_x = zeros(size(X2D(:)));
% % % grad_theta_y = zeros(size(Y2D(:)));
% % % grad_theta_z = -1 ./ sqrt(1 - Z2D(:).^2);
% % % 
% % % % Correct grad(lambda):
% % % denomXY = X2D(:).^2 + Y2D(:).^2;
% % % denomXY(denomXY<1e-12) = 1e-12;   % avoid zero division near poles
% % % grad_lambda_x = -Y2D(:) ./ denomXY;
% % % grad_lambda_y = +X2D(:) ./ denomXY;
% % % grad_lambda_z = zeros(size(X2D(:)));
% % % 
% % % % Combine via chain rule:
% % % grad_f_x = df_dtheta .* grad_theta_x + df_dlambda .* grad_lambda_x;
% % % grad_f_y = df_dtheta .* grad_theta_y + df_dlambda .* grad_lambda_y;
% % % grad_f_z = df_dtheta .* grad_theta_z + df_dlambda .* grad_lambda_z;
% % % 
% % % df_exact = grad_f_x .* v_x + grad_f_y .* v_y + grad_f_z .* v_z;
% % % 
% % % if 0
% % %     epsilon = 1e-6;
% % % 
% % % % Perturb along v vector:
% % % x_pert = X2D(:) + epsilon * v_x;
% % % y_pert = Y2D(:) + epsilon * v_y;
% % % z_pert = Z2D(:) + epsilon * v_z;
% % % 
% % % % Re-normalize to unit sphere:
% % % norms = sqrt(x_pert.^2 + y_pert.^2 + z_pert.^2);
% % % x_pert = x_pert ./ norms;
% % % y_pert = y_pert ./ norms;
% % % z_pert = z_pert ./ norms;
% % % 
% % % % Convert to spherical coordinates:
% % % lambda_pert = atan2(y_pert, x_pert);
% % % theta_pert  = acos(z_pert);
% % % 
% % % % Evaluate f at perturbed points:
% % % C_pert = sin(theta_pert).*sin(theta0).*cos(lambda_pert - lambda0) + cos(theta_pert).*cos(theta0);
% % % C_pert = min(max(C_pert, -1),1);
% % % alpha_pert = acos(C_pert);
% % % f_pert = exp(-0.5*(alpha_pert/sigma).^2);
% % % 
% % % % Finite difference derivative:
% % % df_fd = (f_pert - f_test) / epsilon;
% % % l2_error_fd = sqrt(mean((df_fd - df_exact).^2));
% % % fprintf('L2 error vs finite diff: %.3e\n', l2_error_fd);
% % % 
% % % epsilon = 1e-6;
% % % 
% % % theta_pert  = theta_nodes2D + epsilon;
% % % lambda_pert = lambda_nodes2D;
% % % 
% % % C_pert_theta = sin(theta_pert).*sin(theta0).*cos(lambda_pert - lambda0) + cos(theta_pert).*cos(theta0);
% % % C_pert_theta = min(max(C_pert_theta, -1),1);
% % % alpha_pert_theta = acos(C_pert_theta);
% % % f_pert_theta = exp(-0.5*(alpha_pert_theta/sigma).^2);
% % % 
% % % df_dtheta_fd = (f_pert_theta - f_test) / epsilon;
% % % 
% % % theta_pert = theta_nodes2D;
% % % lambda_pert = lambda_nodes2D + epsilon;
% % % 
% % % C_pert_lambda = sin(theta_pert).*sin(theta0).*cos(lambda_pert - lambda0) + cos(theta_pert).*cos(theta0);
% % % C_pert_lambda = min(max(C_pert_lambda, -1),1);
% % % alpha_pert_lambda = acos(C_pert_lambda);
% % % f_pert_lambda = exp(-0.5*(alpha_pert_lambda/sigma).^2);
% % % 
% % % df_dlambda_fd = (f_pert_lambda - f_test) / epsilon;
% % % 
% % % err_theta = sqrt(mean((df_dtheta - df_dtheta_fd).^2));
% % % err_lambda = sqrt(mean((df_dlambda - df_dlambda_fd).^2));
% % % 
% % % fprintf('L2 error in df/dtheta vs finite diff: %.3e\n', err_theta);
% % % fprintf('L2 error in df/dlambda vs finite diff: %.3e\n', err_lambda);
% % % 
% % % %    wer
% % % end
% % % 
% % % % Compute norms and L2 error
% % % norm_num = norm(df_numeric);
% % % norm_exact = norm(df_exact);
% % % l2_error = sqrt(mean((df_numeric - df_exact).^2));
% % % 
% % % fprintf('\n--- Directional Derivative Test ---\n');
% % % fprintf('Norm of numerical derivative: %.6e\n', norm_num);
% % % fprintf('Norm of exact derivative:     %.6e\n', norm_exact);
% % % fprintf('Relative norm difference:     %.2f%%\n', 100 * abs(norm_num - norm_exact)/norm_exact);
% % % fprintf('L2 error:                     %.3e\n', l2_error);
% % % 
% % % % Visualize the numerical and exact derivatives and their error
% % % figure;
% % % subplot(1,3,1);
% % % scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric, 'filled');
% % % axis equal; colorbar;
% % % title('Numerical derivative');
% % % 
% % % subplot(1,3,2);
% % % scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_exact, 'filled');
% % % axis equal; colorbar;
% % % title('Exact derivative');
% % % 
% % % subplot(1,3,3);
% % % scatter3(X2D(:), Y2D(:), Z2D(:), 15, df_numeric - df_exact, 'filled');
% % % axis equal; colorbar;
% % % title('Error (numerical - exact)');
% % % 
% % % figure
% % % scatter3(X2D(:), Y2D(:), Z2D(:), 15, f_test, 'filled');
% % % axis equal; colorbar;
% % % title('Solution')
% % % 
% % % %% === Visualize solution as surface and lat-lon plot ===
% % % 
% % % % Compute spherical coordinates of the global nodes
% % % lambda_nodes = atan2(Y2D(:), X2D(:));         % longitude in radians [-pi, pi]
% % % theta_nodes = acos(Z2D(:));                   % colatitude in radians [0, pi]
% % % latitude_nodes = pi/2 - theta_nodes;          % latitude in radians [-pi/2, +pi/2]
% % % 
% % % % Create regular grid in longitude and latitude
% % % n_lon = 360;      % number of longitude samples
% % % n_lat = 180;      % number of latitude samples
% % % lon_grid = linspace(-pi, pi, n_lon);            % longitude grid
% % % lat_grid = linspace(-pi/2, pi/2, n_lat);        % latitude grid
% % % [Lon, Lat] = meshgrid(lon_grid, lat_grid);
% % % 
% % % % Interpolate solution onto the grid
% % % F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, f_test(:), 'linear', 'nearest');
% % % F_grid = F_interp(Lon, Lat);
% % % 
% % % % Convert grid to Cartesian coordinates for 3D surface plot
% % % X_surf = cos(Lat).*cos(Lon);
% % % Y_surf = cos(Lat).*sin(Lon);
% % % Z_surf = sin(Lat);
% % % 
% % % %% 3D surface plot
% % % figure;
% % % surf(X_surf, Y_surf, Z_surf, F_grid, 'EdgeColor', 'none');
% % % axis equal off;
% % % xlabel('X'); ylabel('Y'); zlabel('Z');
% % % title('Solution on sphere surface');
% % % colorbar;
% % % shading interp;
% % % lighting gouraud;
% % % camlight headlight;
% % % 
% % % %% 2D latitude–longitude projection
% % % figure;
% % % imagesc(lon_grid*180/pi, lat_grid*180/pi, F_grid);
% % % axis xy;
% % % xlabel('Longitude (deg)');
% % % ylabel('Latitude (deg)');
% % % title('Solution in latitude-longitude projection');
% % % colorbar;
% % % return
% % % 
% % % pause
% % % close all
% % % 
% % % %Globals2D;
% % % time=0;
% % % 
% % % %runge-Kutta resiual storage
% % % resu=X2D(:)*0; %;zeros(Np,K);
% % % rhsu=X2D(:)*0;
% % % 
% % % %Compute time step size
% % % %rLGL=JacobiGQ(0,0,N);
% % % %rmin=abs(rLGL(1)-rLGL(2));
% % % %dtscale=dtscale2D;
% % % %dt=min(dtscale)*rmin*2/3;
% % % 
% % % u = X2D(:)*0;
% % % u(:) = f_test(:);
% % % dt = 0.02;
% % % %FinalTime = 2.0;
% % % %FinalTime = 2*pi / v_lambda;% v_phi;
% % % FinalTime = 2*pi / v_phi;% v_phi;
% % % Nsteps = round(FinalTime/dt);
% % % dt = FinalTime/Nsteps;
% % % 
% % % div_v = DMx2D(v_x) + DMy2D(v_y) + DMz2D(v_z);
% % % 
% % % frames = {};   % cell array to collect frames
% % % cmin = min(u(:));
% % % cmax = max(u(:));
% % % padding = 0.05*(cmax - cmin);
% % % cmin = cmin - padding;
% % % cmax = cmax + padding;
% % % 
% % % %outer time step loop
% % % tstep=0;
% % % while(time<FinalTime)
% % %     tstep=tstep+1;
% % %     if(time+dt>FinalTime)
% % %         dt=FinalTime-time;
% % %     end
% % % 
% % %     for INTRK=1:5
% % %         % rhsu = Advec2d_sphereRHS(u, c_phi, c_theta);
% % %         %rhsu = Ddir2D(u);
% % %         rhsu(:) = -( Ddir2D(u(:)) + u(:) .* div_v(:) );
% % %         resu = rk4a(INTRK) * resu + dt * rhsu;
% % %         u = u + rk4b(INTRK) * resu;
% % % 
% % %     end
% % % 
% % %     %increment time
% % %     time=time+dt;
% % %     if mod(tstep,10)==0
% % %         figure(98)
% % %         if 0
% % %             scatter3(X2D(:), Y2D(:), Z2D(:), 15, u(:), 'filled');
% % %             axis equal;
% % %             xlim([-1 1]);
% % %             ylim([-1 1]);
% % %             zlim([-1 1]);
% % %             xlabel('X');
% % %             ylabel('Y');
% % %             zlabel('Z');
% % %             colorbar;
% % %             title(sprintf('Solution, time: %.4f',time));
% % %             drawnow
% % %         else
% % %             % Interpolate solution onto the grid
% % %             F_interp = scatteredInterpolant(lambda_nodes, latitude_nodes, u(:), 'linear', 'nearest');
% % %             u_grid = F_interp(Lon, Lat);
% % %             surf(X_surf, Y_surf, Z_surf, u_grid, 'EdgeColor', 'none');
% % % %            surf(X_surf, Y_surf, Z_surf, F_grid, 'EdgeColor', 'none');
% % %             axis equal off;
% % %             xlabel('X'); ylabel('Y'); zlabel('Z');
% % %             title(sprintf('Solution, time: %.4f, step:%d/%d',time,tstep,Nsteps));
% % %             %title('Solution on sphere surface');
% % %             %colorbar;
% % %             shading interp;
% % %             lighting gouraud;
% % %             %camlight headlight;
% % %             axis tight
% % %             caxis([cmin cmax]);   % <-- fix the color range here
% % % %colorbar;            
% % %             drawnow
% % %         end
% % % 
% % %         % Capture the frame
% % %         frame = getframe(gcf);
% % %         img = frame2im(frame);
% % %         frames{end+1} = img;
% % % 
% % %     end
% % % end
% % % 
% % % gif_filename = 'SEMsphere2Dadvection_simulation.gif';
% % % 
% % % for k = 1:length(frames)
% % %     [A,map] = rgb2ind(frames{k},256);
% % %     if k==1
% % %         imwrite(A,map,gif_filename,'gif','LoopCount',Inf,'DelayTime',0.2);
% % %     else
% % %         imwrite(A,map,gif_filename,'gif','WriteMode','append','DelayTime',0.2);
% % %     end
% % % end
% % % 
% % % fprintf('Animated GIF saved to: %s\n', gif_filename);
% % % 
% % % return
% % % 
% % % 
