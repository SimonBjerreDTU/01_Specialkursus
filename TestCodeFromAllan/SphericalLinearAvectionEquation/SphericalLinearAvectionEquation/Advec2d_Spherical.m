
function [u,time]=Advec2d_Spherical(u, FinalTime,c_phi, c_theta)

Globals2D;
time=0;

%runge-Kutta resiual storage
resu=zeros(Np,K);

%Compute time step size
rLGL=JacobiGQ(0,0,N);
rmin=abs(rLGL(1)-rLGL(2));
dtscale=dtscale2D;
dt=min(dtscale)*rmin*2/3;


%outer time step loop
tstep=0;
while(time<FinalTime)
    tstep=tstep+1;
    if(time+dt>FinalTime)
        dt=FinalTime-time;
    end

    for INTRK=1:5
        rhsu = Advec2d_sphereRHS(u, c_phi, c_theta);
        resu = rk4a(INTRK) * resu + dt * rhsu;
        u = u + rk4b(INTRK) * resu;

    end

    %increment time
    time=time+dt;
end
return


