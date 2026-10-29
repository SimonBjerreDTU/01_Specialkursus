
function X_nodes = NodesOnSphere(P, tri, N)
% Generate high-order internal nodes on the sphere using warp & blend mappings

[x_ref, y_ref] = Nodes2D(N);
[r, s] = xytors(x_ref, y_ref);
Np = length(r);
K = size(tri, 1);
X_nodes = zeros(3, Np * K);
sk = 1;

for k = 1:K
    v1 = tri(k, 1); v2 = tri(k, 2); v3 = tri(k, 3);
    V1 = P(:, v1); V2 = P(:, v2); V3 = P(:, v3);

    xk = 0.5 * (-(r + s) * V1(1) + (1 + r) * V2(1) + (1 + s) * V3(1));
    yk = 0.5 * (-(r + s) * V1(2) + (1 + r) * V2(2) + (1 + s) * V3(2));
    zk = 0.5 * (-(r + s) * V1(3) + (1 + r) * V2(3) + (1 + s) * V3(3));

    for n = 1:Np
        v = [xk(n); yk(n); zk(n)];
        X_nodes(:, sk) = v / norm(v);
        sk = sk + 1;
    end
end
end

