function cc = maxEntropyMatrix(V)
    % Newton's method for max matrix entropy
    maxiterCG = 5000;
    maxiterNewton = 100;
    tolNewton = 1e-8;

    n = size(V,1);
    M = [eye(n-1,n-1); -ones(1,n-1)];
    last = zeros(n,1);
    last(end) = 1;

    c = 1/n*ones(n-1,1); %initial guess
    update = zeros(n-1,1); %initial CG guess
    for k=1:maxiterNewton %Newton iteration
        A = V*diag(M*c + last)*V';
        [U,LambdaM,~] = svd(A);
        Lambda = diag(LambdaM);
        dE = 1+log(Lambda);
        grad = M'*diag(V'*U*diag(dE)*U'*V);
        disp(['Newton iteration= ', num2str(k), ', norm(grad f)= ',...
        num2str(norm(grad))]);
        if (norm(grad) < tolNewton)
            break;
        end
        applyHess = @(dc) applyEntropyHessian(V,U,Lambda,dc);
        %Conjugate gradient to get H^-1 grad
        %update = randn(length(trilInds),1);
        res = grad - applyHess(update);
        dir = res;
        for jj=1:maxiterCG
            Hdir = applyHess(dir);
            aa = res'*res/(dir'*Hdir);
            update = update + aa*dir;
            res = res - aa*Hdir;
            if (norm(res) < 0.9*min(1,norm(grad)^2))
                break;
            end
            dir = res + 1/(res'*res)*res*(res'*dir);
            if (jj==maxiterCG)
                disp('maximum CG iterations reached');
            end
        end
        %disp(['CG iterations= ', num2str(jj)]);

        %Linesearch for step length
        options = optimset('Display','off');
        warning('off','all');
        sopt = fminunc(@(s) objectiveFct(c - s*update,V),1,options);
        warning('on','all');
        c = c - sopt*update;
        %{
        %Since the starting point is not smart, do a few half-steps before
        % proceeding to full Newton.
        if (k < 5)
            c = c - 0.5*update;
        else
            c = c - update;
        end
        %}
    end
    cc = [c; 1-sum(c)];
end

function Hdc = applyEntropyHessian(V,U,Lambda,dc)
%Applys the hessian of the constraint reduced entropy Hessian
    n = size(V,1);
    M = [eye(n-1,n-1); -ones(1,n-1)];
    dA = V*diag(M*dc)*V';
    C = U'*dA*U;
    for ii=1:n
        for jj=1:n
            if (ii == jj)
                C(ii,jj) = 0;
            else
                C(ii,jj) = C(ii,jj)/(Lambda(jj)-Lambda(ii));
            end
        end
    end
    dU = U*C;
    dE = 1+log(Lambda);
    ddE = 1./Lambda;
    dLambda = diag(U'*dA*U);
    Hdc = M'*diag(V'*(2*dU*diag(dE)*U' + U*diag(ddE.*dLambda)*U')*V);
end

function [f,g] = objectiveFct(c,V)
    n = length(c) + 1;
    last = zeros(n,1);
    last(end) = 1;
    M = [eye(n-1,n-1); -ones(1,n-1)];
    A = V*diag(M*c + last)*V';
    [U,LambdaM,~] = svd(A);
    Lambda = diag(LambdaM);
    f = sum(Lambda.*log(Lambda)); %function
    dE = 1+log(Lambda);
    g = M'*diag(V'*U*diag(dE)*U'*V); %gradient
end