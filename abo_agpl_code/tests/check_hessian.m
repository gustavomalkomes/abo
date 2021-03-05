function [total_diff,cdX,ndX] = check_hessian(fun_fx,fun_dx, X, e,n)

% finite difference
d   = length(X);        % dimension
cdX = zeros(n,n,d,d);   % computed deriviatives
ndX = zeros(n,n,d,d);   % numerical derivatives

total_diff = 0;

for i = 1:d
    
    dX    = X;
    dX(i) = X(i)+e; % a small step on i
    
    for j = 1:d,
        fX  = fun_fx(X,j);  
        fdX = fun_fx(dX,j); 
        
        ndX(:,:,i,j)  = (fdX-fX)/e;    % numerical dX
        cdX(:,:,i,j)  = fun_dx(X,i,j); % computed  dX
        
        diff = norm(ndX(:,:,i,j)-cdX(:,:,i,j))/(norm(ndX(:,:,i,j)+cdX(:,:,i,j))+1E-18);
        
        % warning if something seems to be different
        if (diff > 1e-3)
            name = func2str(fun_dx);
            warning(sprintf('%s\nEntry H(%d,%d) is potential wrong, its difference is larger than %f',name(11:end),i,j,diff));
            [(fdX-fX)/e, fun_dx(X,i,j)];
            diff = 0;
        end
        
        total_diff = total_diff + diff;        
    end
end


end
