function samples = sobol_sample(n,d)
    if nargin == 0
        n = 100;
        d = 1;
    elseif nargin == 1
        d = 1;
    end
    
    p = sobolset(d,'Skip',1e3,'Leap',1e2);
    p = scramble(p,'MatousekAffineOwen');
    samples = net(p,n);
end