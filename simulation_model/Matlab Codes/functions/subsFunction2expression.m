function Fun = subsFunction2expression(varargin)
syms t
Fun = varargin{1};
X = varargin{2};

DDX = diff(X,t,t);
DX = diff(X,t);

var = {X,DX,DDX};

nu = nargin-2;
for i=nu:-1:1
    f = subs(Fun,var{i},varargin{i+2});
    Fun = f;
end

end
