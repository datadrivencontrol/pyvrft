



% 
% Gi=inv(G)
% 
% Gs=ss(G)
% 
% [A,B,C,D]=ssdata(Gs)
% 
% %while rank(D)<2
%     C=C*A;
%     D=C*B;
% %end
% 
% Gn=ss(A,B,C,D,1)
% 
% u1=zeros(10,1)
% u1(3)=1
% u2=zeros(10,1)
% u2(4)=1
% 
% u=[u1 u2]
% 
% y1=lsim(G,u)
% y2=lsim(inv(G),u)
% 
% [y1 y2]


A=[0.9 1; 0 0.8]
B=[1 0;0 1]
C=[1 1;0 1]
D=[1 0; 0 0]

Gss=ss(A,B,C,D,1)
Gtf=tf(Gss)
iGtf=inv(Gtf)*tf([0 1],[1 0],1)
iGss=ss(iGtf)


u1=zeros(10,1)
u1(3)=1
u2=zeros(10,1)
u2(4)=1
 
u=[u1 u2]

y1=lsim(Gss,u)
y2=lsim(iGss,u)

Z=zeros(2,2)
I=eye(2,2)


ML=[D Z;
    C*B D]

%ML2=[D Z Z;
%    C*B D Z;
%    C*A*B C*B D]

DDD=[eye(2,2) zeros(2,2)]


k=sym('k',[2 4])
eq= k*ML==DDD
kkkk=solve(eq,k)
K=eval(subs(k,kkkk))

%K1=[1 0;0 0]
%K2=[0 0;0 1]

K1=K(:,1:2)
K2=K(:,3:4)







AA=[Z Z;B*K1 A-B*(K1*C+K2*C*A)]
BB=[I;B*K2]
CC=[K1 -K1*C-K2*C*A]
DD=K2


AA=[Z Z;B*K1 A-B*(K1*C+K2*C*A)]
BB=[I;B*K2]
CC=[K1 -K1*C-K2*C*A]
DD=K2


GGG=ss(AA,BB,CC,DD,1)

y3=lsim(GGG,u)



S1=eye(2,2)
C1=C(1,:)
C2=C(2,:)


