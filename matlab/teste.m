clear all

A=[0.9 0; 0 0.8]
B=[1 0;0 1]
C=[1 0;0 1]
D=[1 0; 0 0]

G=ss(A,B,C,D,1)

u1=zeros(10,1)
u1(2)=1
u2=zeros(10,1)
u2(3)=1
 
u=[u1 u2]

y=lsim(G,u)


S1=eye(2)
D0=D(1,:)
C1=C(1,:)
C2=C(2,:)

S2=1
C2t=C2


S=[1 0;0 S2]*S1
%S=eye(2,2)

T=eye(2,2)


An=A(1,1)
Bn=B(1,:)
Cn=[C(1,1);A(2,1)]
Dn=[1 0;0 1]

%L=ss(A-B*inv(D)*C,B*inv(D),-inv(D)*C,inv(D),1)

TTT=ss(An,Bn,Cn,Dn,1)

L=ss(An-Bn*inv(Dn)*Cn,Bn*inv(Dn),-inv(Dn)*Cn,inv(Dn),1)

y=lsim(G,u)

y1=y(:,1)
y2=[y(2:end,2);0]
yc=[y1 y2-A(2,2)*y(:,2)]

uu=lsim(L,yc)
%atraso=tf([0 1],[1 0],1)

%uuu=lsim(inv(G)*atraso,y)


M=@(alfa) [A-alfa*eye(2) B;C D]

