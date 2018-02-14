clear all
clc
% carrega arquivo de dados
data = importdata('parkson.mat');
Ne_oculta=50;
Ne_Saida=2;
nClass=2;
epocas=1000;
bias=-1;
[m,n] = size(data);
y=data(:,end); 
x=data(:,1:end-1);
for i=1:m,
	for j=1:n-1		
		x(i,j)=x(i,j)/max(x(:,j));
	end
end
x=[bias*ones(m,1) x];
a=0; b=0.1; % define intervalo dos pesos
[lin,col] = size(x);
%W=a+(b-a).*rand(Ne_oculta,col);
sig=0.1; % define desvio-padrao dos pesos
W=sig*randn(Ne_oculta,col); % gera numeros gaussianos
M=[];
D=zeros(lin,Ne_Saida);

for t=1:epocas,
        %para cada individuo no banco de dados
        D=zeros(lin,Ne_Saida);
        I=randperm(lin);
        x=x(I,:);
        y=y(I,:);
        Zk=zeros(Ne_oculta+1,lin);
     
        for Ne=1:lin,
            %Foward Propagation
            D(Ne,y(Ne)+1)=1;
            %loop para os pesos sinapticos ocultos
            Ui=W*x(Ne,:)';
            Zi=1./(1+exp(-Ui));
            Zi=[-1 Zi'];
            Zk(:,Ne)=Zi;
        end    
        D=D';
        M = D*pinv(Zk);
end

    A=M*Zk;Et=0;E0=0;E1=0;
    for Ne=1:lin,
        
        Y=zeros(Ne_Saida,1);
        Y=D(:,Ne);
        Yt=A(:,Ne);
        
        
        [out_OK iout_OK]=max(Y);  % Indice da saida desejada de maior valor
        [out_T iout_T]=max(Yt); % Indice do neuronio cuja saida eh a maior
        if iout_OK~=iout_T,   % Conta acerto se os dois indices coincidem 
            if(y(Ne)==0)
                E0=E0+1;
            else
                E1=E1+1;
            end
        end
        Et=E0+E1;
    end 
