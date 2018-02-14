clear all
clc
% carrega arquivo de dados
data = importdata('parkson.mat');
Ne_oculta=50;
Ne_Oculta_inicio=10;
Ne_Oculta_fim=100;
Ne_Oculta_passo=10;
Ne_Saida=2;
alpha=0.2;
nClass=2;
iterations=100;
acertos=zeros(100,2);
baseCorreta=zeros(100,2);
[m,n] = size(data);
for i=1:m,
	for j=1:n-1		
		data(i,j)=data(i,j)/max(data(:,j));
	end
end
index=zeros(1,100);
for i=1:100, 
    data=data(randperm(m),:);
    k=round((2*m)/10);

    treino=data(1:3*k,:);
    validacao=data(3*k+1:4*k,:);
    teste=data(4*k+1:end,:);

    y=treino(:,end); 
    x=treino(:,1:end-1);
    
    yv=validacao(:,end); 
    xv=validacao(:,1:end-1);
    
    yt=teste(:,end); 
    xt=teste(:,1:end-1);
    Ett=100000000000000;
    for j=Ne_Oculta_inicio:Ne_Oculta_passo:Ne_Oculta_fim,
        %[W,M]=MLPTrain(Ne_oculta,Ne_Saida,iterations,alpha,x,y);
        %[Et,E0,E1]=MLPTest(W,M,xt,yt);
        [W,M]=ELMTrain(j,Ne_Saida,iterations,x,y);
        [Et,E0,E1]=ELMTest(W,M,xv,yv);
        if Et<Ett,
            ind=j;
        end
        Ett=Et;
        j;
    end
    [Et,E0,E1]=ELMTest(W,M,xt,yt);
    baseCorreta(i,1)=sum(yt);%somando 0's w 1's teremos o numero de intancias com 1's.
    baseCorreta(i,2)=m-k-baseCorreta(i,1);%m-k é o numero de intanciar para o teste, depois é só retirar os 1's. 
    acertos(i,1)=baseCorreta(i,1)-E1;
    acertos(i,2)=baseCorreta(i,2)-E0;
    index(i)=ind;
    i
end
index=index';
total=(acertos(:,1)+acertos(:,2))./(baseCorreta(:,1)+baseCorreta(:,2));
AcertoMedio=mean(total);
taxMin=min(total)
taxMax=max(total)
taxMed=mean(total)
desvioP=std(total)
Classe1=mean(100*acertos(:,1)./baseCorreta(:,1))
Classe2=mean(100*acertos(:,2)./baseCorreta(:,2))







