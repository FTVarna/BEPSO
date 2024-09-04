% -----------------------------------------------------------------------  %
% Biased Eavesdropping Particle Swarm Optimisation Algorithm               %
%                                                                          %
% Implemented by Fevzi Tugrul Varna - University of Sussex, 2021           %
% -------------------------------------------------------------------------%
% 
% Cite as: ----------------------------------------------------------------%
% F. T. Varna and P. Husbands, "Biased Eavesdropping Particles: A Novel    %
% Bio-inspired Heterogeneous Particle Swarm Optimisation Algorithm," 2021  %
% IEEE Symposium Series on Computational Intelligence (SSCI), Orlando,     %
% FL, USA, 2021, pp. 1-8, doi: 10.1109/SSCI50451.2021.9660113.             %
% -----------------------------------------------------------------------  %
%% inputs: fhd,fId,n,d,range where fId=function no., n=swarm size, d=dimension, range=lower and upper bounds
%% e.g. BEPSO(fhd,5,40,30,[-100 100])
function [fmin] = BEPSO(fhd,fId,n,d,range)
rng('shuffle');
show_progress=false;
FMax = 10^4*d;                            % max func evaluations
TMax = FMax/n;                            % max iterations
%% Parameters of PSO
w1 = 0.99 + (0.2-0.99)*(1./(1 + exp(-5*(2*(1:TMax)/TMax - 1)))); %Nonlinear decrease inertia weight(Sigmoid function)
c1 = 2.5-(1:TMax)*2/TMax;                    %Personal Acceleration Coefficient
c2 = 0.5+(1:TMax)*2/TMax;                    %Social Acceleration Coefficient

%% upper and lower bounds
LB=range(1);
UB=range(2);

%% velocity clamp
MaxV=0.15*(UB-LB);
MinV=-MaxV;

%% parameters of BEPSO
bias_thold=20;                              %bias threshold, when reached, bias is formed
ac_model=zeros(n,3);                        %accumulator model, neg response, pos response, bias value
ac_model(:,3)=randi([-1 1],n,1);            %initial random biases

%signal radius max and min values
SR_max=0.1*(UB-LB)*d;
SR_min=0.01*(UB-LB)*d;

interceptorLimit=round(n/8);    %max (nearest) number of interceptors, increasing may enhance performance but will increase time complexity
status=zeros(1,n);              %update status of particles, either 0 or 1
guidance=zeros(n,d);

%% Initialisation
V=zeros(n,d);           %initial velocities
X=unifrnd(LB,UB,[n,d]); %initial positions
PX=X;                   %initial pbest positions
F=feval(fhd,X',fId);    %function evaluation
PF=F;                   %initial pbest cost
GX=[];                  %gbest solution vector
GF=inf;                 %gbest cost
prevF=zeros(1,n);       %previous cost
prevX=X;                %previous pos

%update gbest
for i=1:n
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end
end

multiswarm=false;       %multiswarm search status
if multiswarm, error("The DS parameter must be false to initiate the algorithm."), end

fitnessErr=zeros(1,n);
lbest=[];
f_lbest=[];
%% Main Loop of PSO
for t=1:TMax

    %reset accumulator decision model and biases when majority of the population is biased
    if sum(ac_model(:,3)~=0)>=n*0.8
        ac_model=zeros(n,3); %reset the accumulator model
    end
    
    if t<TMax, fitnessErr=(F-prevF).^2; end %calculate fitness errors, used for the calculation of lambda

    %activate multiswarm phase
    if mod(t,20)==0
        if multiswarm==true
            multiswarm=false;
        else
            multiswarm=true;
            %divide population into 3 subgroups based on their bias
            lbestN=inf*ones(1,3);
            g1=[]; g2=[]; g3=[]; %subswarm 1..3
            for kk=-1:1
                lbestN(kk+2)=length(find(ac_model(:,3)'==kk)); %number of particles in the subswarm
                if lbestN(kk+2)~=0
                    if kk+2==1
                        g1=find(ac_model(:,3)'==kk);
                    elseif kk+2==2
                        g2=find(ac_model(:,3)'==kk);
                    elseif kk+2==3
                        g3=find(ac_model(:,3)'==kk);
                    end
                end
            end
            [lbest,f_lbest]=getInitialLbest(g1,g2,g3,X,F); %get initial local best solutions
        end
    end

    for i=1:n
        %update inertia weight
        if F(i) >= mean(F), w = w1(t) + 0.15;
            if w>0.99, w = 0.99; end
        else     % average_n < Average_g
            w = w1(t) - 0.15; if w<0.20,  w = 0.20; end
        end

        if t<TMax*0.9
            if multiswarm==true
                %update group positions using pbest and lbest
                if isempty(find(g1==i))==0 %if ith agent belongs to g1
                    V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(lbest(1,:) - X(i,:));
                elseif isempty(find(g2==i))==0
                    V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(lbest(2,:) - X(i,:));
                elseif isempty(find(g3==i))==0
                    V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(lbest(3,:) - X(i,:));
                end

                V(i,:) = max(V(i,:), MinV); V(i,:) = min(V(i,:), MaxV);  % Apply Velocity Limits
            else %non-multiswarm phase
                %seperate population to two groups in this phase
                if mod(t,10)
                    rndPop=randperm(n);
                    A=rndPop(1:n/2);
                    B=rndPop((n/2)+1:n);
                end

                if status(i)==1, exemplar=guidance(i,:);
                elseif status(i)==0, exemplar=GX;
                end
                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(exemplar - X(i,:));
                V(i,:) = max(V(i,:), MinV); V(i,:) = min(V(i,:), MaxV);  % Apply Velocity Limits

                if status(i)==1
                    if ac_model(i,3)==0 %if particle is unbiased
                        if F(i)<prevF(i) %current position is better than previous
                            %positive response
                            ac_model(i,2)=ac_model(i,2)+0.01*fitnessErr(i); %calculate lambda
                        else %negative response
                            ac_model(i,1)=ac_model(i,1)-0.01*fitnessErr(i); %calculate lambda
                        end

                        %update bias
                        if ac_model(i,1)<=-bias_thold,      ac_model(i,3)=-1;
                        elseif ac_model(i,2)>=bias_thold,   ac_model(i,3)=1;
                        end
                    end
                end
                status(i)=0; %reset update status of the ith particle
            end
        else %final phase of the search - t>TMax*0.9
            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            V(i,:) = max(V(i,:), MinV); V(i,:) = min(V(i,:), MaxV); % Apply Velocity Limits
        end

        %update previous position and fitness value
        prevX(i,:)=X(i,:);
        prevF(i)=F(i);

        X(i,:) = X(i,:) + V(i,:); % Update position
        X(i,:) = max(X(i,:), LB); X(i,:) = min(X(i,:), UB); % Apply Lower and Upper Bound Limits
    end

    F=feval(fhd,X',fId); % Evaluation

    for i=1:n
        if F(i)<PF(i), PF(i)=F(i); PX(i,:)=X(i,:); end  %update pbests
        if PF(i)<GF, GF=PF(i); GX=PX(i,:); end          %update gbest
    end

    if t<TMax*0.9
        for i=1:n
            %if current fitness value is better than previous,
            %particle sends a signal call
            if F(i)<prevF(i)
                TP=X(i,:)+rand*(prevX(i,:)-X(i,:));           %transmission point
                dist=bsxfun(@minus,X',TP');                   %distances between the transmission point and other particles
                interceptorList=find(sqrt(sum(dist.*dist))<calculateSR(SR_min,SR_max,F(i),mean(F))); %list of interceptors within the signal range
                
                %select the nearest (interceptorLimit) interceptors for faster performance
                if length(interceptorList)>interceptorLimit, interceptorList=interceptorList(1:interceptorLimit); end

                if isempty(interceptorList)==false             %if there are any interceptors
                    calls=nonuniform_mutation(repmat(X(i,:),length(interceptorList),1),0.1,1,TMax,[LB UB]);
                    for interceptor=1:length(interceptorList)
                        if sum(ismember(A,[i interceptorList(interceptor)]))==2 || sum(ismember(B,[i interceptorList(interceptor)]))==2 %if signaller and interceptor are in the same group (conspecific)
                            if ac_model(interceptorList(interceptor),3)==1 || ac_model(interceptorList(interceptor),3)==0     %signal based guidance
                                guidance(interceptorList(interceptor),:)=calls(interceptor,:);
                                status(interceptorList(interceptor))=1;
                            else                                            %non-signal based guidance
                                guidance(interceptorList(interceptor),:)=getExemplar(length(interceptorList),interceptorList,X,n,i);
                                status(interceptorList(interceptor))=1;
                            end
                        else               %heterospecific
                            if ac_model(interceptorList(interceptor),3)==1 || ac_model(interceptorList(interceptor),3)==-1 %signal based guidance
                                guidance(interceptorList(interceptor),:)=calls(interceptor,:);
                                status(interceptorList(interceptor))=1;
                            else           %non-signal based guidance
                                guidance(interceptorList(interceptor),:)=getExemplar(length(interceptorList),interceptorList,X,n,i);
                                status(interceptorList(interceptor))=1;
                            end
                        end
                    end
                end
            end
        end
    end

    if multiswarm==true
        [lbest,f_lbest]=getInitialLbest(g1,g2,g3,X,F); %get initial local best solutions
    end

    if show_progress, disp(['Iteration ' num2str(t) ': best cost = ' num2str(GF)]); end
end

fmin = GF;

function SR=calculateSR(SR_min,SR_max,f,delta)
if randi([0 1])==0
    SR=unifrnd(SR_min,(SR_min+SR_max)/2);
else
    if f<delta
        SR=unifrnd(SR_min,(SR_min+SR_max)/2);
    else
        SR=unifrnd((SR_min+SR_max)/2,SR_max);
    end
end

%returns the exemplar
function [exemplar]=getExemplar(interceptorN,interceptorList,X,n,i)
if interceptorN>4
    rnd=randperm(length(interceptorList),4);
    c_x(1,:)=X(rnd(1),:) + rand*(X(rnd(2),:)-X(rnd(1),:));
    c_x(2,:)=X(rnd(3),:) + rand*(X(rnd(4),:)-X(rnd(3),:));
    exemplar=mean(c_x);
else
    behaviour=randi([2 3]);
    if behaviour==2
        rndPop=randperm(n);
        non_interceptors=rndPop(find(ismember(rndPop,interceptorList)==0));
        exemplar=X(non_interceptors(1),:);
    else
        group=randperm(n,n/2);
        similarity=zeros(1,length(group));
        for ii=1:length(group)
            similarity(ii)=immse(X(group(ii),:),X(i,:));
        end
        [~,dissimilarId]=max(similarity);
        exemplar=X(group(dissimilarId),:);
    end
end

%returns Lbest for all subgroups
function [lbest,f_lbest]=getInitialLbest(g1,g2,g3,X,F)
if isempty(g1)==false, g1_costs=F(g1); [~,g1_lbest_id]=min(g1_costs); lbest(1,:)=X(g1(g1_lbest_id),:); f_lbest(1)=F(g1(g1_lbest_id)); end
if isempty(g2)==false, g2_costs=F(g2); [~,g2_lbest_id]=min(g2_costs); lbest(2,:)=X(g2(g2_lbest_id),:); f_lbest(2)=F(g2(g2_lbest_id)); end
if isempty(g3)==false, g3_costs=F(g3); [~,g3_lbest_id]=min(g3_costs); lbest(3,:)=X(g3(g3_lbest_id),:); f_lbest(3)=F(g3(g3_lbest_id)); end

function [mutant_pop] = nonuniform_mutation(pop,pm,t,MaxIt,Bound)
b = 5;
[ps,D]=size(pop);
mutant_pop = pop;
for i = 1:ps
    for j = 1:D
        if rand() < pm
            N_mm = diag(rand(1,D));
            if round(rand()) == 0
                mutant_pop(i,j) = pop(i,j) + N_mm(j,j)*(Bound(2) - pop(i,j))*(1 - t/MaxIt)^b;
            else  %round(rand()) == 1
                mutant_pop(i,j) = pop(i,j) - N_mm(j,j)*(pop(i,j) - Bound(1))*(1 - t/MaxIt)^b;
            end
        end
    end
end