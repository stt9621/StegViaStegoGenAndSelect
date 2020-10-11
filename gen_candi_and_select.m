function flag = gen_candi_and_select(BossbassDir, BowsDir, StegoDir, CostDir, ParamsDir, PreStegoDir, PreCostDir, PreGradDir, payload, listNum)

    if not(exist(StegoDir,'dir'))
        mkdir(StegoDir)
    end

    if not(exist(CostDir,'dir'))
        mkdir(CostDir)
    end
    
    if not(exist(ParamsDir,'dir'))
        mkdir(ParamsDir)
    end
    


    %% hyperparams
    IMAGE_SIZE = 256;
    RANDOM_NUM = 100;
    ALPHA = 2.0;
    L_P = 0.5;
    S_P = 0.1;


 
    
    
    
    %% load test index list
    indexListPath = ['./index_list/', num2str(listNum), '/test_list.mat'];
    IndexList = load(indexListPath);
    index_list = IndexList.index;
    len = length(index_list);
    %len = 100;
    
    
    
    
    if_success = zeros(len,1);
    p_list = zeros(len,1);
    
    hill_line = zeros(len,1);
    candi_line = zeros(len,1);
    
    parfor index_it = 1:len
        total_start = tic;
    
        index = index_list(index_it);

        %% load data
        if index <= 10000
            coverPath = [BossbassDir, '/', num2str(index), '.pgm'];
        else
            coverPath = [BowsDir, '/', num2str(index-10000), '.pgm'];
        end
        
        stegoPath = [StegoDir, '/', num2str(index), '.pgm'];
        costPath = [CostDir, '/', num2str(index), '.mat'];
        preCostPath = [PreCostDir, '/', num2str(index), '.mat'];
        preGradPath = [PreGradDir, '/', num2str(index), '.mat'];
        preStegoPath = [PreStegoDir, '/', num2str(index), '.pgm'];
        
        

     
        [pre_rhoP1, pre_rhoM1] = load_cost(preCostPath);
        [sign_grad, grad] = load_grad(preGradPath);
        
        
        %% load cover hill stego   
        cover = double(imread(coverPath));
        hill_stego = double(imread(preStegoPath));
        best_stego = hill_stego;
        best_cost_p1 = pre_rhoP1;
        best_cost_m1 = pre_rhoM1;
        
        
        
        %% Calculate 3 filter of image cover residual & hill stego residual & distance between them
        w = 7;
        col_1 = im2col(cover, [1, w], 'sliding');
        col_2 = im2col(cover', [1, w], 'sliding');
        col = cat(2, col_1, col_2);
        neighbor = cat(1, col(1 : floor(1 * w / 2), :), col(floor(1 * w / 2) + 2 : 1 * w, :));
        target = col(floor(1 * w / 2) + 1, :);
        sol = lsqlin(neighbor', target);
        base_f = -ones(1, w);
        base_f(1 : floor(1 * w / 2)) = sol(1 : floor(1 * w / 2));
        base_f(floor(1 * w / 2) + 2 : 1 * w) = sol(floor(1 * w / 2) + 1 : 1 * w - 1);

        f_array = {
            padarray(base_f, [floor(w / 2), 0]),
            padarray(base_f', [0, floor(w / 2)]),
            conv2(base_f, base_f') / -1,
        };
    
        f_array_length = length(f_array);
        [f_n1, f_n2] = size(f_array{1});
        
        
        [n1, n2] = size(cover);

        residual_cover = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
        residual_hill_stego = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
        for i = 1 : f_array_length
            f_item = f_array{i};
            residual_cover(i, :, :) = conv2(cover, f_item, 'full');
            residual_hill_stego(i, :, :) = conv2(hill_stego, f_item, 'full');
        end       
        
        
        min_dis = sum(sum(sum(abs(residual_hill_stego - residual_cover))));
        hill_line(index_it) = min_dis;
        candi_line(index_it) = min_dis;

        
        
        
        %% preprocessing of grad and cost
        s = (sign_grad<0);
        l = (sign_grad>0);
        se = (sign_grad<=0);
        le = (sign_grad>=0);
            
        temp_grad = grad;
        flat_grad = reshape(temp_grad,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_grad, ~] = sort(flat_grad);
        
        
        temp_pre_rhoP1 = pre_rhoP1;
        temp_pre_rhoM1 = pre_rhoM1;

        flat_pre_rhoP1 = reshape(temp_pre_rhoP1,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_P1, ~] = sort(flat_pre_rhoP1);
        flat_pre_rhoM1 = reshape(temp_pre_rhoM1,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_M1, ~] = sort(flat_pre_rhoM1);
        
        
        
        
        for it = 1:RANDOM_NUM
            
            %% generate random varialbles
            myp = unifrnd(S_P,L_P);


            %% calculate abs_grad abs_cost
            % smaller p grad
            low_grad = x_grad(round(IMAGE_SIZE*IMAGE_SIZE*0.5*myp));
            high_grad = x_grad((IMAGE_SIZE*IMAGE_SIZE)-round(IMAGE_SIZE*IMAGE_SIZE*0.5*myp));
            abs_grad = (temp_grad<low_grad) + (temp_grad>high_grad);

            % smaller p cost 
            num_P1 = x_P1(round(IMAGE_SIZE*IMAGE_SIZE*myp));
            small_pre_rhoP1 = (temp_pre_rhoP1<num_P1);
            num_M1 = x_M1(round(IMAGE_SIZE*IMAGE_SIZE*myp));
            small_pre_rhoM1 = (temp_pre_rhoM1<num_M1);
            
            abs_bool_p1 = abs_grad .* small_pre_rhoP1;
            abs_bool_m1 = abs_grad .* small_pre_rhoM1;
            
            rhoP1 = pre_rhoP1 + (s*0 + ALPHA*le) .* abs_bool_p1;
            rhoM1 = pre_rhoM1 + (l*0 + ALPHA*se) .* abs_bool_m1;




            %% Get embedding costs & stego
            % inicialization
            wetCost = 10^8;

            % adjust embedding costs
            rhoP1(rhoP1 > wetCost) = wetCost; % threshold on the costs
            rhoM1(rhoM1 > wetCost) = wetCost;
            rhoP1(isnan(rhoP1)) = wetCost; % if all xi{} are zero threshold the cost
            rhoM1(isnan(rhoM1)) = wetCost;
            rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
            rhoM1(cover==0) = wetCost;

            stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);
            stego = uint8(stego);
            
            
            residual_temp = zeros(f_array_length, n1 + f_n1 - 1, n2 + f_n2 - 1);
            for i = 1 : f_array_length
                f_item = f_array{i};
                residual_temp(i, :, :) = conv2(stego, f_item, 'full');
            end
            
            
            
            
            %% select processing
            dis_temp = sum(sum(sum(abs(residual_temp - residual_cover))));
            

            
            
            % sum wise
            if (dis_temp < min_dis)
                min_dis = dis_temp;
                best_stego = stego;
                best_cost_p1 = rhoP1;
                best_cost_m1 = rhoM1;
                
                p_list(index_it) = myp;
                
                if_success(index_it) = 1;
                candi_line(index_it) = dis_temp;
   
            end
          
            
        end
        
        best_stego = uint8(best_stego);
        imwrite(best_stego, stegoPath);
        
        save_cost(best_cost_p1, best_cost_m1, costPath);
        
        
        total_end = toc(total_start);
        fprintf("Finish index %d(%d) in %.2fs\n", index, RANDOM_NUM, total_end);   
        
    end
    
    paramsPath_p = [ParamsDir, '/p.mat'];
    save(paramsPath_p, 'p_list');
    
    
    save('./hill_line.mat', 'hill_line');
    save('./candi_line.mat', 'candi_line');
    
    % success rate
    success_rate= sum(if_success) / len; 
    fprintf("Success percentage: %.2f\n", success_rate);

    flag = 'Finish';

end






function save_cost(best_cost_p1, best_cost_m1, costPath)
    
    rhoP1 = best_cost_p1;
    rhoM1 = best_cost_m1;
    save(costPath, 'rhoP1', 'rhoM1');
    
end



function [pre_rhoP1, pre_rhoM1] = load_cost(preCostPath)

    Pre_Rho = load(preCostPath);
    pre_rhoP1 = Pre_Rho.rhoP1;
    pre_rhoM1 = Pre_Rho.rhoM1;

end





function [sign_grad, allgrad] = load_grad(preGradPath)

    Grad = load(preGradPath);
    sign_grad = Grad.sign_grad;
    allgrad = Grad.grad;


end





%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);   
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    if fixEmbeddingChanges == 1
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    else
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    end

    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
    
    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2; 
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
        end
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
