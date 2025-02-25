% This class maps the numerical values to physical ones to interface with
% implementation of sim_array

classdef xbar < handle % TODO: backend
    properties
        base
        % store the weights
        Vg0 = 1.0; % Initial SET gate voltage ???
        Vg_max = 1.6; % Max SET gate voltage
        Vg_min = 0.7; % Min SET gate voltage
        V_set = 2.5; % Fixed SET voltage
        V_reset = 1.7; % Fixed RESET voltage
        V_gate_reset = 5; % Fixed RESET gate voltage
        
        V_gate_set; % tuned parameter determines the final G

        V_read = 0.2;
        
        ratio_G_W = 100e-6;
        ratio_Vg_G = 1/98e-6;
        
        layer_ratio = [1 1];
        
        % the array conductance for multiply reverse
        % update the value after weight update
        array_G
        
        % if true then plot
        draw = 0;
        
        %history; save history if true
        save = 0;
        G_history = {};
        V_gate_history = {};
        V_reset_history = {};
        I_history = {};
        V_vec_history = {};
        
    end
    methods
        %%
        function obj = xbar( base )
            obj.base = base;
        end
        %%
        function add_layer(obj, weight_dim, net_corner)
            % ADD_LAYER add another layer to the software backend.
            %
            %
            obj.base.add_sub(net_corner, [weight_dim(2)*2 weight_dim(1)] );
        end
        %%
        function initialize_weights(obj, varargin)
            okargs = {'draw', 'save'};
            defaults = {0, 0};
            [obj.draw, obj.save] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.V_gate_set = cellfun(@(x)zeros(x.net_size)+obj.Vg0, obj.base.subs,'UniformOutput',false);
            % update the conductance for software backpropogation
            obj.base.update_subs('GND', obj.V_reset, obj.V_gate_reset); % RESET pulse
            obj.base.update_subs(obj.V_set, 'GND', obj.V_gate_set ); % SET pulse
            
            obj.read_conductance('mode', 'fast');
        end
        %%
        function update(obj, dWs )
            th_reset = 0;
%             th_set = 0;
            
            nlayer = numel(dWs);
            
            if nlayer ~= numel(obj.base.subs)
                error('Wrong number of weight gradients');
            end
            
            Vr = cell(1, nlayer);
            Vg = cell(1, nlayer);
            
            for layer = 1: numel(dWs)
                dW = dWs{layer};
            
                dVg = obj.ratio_Vg_G * obj.ratio_G_W *[dW -dW]';

                Vr{layer} = obj.V_reset .* (dVg < th_reset);
                Vg{layer} = obj.V_gate_set{layer} + dVg;
                Vg{layer}(Vg{layer} > obj.Vg_max) = obj.Vg_max;
                Vg{layer}(Vg{layer} < obj.Vg_min) = obj.Vg_min;
            end

            p1 = obj.base.update_subs('GND', Vr, obj.V_gate_reset); % RESET pulse
            p2 = obj.base.update_subs(obj.V_set, 'GND', Vg ); % SET pulse
            
            obj.V_gate_set =  Vg;
            
            % update the conductance for software backpropogation
            obj.read_conductance('mode', 'fast');
            
            if obj.draw 
                subplot(3,3,2);
                imagesc(p1{2});colorbar;
                title('Reset voltages');
                
                subplot(3,3,3);
                imagesc(p2{3});colorbar;
                title('Gate voltage');
                drawnow;
            end
            
            if obj.save
                obj.V_gate_history{end+1} = p2{3};
                obj.V_reset_history{end+1} = p1{2};
            end
            
        end
        %%
        function fullG = read_conductance(obj, varargin)
            [obj.array_G , fullG] = obj.base.read_subs(varargin{:});
            
            if obj.draw 
                figure(11);
                subplot(3,3,1);
                imagesc(fullG); colorbar;
                title('Conductance');
                drawnow;
            end
            
            if obj.save
                obj.G_history{end+1} = fullG;
            end
        end
        %%
        function output = multiply(obj, vec, layer)
            obj.check_layer(layer);
            
            % Input voltage (either scaling or not-scaling)            
            voltage = [vec; -vec] * obj.V_read; % Nonscaling
            % scaling = max(vec(:)); % Scaling
            % voltage = [vec; -vec] / scaling * obj.V_read; % Scaling
            current = obj.base.subs{ layer }.read_current( voltage, 'gain', 2 );
            
            output = current / obj.V_read / obj.ratio_G_W * obj.layer_ratio(layer); % Nonscaling
            % output = current / obj.V_read / obj.ratio_G_W * obj.layer_ratio(layer) * scaling; % Scaling
            
            if obj.draw >= 2
                figure(11);
                
                if layer <= 3
                    subplot(3,3,layer + 3);
                    plot(1:size(current, 1), current , 'o', 'MarkerSize', 1);

                    title(['I@' num2str(layer) '=' num2str(current(1))]);
                    grid on; box on; 
                    ylim([-2.4e-4 2.4e-4]);
                    
                    subplot(3,3,layer + 6);
                    plot(1:size(voltage, 1), voltage , 'o', 'MarkerSize', 1);

                    title(['V@' num2str(layer) '=' num2str(voltage(1))]);
                    grid on; box on; 
                    ylim([-0.3 0.3]);
                end
                
                drawnow;
            end
            
            if obj.save
                if layer == 1
                    obj.V_vec_history{end+1, layer} = voltage;
                    obj.I_history{end+1, layer} = current;
                else
                    obj.V_vec_history{end, layer} = voltage;
                    obj.I_history{end, layer} = current;
                end
            end
        end
        %%
        function output = multiply_reverse(obj, vec, layer)
            obj.check_layer(layer);
            
            G = obj.array_G;
            w = cellfun(@(x) transpose(x(1:end/2,:)-x(end/2+1:end,:)), G,'UniformOutput',false);
            
            output = w{layer}' * vec / obj.ratio_G_W * obj.layer_ratio(layer);    % w is tranposed compared to upper level algorithrms
        end
        %%
        function check_layer(obj, layer )
            if layer > numel( obj.base.subs )
                error(['layer number should be less than ' num2str(numel(obj.W))]);
            end
        end
    end
end