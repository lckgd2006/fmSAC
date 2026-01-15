function save_subplots_separately(fig_handle, output_dir, varargin)
    % SAVE_SUBPLOTS_SEPARATELY: Save all subplots in a figure as separate files
    %
    % Input Arguments:
    %   fig_handle   - Figure handle, can be a figure object or figure number
    %   output_dir   - Output folder path
    %
    % Optional Parameters:
    %   'FileFormat' - Saved file format, default is 'png'
    %                  Valid values: 'png', 'jpg', 'jpeg', 'tiff', 'pdf', 'eps'
    %   'DPI'        - Image resolution, default is 300
    %   'Prefix'     - File name prefix, default is 'subplot_'
    %   'Silent'     - Whether to display progress information, default is false
    %
    % Examples:
    %   % Create sample figure
    %   fig = figure;
    %   for i = 1:6
    %       subplot(2, 3, i);
    %       imagesc(rand(100, 100));
    %       colorbar;
    %       title(sprintf('Subplot %d', i));
    %   end
    %   
    %   % Save subplots
    %   save_subplots_separately(fig, 'output_plots', 'FileFormat', 'png', 'DPI', 300);
    %
    
    % Default parameter settings
    default_format = 'tiff';
    default_dpi = 600;
    default_prefix = 'subplot_';
    default_silent = false;
    
    % Parse input parameters
    p = inputParser;
    p.addRequired('fig_handle');
    p.addRequired('output_dir');
    p.addParameter('FileFormat', default_format, @(x) ismember(x, {'png', 'jpg', 'jpeg', 'tiff', 'pdf', 'eps'}));
    p.addParameter('DPI', default_dpi, @(x) isscalar(x) && x > 0);
    p.addParameter('Prefix', default_prefix, @ischar);
    p.addParameter('Silent', default_silent, @islogical);
    p.parse(fig_handle, output_dir, varargin{:});
    
    FileFormat = p.Results.FileFormat;
    DPI = p.Results.DPI;
    Prefix = p.Results.Prefix;
    Silent = p.Results.Silent;
    
    % Validate figure handle
    if ~ishandle(fig_handle) || ~strcmp(get(fig_handle, 'Type'), 'figure')
        error('The first input parameter must be a valid figure handle');
    end
    
    % Create output folder
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        if ~Silent
            fprintf('Output folder created: %s\n', output_dir);
        end
    end
    
    % Get all subplots in the figure
    axes_handles = findobj(fig_handle, 'Type', 'axes');
    
    % Filter out possible colorbar axes
    valid_axes = [];
    for i = 1:length(axes_handles)
        % Check if it is a colorbar
        if ~strcmp(get(axes_handles(i), 'Tag'), 'Colorbar')
            valid_axes = [valid_axes; axes_handles(i)];
        end
    end
    
    num_subplots = length(valid_axes);
    
    if num_subplots == 0
        warning('No subplots found in the figure');
        return;
    end
    
    if ~Silent
        fprintf('Found %d subplots, saving to %s...\n', num_subplots, output_dir);
    end
    
    % Save each subplot
    for i = 1:num_subplots
        % Activate current subplot
        axes(valid_axes(i));
        
        % Get subplot title
        title_handle = get(valid_axes(i), 'Title');
        title_text = get(title_handle, 'String');
        
        % Create file name
        if isempty(title_text) || strcmp(title_text, '')
            filename = sprintf('%s%d.%s', Prefix, i, FileFormat);
        else
            % Clean title text for use as file name
            clean_title = strrep(title_text, ' ', '_');
            clean_title = strrep(clean_title, '/', '_');
            clean_title = strrep(clean_title, '\', '_');
            clean_title = strrep(clean_title, ':', '_');
            clean_title = strrep(clean_title, '*', '_');
            clean_title = strrep(clean_title, '?', '_');
            clean_title = strrep(clean_title, '"', '_');
            clean_title = strrep(clean_title, '<', '_');
            clean_title = strrep(clean_title, '>', '_');
            clean_title = strrep(clean_title, '|', '_');
            
            filename = sprintf('%s_%s.%s', Prefix, clean_title, FileFormat);
        end
        
        % Full file path
        filepath = fullfile(output_dir, filename);
        
        % Save subplot
        try
            % Create temporary figure to save single subplot
            temp_fig = figure('Visible', 'off');
            temp_ax = axes(temp_fig);
            
            % Copy contents of original subplot
            copyobj(get(valid_axes(i), 'Children'), temp_ax);
            
            % Copy title
            if ~isempty(title_text) && ~strcmp(title_text, '')
                title(temp_ax, title_text);
            end
            
            % Copy axis labels
            xlabel(temp_ax, get(valid_axes(i), 'XLabel').String);
            ylabel(temp_ax, get(valid_axes(i), 'YLabel').String);
            
            % Copy axis range
            xlim(temp_ax, get(valid_axes(i), 'XLim'));
            ylim(temp_ax, get(valid_axes(i), 'YLim'));
            
            % Copy grid settings
            grid(temp_ax, get(valid_axes(i), 'GridAlpha') > 0);
            
            % Adjust layout
            tightfig(temp_fig);
            
            % Save file
            print(temp_fig, '-dpng', sprintf('-r%d', DPI), filepath);
            
            % Close temporary figure
            close(temp_fig);
            
            if ~Silent
                fprintf('Saved: %s\n', filename);
            end
            
        catch ME
            warning('Error saving subplot %d: %s', i, ME.message);
            if exist('temp_fig', 'var') && ishandle(temp_fig)
                close(temp_fig);
            end
        end
    end
    
    if ~Silent
        fprintf('Saving completed！\n');
    end
end

% Helper function: Adjust figure layout for compact display
function tightfig(fig_handle)
    % Get all axes in the figure
    axes_handles = findobj(fig_handle, 'Type', 'axes');
    
    if isempty(axes_handles)
        return;
    end
    
    % Get current figure position
    fig_pos = get(fig_handle, 'Position');
    
    % Get position and size of the first axis
    ax_pos = get(axes_handles(1), 'Position');
    
    % Set figure size to match axis size
    set(fig_handle, 'Position', [fig_pos(1), fig_pos(2), ax_pos(3)*fig_pos(3), ax_pos(4)*fig_pos(4)]);
    
    % Move axis to top-left corner of the figure
    set(axes_handles(1), 'Position', [0, 0, 1, 1]);
end

% Example usage
function demo()
    % Create sample figure
    fig = figure('Position', [100, 100, 1200, 800]);
    
    % First row of subplots
    subplot(2, 4, 1);
    imagesc(rand(100, 100));
    colorbar;
    title('Sample Structure');
    
    subplot(2, 4, 2);
    imagesc(rand(100, 100) * 0.5 + 0.5);
    colorbar;
    title('Confocal');
    
    subplot(2, 4, 3);
    imagesc(rand(100, 100) * 0.7 + 0.3);
    colorbar;
    title('SAC (1, 500W/cm²)');
    
    subplot(2, 4, 4);
    imagesc(rand(100, 100) * 0.8 + 0.2);
    colorbar;
    title('IntraC (1, 100W/cm²)');
    
    % Second row of subplots
    subplot(2, 4, 5);
    imagesc(rand(100, 100) * 0.6 + 0.4);
    colorbar;
    title('Confocal after Blanking');
    
    subplot(2, 4, 6);
    imagesc(rand(100, 100) * 0.7 + 0.3);
    colorbar;
    title('SAC after Blanking');
    
    subplot(2, 4, 7);
    imagesc(rand(100, 100) * 0.9 + 0.1);
    colorbar;
    title('IntraC after Blanking');
    
    % Hide last subplot position
    subplot(2, 4, 8);
    axis off;
    
    % Adjust overall layout
    sgtitle('Sample Figure: Multi-subplot Layout', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save subplots to folder
    output_dir = 'output_subplots';
    save_subplots_separately(fig, output_dir, 'FileFormat', 'png', 'DPI', 300, 'Prefix', 'plot_');
    
    % Display results
    fprintf('\nExample completed! Subplots have been saved to: %s\n', fullfile(pwd, output_dir));
    fprintf('You can use this function by entering the following command in the MATLAB command window：\n');
    fprintf('save_subplots_separately(figure_handle, ''output_directory'')\n');
end
