function save_subplots_separately(fig_handle, output_dir, varargin)
    % SAVE_SUBPLOTS_SEPARATELY 保存图形中的所有子图为单独的文件
    %
    % 语法:
    %   save_subplots_separately(fig_handle, output_dir)
    %   save_subplots_separately(fig_handle, output_dir, 'PropertyName', PropertyValue, ...)
    %
    % 输入参数:
    %   fig_handle   - 图形句柄，可以是图形对象或图形编号
    %   output_dir   - 输出文件夹路径
    %
    % 可选参数 (键值对):
    %   'FileFormat' - 保存的文件格式，默认为 'png'
    %                  可选值: 'png', 'jpg', 'jpeg', 'tiff', 'pdf', 'eps'
    %   'DPI'        - 图像分辨率，默认为 300
    %   'Prefix'     - 文件名前缀，默认为 'subplot_'
    %   'Silent'     - 是否显示进度信息，默认为 false
    %
    % 示例:
    %   % 创建示例图形
    %   fig = figure;
    %   for i = 1:6
    %       subplot(2, 3, i);
    %       imagesc(rand(100, 100));
    %       colorbar;
    %       title(sprintf('Subplot %d', i));
    %   end
    %   
    %   % 保存子图
    %   save_subplots_separately(fig, 'output_plots', 'FileFormat', 'png', 'DPI', 300);
    %
    % 注意:
    %   - 如果输出文件夹不存在，将自动创建
    %   - 子图将按照它们在图形中的顺序编号
    %   - 支持大多数常见的MATLAB图形类型
    
    % 默认参数设置
    default_format = 'tiff';
    default_dpi = 600;
    default_prefix = 'subplot_';
    default_silent = false;
    
    % 解析输入参数
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
    
    % 验证图形句柄
    if ~ishandle(fig_handle) || ~strcmp(get(fig_handle, 'Type'), 'figure')
        error('第一个输入参数必须是有效的图形句柄');
    end
    
    % 创建输出文件夹（如果不存在）
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        if ~Silent
            fprintf('已创建输出文件夹: %s\n', output_dir);
        end
    end
    
    % 获取图形中的所有子图
    axes_handles = findobj(fig_handle, 'Type', 'axes');
    
    % 过滤掉可能的颜色条轴
    valid_axes = [];
    for i = 1:length(axes_handles)
        % 检查是否为颜色条
        if ~strcmp(get(axes_handles(i), 'Tag'), 'Colorbar')
            valid_axes = [valid_axes; axes_handles(i)];
        end
    end
    
    num_subplots = length(valid_axes);
    
    if num_subplots == 0
        warning('在图形中未找到任何子图');
        return;
    end
    
    if ~Silent
        fprintf('找到 %d 个子图，正在保存到 %s...\n', num_subplots, output_dir);
    end
    
    % 保存每个子图
    for i = 1:num_subplots
        % 激活当前子图
        axes(valid_axes(i));
        
        % 获取子图标题（如果有的话）
        title_handle = get(valid_axes(i), 'Title');
        title_text = get(title_handle, 'String');
        
        % 创建文件名
        if isempty(title_text) || strcmp(title_text, '')
            filename = sprintf('%s%d.%s', Prefix, i, FileFormat);
        else
            % 清理标题文本以用作文件名
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
        
        % 完整的文件路径
        filepath = fullfile(output_dir, filename);
        
        % 保存子图
        try
            % 创建临时图形以保存单个子图
            temp_fig = figure('Visible', 'off');
            temp_ax = axes(temp_fig);
            
            % 复制原始子图的内容
            copyobj(get(valid_axes(i), 'Children'), temp_ax);
            
            % 复制标题
            if ~isempty(title_text) && ~strcmp(title_text, '')
                title(temp_ax, title_text);
            end
            
            % 复制轴标签
            xlabel(temp_ax, get(valid_axes(i), 'XLabel').String);
            ylabel(temp_ax, get(valid_axes(i), 'YLabel').String);
            
            % 复制轴范围
            xlim(temp_ax, get(valid_axes(i), 'XLim'));
            ylim(temp_ax, get(valid_axes(i), 'YLim'));
            
            % 复制网格设置
            grid(temp_ax, get(valid_axes(i), 'GridAlpha') > 0);
            
            % 调整布局
            tightfig(temp_fig);
            
            % 保存文件
            print(temp_fig, '-dpng', sprintf('-r%d', DPI), filepath);
            
            % 关闭临时图形
            close(temp_fig);
            
            if ~Silent
                fprintf('已保存: %s\n', filename);
            end
            
        catch ME
            warning('保存子图 %d 时出错: %s', i, ME.message);
            if exist('temp_fig', 'var') && ishandle(temp_fig)
                close(temp_fig);
            end
        end
    end
    
    if ~Silent
        fprintf('保存完成！\n');
    end
end

% 辅助函数：调整图形布局以紧凑显示
function tightfig(fig_handle)
    % 获取图形中的所有轴
    axes_handles = findobj(fig_handle, 'Type', 'axes');
    
    if isempty(axes_handles)
        return;
    end
    
    % 获取当前图形位置
    fig_pos = get(fig_handle, 'Position');
    
    % 获取第一个轴的位置和大小
    ax_pos = get(axes_handles(1), 'Position');
    
    % 设置图形大小以匹配轴大小
    set(fig_handle, 'Position', [fig_pos(1), fig_pos(2), ax_pos(3)*fig_pos(3), ax_pos(4)*fig_pos(4)]);
    
    % 将轴移动到图形的左上角
    set(axes_handles(1), 'Position', [0, 0, 1, 1]);
end

% 示例用法
function demo()
    % 创建示例图形（类似参考图中的结构）
    fig = figure('Position', [100, 100, 1200, 800]);
    
    % 第一行子图
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
    
    % 第二行子图
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
    
    % 隐藏最后一个子图位置
    subplot(2, 4, 8);
    axis off;
    
    % 调整整体布局
    sgtitle('示例图形：多子图布局', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存子图到文件夹
    output_dir = 'output_subplots';
    save_subplots_separately(fig, output_dir, 'FileFormat', 'png', 'DPI', 300, 'Prefix', 'plot_');
    
    % 显示结果
    fprintf('\n示例完成！子图已保存到: %s\n', fullfile(pwd, output_dir));
    fprintf('您可以在MATLAB命令窗口中输入以下命令来使用此函数：\n');
    fprintf('save_subplots_separately(figure_handle, ''output_directory'')\n');
end
