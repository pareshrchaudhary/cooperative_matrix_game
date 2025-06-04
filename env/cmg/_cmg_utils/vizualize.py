import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CMGVisualizer:
    @staticmethod
    def get_discrete_colors(payoff_matrix, M=None):
        if torch.is_tensor(payoff_matrix):
            M_values = payoff_matrix.cpu().numpy()
        else:
            M_values = payoff_matrix
            
        if torch.is_tensor(payoff_matrix):
            payoff_flat = payoff_matrix.flatten()
            mask = payoff_flat > 0
            if torch.any(mask):
                unique_values = torch.unique(payoff_flat[mask]).tolist()
                unique_values = sorted(unique_values)
            else:
                unique_values = []
        else:
            unique_values = sorted(set(M_values.flatten()) - {0})
        
        if M is not None and M > 1:
            cmap = plt.colormaps['tab10'] if hasattr(plt.cm, 'colormaps') else plt.cm.get_cmap('tab10', max(M, 1))
            colors = {val: cmap(i % 10)[:3] for i, val in enumerate(unique_values)}
        else:
            if len(unique_values) > 1:
                max_val = max(unique_values)
                min_val = min(unique_values)
                norm_range = max(max_val - min_val, 1e-10)
                colors = {val: [0, 0.2 + 0.8 * (val - min_val) / norm_range, 0.8 - 0.8 * (val - min_val) / norm_range] 
                        for val in unique_values}
            else:
                colors = {val: [0, 0.8, 0.2] for val in unique_values}
                
        return unique_values, colors

    @staticmethod
    def plot_discrete_payoff(ax, payoff_matrix, n_dim, thetas=None, opponent_thetas=None, layout=None, with_overlay=False, agents=None):
        if torch.is_tensor(payoff_matrix):
            M_values = payoff_matrix.cpu().numpy()
        else:
            M_values = payoff_matrix
            
        M = getattr(thetas, 'M', None) if thetas is not None else None
        unique_values, colors = CMGVisualizer.get_discrete_colors(payoff_matrix, M)
        
        colors = {val: np.clip(colors[val], 0, 1) for val in unique_values}
        
        img_tensor = torch.ones((n_dim, n_dim, 3))
        
        if len(unique_values) > 0:
            if torch.is_tensor(payoff_matrix):
                payoff_cpu = payoff_matrix.cpu()
                for val in unique_values:
                    mask = (payoff_cpu == val)
                    if torch.any(mask):
                        color_tensor = torch.tensor(colors[val])
                        for i in range(n_dim):
                            for j in range(n_dim):
                                if mask[i, j]:
                                    img_tensor[i, j] = color_tensor
            else:
                img = np.ones((n_dim, n_dim, 3))
                for i in range(n_dim):
                    for j in range(n_dim):
                        val = M_values[i, j]
                        if val > 0:
                            img[i, j] = colors.get(val, [0, 0.5, 0.5])
                img_tensor = torch.from_numpy(img).float()
        
        img = img_tensor.numpy()
        
        high_dim_threshold = 10
        is_high_dim = n_dim > high_dim_threshold
        too_many_values = len(unique_values) > 7

        ax.imshow(img, origin="upper", interpolation='nearest' if is_high_dim else None)
        
        if agents:
            agent_names = []
            for a in agents:
                name = a.replace('agent_', 'Agent ')
                if name.endswith('_C'):
                    name = name.replace('_C', ' (Column)')
                elif name.endswith('_R'):
                    name = name.replace('_R', ' (Row)')
                agent_names.append(name)
        else:
            agent_names = ["Agent C (Column)", "Agent R (Row)"]
        
        ax.set_xlabel(f"{agent_names[0]} Action", fontsize=12)
        ax.set_ylabel(f"{agent_names[1]} Action", fontsize=12)
        
        agent_c_action = None
        agent_r_action = None
        
        if with_overlay and thetas is not None and opponent_thetas is not None:
            agent_c_action = torch.argmax(thetas).item() if torch.is_tensor(thetas) else np.argmax(thetas)
            agent_r_action = torch.argmax(opponent_thetas).item() if torch.is_tensor(opponent_thetas) else np.argmax(opponent_thetas)
            
            row_highlight = plt.Rectangle((-0.5, agent_r_action-0.5), n_dim, 1, 
                                        fill=True, color='red', alpha=0.2)
            ax.add_patch(row_highlight)
            
            col_highlight = plt.Rectangle((agent_c_action-0.5, -0.5), 1, n_dim, 
                                        fill=True, color='blue', alpha=0.2)
            ax.add_patch(col_highlight)
            
            ax.plot(agent_c_action, agent_r_action, 'ko', markersize=10 if not is_high_dim else 5)
        
        if not is_high_dim:
            font_size = max(4, min(12, 120 / n_dim))
            if torch.is_tensor(payoff_matrix):
                vals = payoff_matrix.cpu().numpy()
            else:
                vals = M_values
                
            for i in range(n_dim):
                for j in range(n_dim):
                    val = vals[i, j]
                    text = "1" if val == 1 else ("0" if val == 0 else f"{val:.3f}")
                    
                    if with_overlay and i == agent_r_action and j == agent_c_action:
                        ax.text(j, i, text, va='center', ha='center', fontsize=font_size,
                                fontweight='bold', color='white')
                    else:
                        ax.text(j, i, text, va='center', ha='center', fontsize=font_size)
            
            ax.grid(True)
            ax.set_xticks(np.arange(-0.5, n_dim-0.5))
            ax.set_yticks(np.arange(-0.5, n_dim-0.5))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            tick_interval = max(1, n_dim // 10)
            ax.set_xticks(np.arange(0, n_dim, tick_interval))
            ax.set_yticks(np.arange(0, n_dim, tick_interval))
            ax.set_xticklabels(np.arange(0, n_dim, tick_interval))
            ax.set_yticklabels(np.arange(0, n_dim, tick_interval))
        
        if too_many_values:
            cmap = LinearSegmentedColormap.from_list(
                'reward_cmap', 
                [[1, 1, 1]] + [colors[val] for val in unique_values], 
                N=256
            )
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            
            min_val = min(unique_values) if unique_values else 0
            max_val = max(unique_values) if unique_values else 1
            norm = plt.Normalize(vmin=0, vmax=max_val)
            
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cbar.set_label('Reward Value', fontsize=10)
            
            if with_overlay and agent_c_action is not None and agent_r_action is not None:
                legend_elements = [
                    Patch(facecolor='blue', alpha=0.2, label=f'{agent_names[0]} Action: {agent_c_action}'),
                    Patch(facecolor='red', alpha=0.2, label=f'{agent_names[1]} Action: {agent_r_action}')
                ]
                
                if agent_c_action < n_dim / 2 and agent_r_action < n_dim / 2:
                    legend_loc = 'lower right'
                elif agent_c_action >= n_dim / 2 and agent_r_action < n_dim / 2:
                    legend_loc = 'lower left'
                elif agent_c_action < n_dim / 2 and agent_r_action >= n_dim / 2:
                    legend_loc = 'upper right'
                else:
                    legend_loc = 'upper left'
                
                ax.legend(handles=legend_elements, loc=legend_loc, fontsize=10)
        else:
            legend_elements = [Patch(facecolor=colors[val], label=f'Reward = {val:.3f}') for val in unique_values]
            legend_elements.append(Patch(facecolor=[1, 1, 1], label='Reward = 0'))
            
            if with_overlay and agent_c_action is not None and agent_r_action is not None:
                legend_elements.append(Patch(facecolor='blue', alpha=0.2, label=f'{agent_names[0]} Action: {agent_c_action}'))
                legend_elements.append(Patch(facecolor='red', alpha=0.2, label=f'{agent_names[1]} Action: {agent_r_action}'))
                
                if agent_c_action < n_dim / 2 and agent_r_action < n_dim / 2:
                    legend_loc = 'lower right'
                elif agent_c_action >= n_dim / 2 and agent_r_action < n_dim / 2:
                    legend_loc = 'lower left'
                elif agent_c_action < n_dim / 2 and agent_r_action >= n_dim / 2:
                    legend_loc = 'upper right'
                else:
                    legend_loc = 'upper left'
            else:
                legend_loc = 'upper right'
                
            ax.legend(handles=legend_elements, loc=legend_loc, fontsize=10)

    @staticmethod
    def render_matrix_game(payoff_matrix, n_dim, thetas=None, opponent_thetas=None, layout=None, show_layout_only=False, agents=None):
        if torch.is_tensor(payoff_matrix) and payoff_matrix.device.type != 'cpu':
            payoff_matrix = payoff_matrix.cpu()
            
        if torch.is_tensor(thetas) and thetas.device.type != 'cpu':
            thetas = thetas.cpu()
            
        if torch.is_tensor(opponent_thetas) and opponent_thetas.device.type != 'cpu':
            opponent_thetas = opponent_thetas.cpu()
            
        fig_size = min(8, max(5, 5 * (10 / n_dim)))
        fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
        CMGVisualizer.plot_discrete_payoff(ax, payoff_matrix, n_dim, thetas, opponent_thetas, layout, with_overlay=not show_layout_only, agents=agents)
        plt.tight_layout()
        plt.show()
