import numpy as np
import torch
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, env=None):
        self.env = env
        self.n_dim = env.n_dim if env else None

    def set_env(self, env):
        self.env = env
        self.n_dim = env.n_dim

    def visualize_training_progress(self, SP_rewards, XP_rewards=None, n_seeds=None, labels=None, title="Training Progress"):
        if isinstance(SP_rewards, list) and isinstance(SP_rewards[0], list):
            SP_rewards = np.array(SP_rewards)
            
        if XP_rewards is not None and isinstance(XP_rewards, list) and isinstance(XP_rewards[0], list):
            XP_rewards = np.array(XP_rewards)
        
        if isinstance(SP_rewards, np.ndarray) and SP_rewards.ndim == 2:
            SP_rewards = [SP_rewards]
            if XP_rewards is not None and XP_rewards.ndim == 2:
                XP_rewards = [XP_rewards]
        
        if labels is None:
            labels = [f"Series {i+1}" for i in range(len(SP_rewards))]
            
        n_episodes = SP_rewards[0].shape[1]
        x = np.arange(n_episodes)
        
        fig = plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.title("Self-Play Return")
        
        for i, rewards in enumerate(SP_rewards):
            m = np.mean(rewards, axis=0)
            
            if n_seeds is not None:
                s = np.std(rewards, axis=0) / np.sqrt(n_seeds)
                plt.plot(x, m, label=labels[i])
                plt.fill_between(x, m+s, m-s, alpha=0.2)
            else:
                plt.plot(x, m, label=labels[i])
        
        plt.ylabel("Avg. return")
        plt.xlabel("Training Steps")
        plt.ylim(-0.1, 1.1)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if XP_rewards is not None:
            plt.subplot(1, 2, 2)
            plt.title("Cross-Play Return")
            
            for i, rewards in enumerate(XP_rewards):
                m = np.mean(rewards, axis=0)
                
                if n_seeds is not None:
                    s = np.std(rewards, axis=0) / np.sqrt(n_seeds)
                    plt.plot(x, m, label=labels[i])
                    plt.fill_between(x, m+s, m-s, alpha=0.2)
                else:
                    plt.plot(x, m, label=labels[i])
            
            plt.ylabel("Avg. return")
            plt.xlabel("Training Steps")
            plt.ylim(-0.1, 1.1)
            plt.legend(loc="lower right")
            plt.grid(True)
        
        plt.tight_layout()
        return fig

    #=== Policy & Rewards ===
    def visualize_joint_policy(self, policy_C, policy_R, agent_names=None):
        if hasattr(policy_C, 'get_policy') and callable(policy_C.get_policy):
            policy_C_dist, _ = policy_C.get_policy()
            policy_C = policy_C_dist
        if hasattr(policy_R, 'get_policy') and callable(policy_R.get_policy):
            policy_R_dist, _ = policy_R.get_policy()
            policy_R = policy_R_dist
        
        policy_C = torch.as_tensor(policy_C).view(-1)
        policy_R = torch.as_tensor(policy_R).view(-1)
        
        if policy_C.sum() > 0:
            policy_C = policy_C / policy_C.sum()
        if policy_R.sum() > 0:
            policy_R = policy_R / policy_R.sum()
        
        joint_policy = torch.outer(policy_R, policy_C)
        
        total_prob = joint_policy.sum().item()
        if abs(total_prob - 1.0) > 1e-6:
            print(f"Warning: Joint policy sum ({total_prob}) deviates from 1.0")
        
        fig = plt.figure(figsize=(6, 6))
        
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1.5], wspace=0, left=0, right=0.85, top=0.9, bottom=0.12)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        
        im = ax.imshow(joint_policy.detach().numpy(), origin="upper", cmap="viridis")
        
        if agent_names:
            col_agent = agent_names[0].replace('agent_', 'Agent ').replace('_C', ' (Column)')
            row_agent = agent_names[1].replace('agent_', 'Agent ').replace('_R', ' (Row)')
        else:
            col_agent = "Agent C (Column)"
            row_agent = "Agent R (Row)"
            
        ax.set_xlabel(f"{col_agent} Action", fontsize=15)
        ax.set_ylabel(f"{row_agent} Action", fontsize=15)
        
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Joint Probability", fontsize=15)
        
        tick_interval = max(1, self.n_dim // 10)
        ax.set_xticks(np.arange(0, self.n_dim, tick_interval))
        ax.set_yticks(np.arange(0, self.n_dim, tick_interval))
        ax.set_xticklabels(np.arange(0, self.n_dim, tick_interval), fontsize=15)
        ax.set_yticklabels(np.arange(0, self.n_dim, tick_interval), fontsize=15)
        
        cax.tick_params(labelsize=15)
        plt.tight_layout()
        
        return fig, ax

    def visualize_pool(self, policy_input, agent_names=None, aggregation='avg', normalize=False):
        figsize = (8, 6)
        font_size = 18
        tick_interval = None
        width_ratios = [20, 1.0]
        margins = (0.10, 0.85, 0.95, 0.12)

        if hasattr(policy_input, 'get_stacked_policies'):
            joint_policies = policy_input.get_stacked_policies()
            n_policies = len(policy_input)
        else:
            joint_policies = torch.as_tensor(policy_input)
            n_policies = joint_policies.shape[0]
        
        joint_policies = joint_policies.cpu()
        
        if isinstance(policy_input, tuple):
            raise ValueError("visualize_pool expects a PolicyPool instance or stacked joint policies. "
                            "Please pass policy_pool.get_stacked_policies() instead.")
        
        aggregated_policy = torch.zeros_like(joint_policies[0], device='cpu')
        
        if aggregation == 'max':
            for i in range(n_policies):
                aggregated_policy = torch.maximum(aggregated_policy, joint_policies[i])
        elif aggregation == 'sum':
            for i in range(n_policies):
                aggregated_policy += joint_policies[i]
        elif aggregation == 'avg':
            for i in range(n_policies):
                aggregated_policy += joint_policies[i]
            if n_policies > 0:
                aggregated_policy = aggregated_policy / n_policies
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        if normalize and aggregated_policy.sum() > 0:
            aggregated_policy = aggregated_policy / aggregated_policy.sum()
        
        fig = plt.figure(figsize=figsize)
        
        left, right, top, bottom = margins
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0, 
                            left=left, right=right, top=top, bottom=bottom)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        
        im = ax.imshow(aggregated_policy.detach().cpu().numpy(), origin="upper", cmap="viridis") 

        if agent_names:
            col_agent = agent_names[0].replace('agent_', 'Agent ').replace('_C', ' (Column)')
            row_agent = agent_names[1].replace('agent_', 'Agent ').replace('_R', ' (Row)')
        else:
            col_agent = "Agent C (Column)"
            row_agent = "Agent R (Row)"

        ax.set_xlabel(f"{col_agent} Action", fontsize=font_size)
        ax.set_ylabel(f"{row_agent} Action", fontsize=font_size)
        
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Coverage', fontsize=font_size)
        
        if tick_interval is None:
            tick_interval = max(1, self.n_dim // 10)
            
        ax.set_xticks(np.arange(0, self.n_dim, tick_interval))
        ax.set_yticks(np.arange(0, self.n_dim, tick_interval))
        ax.set_xticklabels(np.arange(0, self.n_dim, tick_interval), fontsize=font_size)
        ax.set_yticklabels(np.arange(0, self.n_dim, tick_interval), fontsize=font_size)
        
        cax.tick_params(labelsize=font_size)
        plt.tight_layout()
        return fig
    
    def visualize_policy_distributions(self, p_r_initial, p_c_initial, p_r_final, p_c_final, train_steps):
        p_r_initial = p_r_initial.cpu().detach().numpy() if torch.is_tensor(p_r_initial) else p_r_initial
        p_c_initial = p_c_initial.cpu().detach().numpy() if torch.is_tensor(p_c_initial) else p_c_initial
        p_r_final = p_r_final.cpu().detach().numpy() if torch.is_tensor(p_r_final) else p_r_final
        p_c_final = p_c_final.cpu().detach().numpy() if torch.is_tensor(p_c_final) else p_c_final
        
        n_dim = len(p_r_initial)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        
        axes[0, 0].bar(range(n_dim), p_r_initial)
        axes[0, 0].set_title("Initial Row Policy", fontsize=15)
        axes[0, 0].set_xlabel("Action", fontsize=15)
        axes[0, 0].set_ylabel("Probability", fontsize=15)
        axes[0, 0].tick_params(labelsize=15)
        
        axes[0, 1].bar(range(n_dim), p_c_initial)
        axes[0, 1].set_title("Initial Column Policy", fontsize=15)
        axes[0, 1].set_xlabel("Action", fontsize=15)
        axes[0, 1].set_ylabel("Probability", fontsize=15)
        axes[0, 1].tick_params(labelsize=15)
        
        axes[1, 0].bar(range(n_dim), p_r_final)
        axes[1, 0].set_title("Final Row Policy", fontsize=15)
        axes[1, 0].set_xlabel("Action", fontsize=15)
        axes[1, 0].set_ylabel("Probability", fontsize=15)
        axes[1, 0].tick_params(labelsize=15)
        
        axes[1, 1].bar(range(n_dim), p_c_final)
        axes[1, 1].set_title("Final Column Policy", fontsize=15)
        axes[1, 1].set_xlabel("Action", fontsize=15)
        axes[1, 1].set_ylabel("Probability", fontsize=15)
        axes[1, 1].tick_params(labelsize=15)
        
        plt.tight_layout()
            
        return fig

    #=== Training ===
    def visualize_training_results(self, returns, cross_play=None, show_individual=True, show_mean=True):
        if returns is None or len(returns) == 0:
            print("No training results available to visualize.")
            return None
        
        figures = []
        window_size = max(5, returns.shape[1] // 20)
        
        fig1 = plt.figure(figsize=(10, 6))
        
        if show_individual:
            for i in range(returns.shape[0]):
                if returns.shape[1] > window_size:
                    smoothed = np.convolve(returns[i], np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size, returns.shape[1] + 1), smoothed, alpha=0.5, label=f'Policy {i+1}')
                else:
                    plt.plot(returns[i], alpha=0.5, label=f'Policy {i+1}')
                
        if show_mean:
            mean_returns = np.mean(returns, axis=0)
            if returns.shape[1] > window_size:
                smoothed_mean = np.convolve(mean_returns, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size, returns.shape[1] + 1), smoothed_mean, 'k', linewidth=2, label='Mean')
            else:
                plt.plot(mean_returns, 'k', linewidth=2, label='Mean')
            
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.title('Training Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures.append(fig1)
        
        if cross_play is not None:
            fig2 = plt.figure(figsize=(10, 6))
            
            if show_individual:
                for i in range(cross_play.shape[0]):
                    if cross_play.shape[1] > window_size:
                        smoothed = np.convolve(cross_play[i], np.ones(window_size)/window_size, mode='valid')
                        plt.plot(range(window_size, cross_play.shape[1] + 1), smoothed, alpha=0.3)
                    else:
                        plt.plot(cross_play[i], alpha=0.3)
                    
            if show_mean:
                mean_cross_play = np.mean(cross_play, axis=0)
                if cross_play.shape[1] > window_size:
                    smoothed_mean = np.convolve(mean_cross_play, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size, cross_play.shape[1] + 1), smoothed_mean, 'r', linewidth=2, label='Mean Cross-Play')
                else:
                    plt.plot(mean_cross_play, 'r', linewidth=2, label='Mean Cross-Play')
                
            plt.xlabel('Training Steps')
            plt.ylabel('Cross-Play Reward')
            plt.title('Cross-Play Returns')
            plt.grid(True, alpha=0.3)
            plt.legend()

            figures.append(fig2)
        
        return tuple(figures)

    def visualize_multi_seed_results(self, multi_seed_results):
        if multi_seed_results is None or 'returns' not in multi_seed_results:
            print("No multi-seed training results available to visualize.")
            return None
        
        figures = []
        returns = multi_seed_results['returns']
        
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        
        window_size = max(5, mean_returns.shape[1] // 20)
        
        fig1 = plt.figure(figsize=(10, 6))
        
        for i in range(mean_returns.shape[0]):
            if mean_returns.shape[1] > window_size:
                smoothed_mean = np.convolve(mean_returns[i], np.ones(window_size)/window_size, mode='valid')
                smoothed_std_up = np.convolve(mean_returns[i] + std_returns[i], np.ones(window_size)/window_size, mode='valid')
                smoothed_std_down = np.convolve(mean_returns[i] - std_returns[i], np.ones(window_size)/window_size, mode='valid')
                
                x_range = range(window_size, mean_returns.shape[1] + 1)
                plt.plot(x_range, smoothed_mean, label=f'Policy {i+1}')
                plt.fill_between(
                    x_range,
                    smoothed_std_down,
                    smoothed_std_up,
                    alpha=0.2
                )
            else:
                plt.plot(mean_returns[i], label=f'Policy {i+1}')
                plt.fill_between(
                    range(mean_returns.shape[1]),
                    mean_returns[i] - std_returns[i],
                    mean_returns[i] + std_returns[i],
                    alpha=0.2
                )
        
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Reward')
        plt.title(f'Mean Returns Across {len(multi_seed_results["seeds"])} Seeds')
        plt.legend()
        plt.grid(True, alpha=0.3)

        figures.append(fig1)
        
        if 'cross_play' in multi_seed_results:
            cross_play = multi_seed_results['cross_play']
            mean_cross_play = np.mean(cross_play, axis=0)
            mean_cross_play_overall = np.mean(mean_cross_play, axis=0)
            std_cross_play_overall = np.std(np.mean(cross_play, axis=1), axis=0)
            
            fig2 = plt.figure(figsize=(10, 6))
            
            if len(mean_cross_play_overall) > window_size:
                smoothed_mean = np.convolve(mean_cross_play_overall, np.ones(window_size)/window_size, mode='valid')
                smoothed_std_up = np.convolve(mean_cross_play_overall + std_cross_play_overall, np.ones(window_size)/window_size, mode='valid')
                smoothed_std_down = np.convolve(mean_cross_play_overall - std_cross_play_overall, np.ones(window_size)/window_size, mode='valid')
                
                x_range = range(window_size, len(mean_cross_play_overall) + 1)
                plt.plot(x_range, smoothed_mean, 'r', linewidth=2)
                plt.fill_between(
                    x_range,
                    smoothed_std_down,
                    smoothed_std_up,
                    alpha=0.2,
                    color='r'
                )
            else:
                plt.plot(mean_cross_play_overall, 'r', linewidth=2)
                plt.fill_between(
                    range(len(mean_cross_play_overall)),
                    mean_cross_play_overall - std_cross_play_overall,
                    mean_cross_play_overall + std_cross_play_overall,
                    alpha=0.2,
                    color='r'
                )
                
            plt.xlabel('Training Steps')
            plt.ylabel('Cross-Play Reward')
            plt.title(f'Mean Cross-Play Returns Across {len(multi_seed_results["seeds"])} Seeds')
            plt.grid(True, alpha=0.3)

            figures.append(fig2)
        
        return tuple(figures)
