import torch
from time import time
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import os
import traceback
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Autonomous Driving Scenario Reconstruction Dataset Compression Using the PPO Algorithm
# Create directory for Actor-Critic logs if it doesn't exist
ac_log_dir = './ac_logs'
os.makedirs(ac_log_dir, exist_ok=True)

# Create error log directory if it doesn't exist
error_log_dir = './error_logs'
os.makedirs(error_log_dir, exist_ok=True)

# Initialize Actor-Critic log files
reward_log_file = os.path.join(ac_log_dir, 'ac_rewards.txt')
dataset_size_log_file = os.path.join(ac_log_dir, 'ac_dataset_sizes.txt')

# Write headers to log files if they are new
if not os.path.exists(reward_log_file):
    with open(reward_log_file, 'w') as f:
        f.write("epoch,action,reward,iou_improvement,loss_reduction,efficiency_penalty\n")

if not os.path.exists(dataset_size_log_file):
    with open(dataset_size_log_file, 'w') as f:
        f.write("epoch,dataset_size_pct,num_samples_to_keep,total_train_samples\n")


def log_error(error_message, error_context=""):
    """
    Log error message to a file with timestamp and context information
    """
    timestamp = time()
    datetime_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    error_filename = f"error_{datetime_str}_{int(timestamp)}.txt"
    error_filepath = os.path.join(error_log_dir, error_filename)

    # Create detailed error message
    full_error_message = f"Error occurred at {datetime_str} (timestamp: {timestamp:.0f}):\n"
    full_error_message += f"Error context: {error_context}\n"
    full_error_message += f"{error_message}\n"
    full_error_message += "\nTraceback:\n"
    full_error_message += traceback.format_exc()

    # Write error information to file
    with open(error_filepath, 'w') as f:
        f.write(full_error_message)

    return error_filepath


# Import required modules
from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 ent_coeff=0.01, eps_clip=0.2, update_epochs=10, batch_size=64):
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim  # Store action dimension

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        # Initialize policy_old with policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.value.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        try:
            state = torch.FloatTensor(state).unsqueeze(0)

            # Forward pass through policy network
            with torch.no_grad():  # No need for gradients during action selection
                action_probs = self.policy_old(state)

            # Debug: Print action probabilities for debugging
            # print(f"Action probs before checks: {action_probs}")

            # Check for invalid values in action probabilities
            # First, check if any probability is NaN or infinite
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"Warning: Invalid values (NaN/inf) detected in action probabilities. Random action selected.")
                action = np.random.choice(self.action_dim)
                return action, 0.0

            # Check if all probabilities are zero
            if (action_probs == 0).all():
                print(f"Warning: All action probabilities are zero. Random action selected.")
                action = np.random.choice(self.action_dim)
                return action, 0.0

            # Check if probabilities sum to a reasonable value (close to 1)
            prob_sum = action_probs.sum().item()
            if abs(prob_sum - 1.0) > 0.1:  # More than 10% deviation
                print(f"Warning: Action probabilities sum to {prob_sum:.6f} (expected ~1.0). Random action selected.")
                action = np.random.choice(self.action_dim)
                return action, 0.0

            # Check if any probability is negative
            if (action_probs < 0).any():
                print(f"Warning: Negative probabilities detected. Random action selected.")
                action = np.random.choice(self.action_dim)
                return action, 0.0

            # Final safety check: if any probability is outside [0, 1]
            if (action_probs < 0).any() or (action_probs > 1).any():
                print(f"Warning: Probabilities outside valid range [0, 1]. Random action selected.")
                action = np.random.choice(self.action_dim)
                return action, 0.0

            # If all checks passed, proceed with normal action selection
            # But wrap in try-except as a final safety net
            try:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item()
            except Exception as e:
                print(f"Warning: Failed to create Categorical distribution. Error: {e}. Random action selected.")
                print(f"Action probs: {action_probs}")
                action = np.random.choice(self.action_dim)
                return action, 0.0

        except Exception as e:
            error_filepath = log_error(f"Error in select_action: {str(e)}",
                                       f"State: {state}")
            print(f"Error in select_action! Details saved to: {error_filepath}")
            print(f"Error: {e}")

            # Fallback to random action selection
            action = np.random.choice(self.action_dim)
            return action, 0.0

    def update(self, memory):
        try:
            # Convert memory to tensors
            states = torch.FloatTensor(memory.states)
            actions = torch.LongTensor(memory.actions).unsqueeze(1)
            log_probs = torch.FloatTensor(memory.log_probs).unsqueeze(1)
            rewards = torch.FloatTensor(memory.rewards).unsqueeze(1)
            next_states = torch.FloatTensor(memory.next_states)
            dones = torch.FloatTensor(memory.dones).unsqueeze(1)

            # Compute discounted rewards and advantages
            returns = []
            advantages = []
            discounted_reward = 0
            advantage = 0

            # Get value estimates for all states
            values = self.value(states)
            next_values = self.value(next_states).detach()

            # Compute advantages using GAE (Generalized Advantage Estimation)
            for i in reversed(range(len(rewards))):
                # Update discounted reward
                discounted_reward = rewards[i] + self.gamma * discounted_reward * (1 - dones[i])
                returns.insert(0, discounted_reward)

                # Update advantage
                td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
                advantage = td_error + self.gamma * 0.95 * advantage * (1 - dones[i])  # 0.95 is lambda for GAE
                advantages.insert(0, advantage)

            # Convert to tensors
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update policy for multiple epochs
            for _ in range(self.update_epochs):
                # Create mini-batches
                indices = torch.randperm(len(states))
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]

                    # Get batch data
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_log_probs = log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Get current policy distribution
                    current_action_probs = self.policy(batch_states)

                    # Check for invalid values in action probabilities
                    if torch.isnan(current_action_probs).any() or torch.isinf(current_action_probs).any():
                        # Replace invalid values with uniform distribution
                        batch_size = current_action_probs.shape[0]
                        current_action_probs = torch.ones(batch_size, self.action_dim) / self.action_dim
                        if current_action_probs.is_cuda:
                            current_action_probs = current_action_probs.cuda()
                        print(
                            "Warning: Invalid values detected in action probabilities during update. Using uniform distribution instead.")

                    # Ensure probabilities sum to 1 (numerical stability)
                    current_action_probs = current_action_probs / current_action_probs.sum(dim=1, keepdim=True)

                    # Clip probabilities to avoid underflow/overflow
                    current_action_probs = torch.clamp(current_action_probs, 1e-8, 1.0 - 1e-8)

                    # Final safety check before creating distribution
                    if (current_action_probs < 0).any() or (current_action_probs > 1).any():
                        print(f"Warning: Probabilities outside valid range [0, 1] during update. Clipping...")
                        current_action_probs = torch.clamp(current_action_probs, 0, 1)
                        current_action_probs = current_action_probs / current_action_probs.sum(dim=1, keepdim=True)

                    dist = Categorical(current_action_probs)
                    current_log_probs = dist.log_prob(batch_actions.squeeze(1)).unsqueeze(1)
                    entropy = dist.entropy().mean()

                    # Compute ratio (pi_theta / pi_theta_old)
                    ratio = torch.exp(current_log_probs - batch_log_probs.detach())

                    # Compute surrogate losses
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                    # Compute losses
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self.MseLoss(self.value(batch_states), batch_returns)
                    total_loss = policy_loss + 0.5 * value_loss - self.ent_coeff * entropy

                    # Update networks
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    total_loss.backward()
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()

            # Update old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

        except Exception as e:
            error_filepath = log_error(f"Error in PPO update: {str(e)}",
                                       f"Memory size: {len(memory.states)}")
            print(f"Error in PPO update! Details saved to: {error_filepath}")
            print(f"Error: {e}")
            traceback.print_exc()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, log_prob, reward, next_state, done):
        try:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        except Exception as e:
            error_filepath = log_error(f"Error adding to memory: {str(e)}",
                                       f"State type: {type(state)}, Action type: {type(action)}")
            print(f"Error adding to memory! Details saved to: {error_filepath}")
            print(f"Error: {e}")

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []


def train(version,
          dataroot='/data/nuscenes',
          nepochs=10000,
          gpuid=1,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=4,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,

          # PPO parameters
          ppo_lr_actor=3e-4,
          ppo_lr_critic=3e-4,
          ppo_gamma=0.99,
          ppo_ent_coeff=0.01,
          ppo_eps_clip=0.2,
          ppo_update_epochs=10,
          ppo_batch_size=64,
          ppo_memory_size=1000,
          ):
    try:
        # Create log directory if it doesn't exist
        os.makedirs(logdir, exist_ok=True)

        grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        data_aug_conf = {
            'resize_lim': resize_lim,
            'final_dim': final_dim,
            'rot_lim': rot_lim,
            'H': H, 'W': W,
            'rand_flip': rand_flip,
            'bot_pct_lim': bot_pct_lim,
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': ncams,
        }
        # Use real data loaders
        trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                              grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                              parser_name='segmentationdata')

        # Calculate the total number of training samples
        total_train_samples = len(trainloader.dataset)

        # Generate a list of indices
        indices = list(range(total_train_samples))
        # Shuffle the indices randomly
        np.random.shuffle(indices)

        # Create a SubsetRandomSampler for the selected training indices
        train_sampler = SubsetRandomSampler(indices[:int(0.9 * total_train_samples)])

        # Create a new DataLoader with the subset of data (90%)
        trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=bsz, sampler=train_sampler,
                                                  num_workers=nworkers, drop_last=True)

        device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

        model = compile_model(grid_conf, data_aug_conf, outC=1)
        # Load pre-trained model if available
        modelf = './runs/model99000.pt'
        if os.path.exists(modelf):
            model.load_state_dict(torch.load(modelf))
            print(f"Loaded pre-trained model from {modelf}")
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

        writer = SummaryWriter(logdir=logdir)
        val_step = 1000 if version == 'mini' else 10000

        # Initialize PPO agent
        state_dim = 6  # Current epoch, train loss, val loss, train iou, val iou, dataset size
        action_space = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # 70%-100% in 5% increments
        action_dim = len(action_space)  # Number of discrete actions
        ppo = PPO(state_dim, action_dim,
                  lr_actor=ppo_lr_actor, lr_critic=ppo_lr_critic,
                  gamma=ppo_gamma, ent_coeff=ppo_ent_coeff,
                  eps_clip=ppo_eps_clip, update_epochs=ppo_update_epochs,
                  batch_size=ppo_batch_size)

        # Initialize memory for PPO
        memory = Memory()

        # Move PPO networks to device if using GPU
        if gpuid >= 0:
            ppo.policy.cuda(gpuid)
            ppo.policy_old.cuda(gpuid)
            ppo.value.cuda(gpuid)

        model.train()
        counter = 0

        # Initialize metrics
        prev_train_loss = float('inf')
        prev_val_loss = float('inf')
        prev_train_iou = 0.0
        prev_val_iou = 0.0
        current_dataset_size = 0.9  # Start with 90% dataset

        for epoch in range(nepochs):
            try:
                np.random.seed()
                sample_losses = []  # To store loss values for each sample
                batch_ious = []  # To store IoU values for each batch

                # Get action (dataset size percentage) from PPO agent
                # State includes: current epoch progress, previous train loss, previous val loss,
                # previous train iou, previous val iou, current dataset size
                state = [
                    epoch / nepochs,  # Normalized epoch progress
                    prev_train_loss / 10.0,  # Normalized previous train loss
                    prev_val_loss / 10.0,  # Normalized previous val loss
                    prev_train_iou,  # Previous train IoU
                    prev_val_iou,  # Previous val IoU
                    current_dataset_size  # Current dataset size
                ]

                # Select action using PPO
                try:
                    action_idx, log_prob = ppo.select_action(state)
                    # Convert action index to dataset size percentage
                    dataset_size_pct = action_space[action_idx]
                    current_dataset_size = dataset_size_pct
                except Exception as e:
                    # Log the error and use default action
                    error_filepath = log_error(f"Error selecting action: {str(e)}", f"State: {state}")
                    print(f"Error selecting action! Details saved to: {error_filepath}")
                    # Use default action (90% dataset size)
                    action_idx = 4  # Index for 90%
                    dataset_size_pct = action_space[action_idx]
                    current_dataset_size = dataset_size_pct
                    log_prob = 0.0

                print(f"Epoch {epoch}: Selected dataset size: {dataset_size_pct:.2%}")
                writer.add_scalar('dataset/size', dataset_size_pct, epoch)

                for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
                    try:
                        t0 = time()
                        opt.zero_grad()
                        # Use actual model input
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )
                        binimgs = binimgs.to(device)
                        loss = loss_fn(preds, binimgs)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        opt.step()

                        # Track the loss for each sample
                        sample_losses.extend([loss.item()] * len(imgs))

                        # Calculate and track IoU
                        _, _, iou = get_batch_iou(preds, binimgs)
                        batch_ious.append(iou)

                        counter += 1
                        t1 = time()

                        # Open the file in append mode (so it doesn't overwrite previous content)
                        with open("training_log.txt", "a") as log_file:
                            if counter % 10 == 0:
                                # Print the counter and loss to the console
                                print(counter, loss.item())

                                # Write the same information to the log file
                                log_file.write(f"{counter} {loss.item()}\n")

                                # Log the scalar loss value to TensorBoard
                                writer.add_scalar('train/loss', loss, counter)

                        if counter % 50 == 0:
                            writer.add_scalar('train/iou', iou, counter)
                            writer.add_scalar('train/epoch', epoch, counter)
                            writer.add_scalar('train/step_time', t1 - t0, counter)

                        if counter % val_step == 0:
                            val_info = get_val_info(model, valloader, loss_fn, device)
                            print('VAL', val_info)
                            writer.add_scalar('val/loss', val_info['loss'], counter)
                            writer.add_scalar('val/iou', val_info['iou'], counter)

                        if counter % val_step == 0:
                            model.eval()
                            mname = os.path.join(logdir, "model{}.pt".format(counter))
                            print('saving', mname)
                            torch.save(model.state_dict(), mname)
                            model.train()
                    except Exception as e:
                        error_filepath = log_error(f"Error in batch processing: {str(e)}",
                                                   f"Epoch: {epoch}, Batch: {batchi}")
                        print(f"Error in batch processing! Details saved to: {error_filepath}")
                        print(f"Continuing with next batch...")
                        continue

                # Calculate average metrics for this epoch
                avg_train_loss = np.mean(sample_losses) if sample_losses else 0.0
                avg_train_iou = np.mean(batch_ious) if batch_ious else 0.0

                # Get validation metrics
                val_info = get_val_info(model, valloader, loss_fn, device)
                avg_val_loss = val_info['loss']
                avg_val_iou = val_info['iou']

                # Log epoch metrics
                writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
                writer.add_scalar('epoch/train_iou', avg_train_iou, epoch)
                writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
                writer.add_scalar('epoch/val_iou', avg_val_iou, epoch)

                # Calculate reward for PPO
                iou_improvement = avg_val_iou - prev_val_iou
                loss_reduction = prev_val_loss - avg_val_loss
                efficiency_penalty = (dataset_size_pct - 0.7) * 0.1  # Penalize larger dataset sizes

                # Calculate base performance metric
                performance_metric = iou_improvement + loss_reduction

                # Modified reward calculation
                if performance_metric >= 0:
                    # Performance didn't get worse, give positive reward
                    # Use a sigmoid function to map [0, ∞) to (0, 1]
                    reward = 2 * (1 / (1 + np.exp(-performance_metric))) - 1
                else:
                    # Performance got worse, give negative reward
                    # Double penalty for performance degradation
                    reward = performance_metric * 2

                # Add efficiency penalty (this makes larger datasets less favorable)
                reward -= efficiency_penalty

                # Clamp reward to reasonable range
                reward = max(min(reward, 1.0), -1.0)

                print(f"Epoch {epoch}: Reward = {reward:.4f} (IoU improvement: {iou_improvement:.4f}, "
                      f"Loss reduction: {loss_reduction:.4f}, Efficiency penalty: {efficiency_penalty:.4f})")
                writer.add_scalar('ac/reward', reward, epoch)

                # Log reward information to file
                with open(reward_log_file, 'a') as f:
                    f.write(
                        f"{epoch},{action_idx},{reward:.6f},{iou_improvement:.6f},{loss_reduction:.6f},{efficiency_penalty:.6f}\n")

                # Prepare next state
                next_state = [
                    (epoch + 1) / nepochs,  # Normalized epoch progress
                    avg_train_loss / 10.0,  # Normalized train loss
                    avg_val_loss / 10.0,  # Normalized val loss
                    avg_train_iou,  # Train IoU
                    avg_val_iou,  # Val IoU
                    current_dataset_size  # Current dataset size
                ]

                # Add experience to PPO memory
                done = 1 if (epoch + 1) >= nepochs else 0
                memory.add(state, action_idx, log_prob, reward, next_state, done)

                # Update PPO agent if memory is full or at the end of training
                if len(memory.states) >= ppo_memory_size or (epoch + 1) >= nepochs:
                    print(f"Updating PPO agent at epoch {epoch}")
                    ppo.update(memory)
                    memory.clear()  # Clear memory after update

                # Update previous metrics
                prev_train_loss = avg_train_loss
                prev_val_loss = avg_val_loss
                prev_train_iou = avg_train_iou
                prev_val_iou = avg_val_iou

                # Select the samples with the lowest loss values based on the selected dataset size
                num_samples_to_keep = int(dataset_size_pct * total_train_samples)
                sample_loss_indices = np.argsort(sample_losses)[:num_samples_to_keep] if sample_losses else []

                # Log dataset size information to file
                with open(dataset_size_log_file, 'a') as f:
                    f.write(f"{epoch},{dataset_size_pct:.6f},{num_samples_to_keep},{total_train_samples}\n")

                # Create new sampler with selected indices
                if sample_loss_indices.size > 0:
                    train_sampler = SubsetRandomSampler(sample_loss_indices)
                    trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=bsz,
                                                              sampler=train_sampler,
                                                              num_workers=nworkers, drop_last=True)



            except Exception as e:
                error_filepath = log_error(f"Error in epoch processing: {str(e)}", f"Epoch: {epoch}")
                print(f"Error in epoch {epoch}! Details saved to: {error_filepath}")
                print(f"Attempting to continue with next epoch...")
                continue

    except Exception as e:
        error_filepath = log_error(f"Critical error in training: {str(e)}")
        print(f"\n{'=' * 80}")
        print("CRITICAL ERROR OCCURRED! Training terminated.")
        print(f"Error details saved to: {error_filepath}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("=" * 80)
        traceback.print_exc()
        raise


def main():
    # Example usage
    try:
        train(
            version='v1.0-trainval',
            nepochs=100,
            gpuid=-1,  # Use CPU for demonstration
            bsz=2,
            nworkers=0,
            logdir='./runs_demo'
        )
    except Exception as e:
        error_filepath = log_error(f"Error in main function: {str(e)}")
        print(f"\n{'=' * 80}")
        print("ERROR OCCURRED IN MAIN FUNCTION!")
        print(f"Error details saved to: {error_filepath}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("Please check the error log file for detailed traceback.")
        print("=" * 80)
        # Re-raise the exception to maintain original behavior
        raise


if __name__ == "__main__":
    main()
