from tqdm import tqdm
import random
import os
import numpy as np
import torch
import argparse
import time
from transformer import DecisionTransformer
from dt_dataset import build_dataset, traj_buffer
from human_aware_rl.data_dir import DATA_DIR
import matplotlib.pyplot as plt
import pickle
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

class Trainer:
    def __init__(self, train_data, test_data, model, optimizer, batch_size, loss_fn, scheduler=None, max_len=10, action_dim=6):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.diagnostics = dict()
        self.max_len = max_len
        self.start_time = time.time()
        self.train_buf = train_data
        self.test_buf = test_data
        self.action_dim = action_dim
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_self_losses = []
        train_teammate_losses = []
        train_rtg_losses = []
        logs = dict()
        train_start = time.time()
        self.model.train()
        for _ in tqdm(range(num_steps)):
            self_loss, teammate_loss, rtg_loss = self.train_step()
            train_self_losses.append(self_loss)
            train_teammate_losses.append(teammate_loss)
            train_rtg_losses.append(rtg_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        logs['time/training'] = time.time() - train_start
        logs['training/self_loss_mean'] = np.mean(train_self_losses)
        logs['training/self_loss_std'] = np.std(train_self_losses)
        logs['training/teammate_loss_mean'] = np.mean(train_teammate_losses)
        logs['training/teammate_loss_std'] = np.std(train_teammate_losses)
        logs['training/rtg_loss_mean'] = np.mean(train_rtg_losses)
        logs['training/rtg_loss_std'] = np.std(train_rtg_losses)
        

        eval_start = time.time()
        self.model.eval()
        test_self_loss, test_teammate_loss, test_rtg_loss = self.test_step()

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['evaluation/self_loss'] = test_self_loss
        logs['evaluation/teammate_loss'] = test_teammate_loss
        logs['evaluation/rtg_loss'] = test_rtg_loss

        return logs, train_self_losses, train_teammate_losses, train_rtg_losses, test_self_loss, test_teammate_loss, test_rtg_loss

    def train_step(self):
        data = self.train_buf.get_batch(batch_size=self.batch_size)
        states = data['ep_states'] # (batchsize, len)
        action0s = data['ep_actions_0']
        action1s = data['ep_actions_1']
        timesteps = data['ep_timesteps']
        rtgs = data['ep_rtgs']
        mask = data['mask']
        mask = np.array(mask)
        action0_preds, action1_preds, rtg_preds = self.model(states, action0s, action1s, rtgs, timesteps, attention_mask=mask)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)
        data_mask = torch.nonzero(mask.reshape(-1)).squeeze()
        action0_gt = torch.tensor(action0s, dtype=torch.long).reshape(-1).to(self.device)
        action1_gt = torch.tensor(action1s, dtype=torch.long).reshape(-1).to(self.device)
        rtg_gt = torch.tensor(rtgs, dtype=torch.float).reshape(-1).to(self.device)
        action0_gt = action0_gt[data_mask]
        action1_gt = action1_gt[data_mask]
        rtg_gt = rtg_gt[data_mask]
        action0_preds = action0_preds.reshape(-1, self.action_dim)[data_mask]
        action1_preds = action1_preds.reshape(-1, self.action_dim)[data_mask]    
        rtg_preds = rtg_preds.reshape(-1)[data_mask]
        # print(action0_preds, action0_gt)
        self_loss = self.loss_fn(action0_preds, action0_gt)
        teammate_loss = self.loss_fn(action1_preds, action1_gt)
        rtg_loss = torch.nn.MSELoss()(rtg_preds, rtg_gt)
        loss = 30 * (self_loss + teammate_loss) + rtg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self_loss.detach().cpu().item(), teammate_loss.detach().cpu().item(), rtg_loss.detach().cpu().item()
    
    def test_step(self):
        data = self.test_buf.get_batch(batch_size=self.batch_size)
        states = data['ep_states'] # (batchsize, len)
        action0s = data['ep_actions_0']
        action1s = data['ep_actions_1']
        timesteps = data['ep_timesteps']
        rtgs = data['ep_rtgs']
        mask = data['mask']
        mask = np.array(mask)
        action0_preds, action1_preds, rtg_preds = self.model(states, action0s, action1s, rtgs, timesteps, attention_mask=mask)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)
        data_mask = torch.nonzero(mask.reshape(-1)).squeeze()
        action0_gt = torch.tensor(action0s, dtype=torch.long).reshape(-1).to(self.device)
        action1_gt = torch.tensor(action1s, dtype=torch.long).reshape(-1).to(self.device)
        action0_gt = action0_gt[data_mask]
        action1_gt = action1_gt[data_mask]
        action0_preds = action0_preds.reshape(-1, self.action_dim)[data_mask]
        action1_preds = action1_preds.reshape(-1, self.action_dim)[data_mask]
        rtg_gt = torch.tensor(rtgs, dtype=torch.float).reshape(-1).to(self.device)
        rtg_gt = rtg_gt[data_mask]
        rtg_preds = rtg_preds.reshape(-1)[data_mask]
        actions0 = torch.softmax(action0_preds, dim=-1)
        actions1 = torch.softmax(action1_preds, dim=-1)
        print(f"Reference actions0: {actions0[5:10]} v.s. {action0_gt[5:10]}")
        actions0 = torch.argmax(actions0, dim=-1)
        print(f"Accuracy for a0: {torch.sum(actions0 == action0_gt).float() / len(actions0)}")
        print(f"Reference actions1: {actions1[5:10]} v.s. {action1_gt[5:10]}")
        actions1 = torch.argmax(actions1, dim=-1)
        print(f"Accuracy for a1: {torch.sum(actions1 == action1_gt).float() / len(actions1)}")
        print(f"Reference rtgs: {rtg_preds[5:10]} v.s. {rtg_gt[5:10]}")
        self_loss = self.loss_fn(action0_preds, action0_gt)
        teammate_loss = self.loss_fn(action1_preds, action1_gt)
        rtg_loss = torch.nn.MSELoss()(rtg_preds, rtg_gt)
        return self_loss.detach().cpu().item(), teammate_loss.detach().cpu().item(), rtg_loss.detach().cpu().item()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    args = parser.parse_args()

    action_dim = len(Action.ALL_ACTIONS)
    train_buf = traj_buffer(data_path=os.path.join(DATA_DIR, "train.pickle"), max_length=args.max_len, pad=action_dim)
    test_buf = traj_buffer(data_path=os.path.join(DATA_DIR, "test.pickle"), max_length=args.max_len, pad=action_dim)
    example = train_buf.get_batch(batch_size=1)
    state_dim = example['ep_states'][0].shape[-1]
    print(action_dim, state_dim)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, 
                                hidden_size=128, max_length=args.max_len,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = torch.from_numpy(np.array([5.503300330033003, 6.5246618106139438, 4.719830200121286, 3.511128945960407, 1.0, 10.558479943701619])).float().to(model.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / args.warmup_steps, 1.0)
    )
    trainer = Trainer(train_data=train_buf, test_data=test_buf, model=model, optimizer=optimizer, batch_size=args.batch_size, loss_fn=loss_fn, scheduler=scheduler, max_len=args.max_len, action_dim=action_dim)
    train_self_losses = []
    train_teammate_losses = []
    train_rtg_losses = []
    test_self_losses = []
    test_teammate_losses = []
    test_rtg_losses = []
    min_loss = 1e5
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        logs, train_self_loss, train_teammate_loss, train_rtg_loss, test_self_loss, test_teammate_loss, test_rtg_loss = trainer.train_iteration(num_steps=2000)
        train_self_loss = np.mean(train_self_loss)
        train_teammate_loss = np.mean(train_teammate_loss)
        train_rtg_loss = np.mean(train_rtg_loss)
        train_self_losses.append(train_self_loss)
        train_teammate_losses.append(train_teammate_loss)
        train_rtg_losses.append(train_rtg_loss)
        test_self_losses.append(test_self_loss)
        test_teammate_losses.append(test_teammate_loss)
        test_rtg_losses.append(test_rtg_loss)
        if test_self_loss + test_teammate_loss < min_loss:
            min_loss = test_self_loss + test_teammate_loss
            print("Saving model...")
            if not os.path.exists("./model"):
                os.mkdir("./model")
            torch.save(model.state_dict(), f"./model/dt_{args.max_len}.pth")
        print(logs)
    x = [i for i in range(len(train_self_losses))]
    fig = plt.figure()
    plt.plot(x, train_self_losses)
    plt.savefig(f"train_self_loss_{args.max_len}.png")
    fig.clear()
    plt.plot(x, train_teammate_losses)
    plt.savefig(f"train_teammate_loss_{args.max_len}.png")
    fig.clear()
    plt.plot(x, test_self_losses)
    plt.savefig(f"test_self_loss_{args.max_len}.png")
    fig.clear()
    plt.plot(x, test_teammate_losses)
    plt.savefig(f"test_teammate_loss_{args.max_len}.png")
    fig.clear()