# simulator.py
import numpy as np
import matplotlib.pyplot as plt
from utils import save_plot

class Scheduler:
    def __init__(self, num_users=5, num_rb=20):
        self.num_users = num_users
        self.num_rb = num_rb
        self.user_rates = np.zeros(num_users)
    
    def round_robin(self):
        """Simple Round-Robin RB allocation"""
        allocation = np.zeros((self.num_users, self.num_rb))
        for rb in range(self.num_rb):
            user = rb % self.num_users
            allocation[user, rb] = 1
        return allocation
    
    def proportional_fair(self):
        """Proportional Fair scheduling (simplified)"""
        allocation = np.zeros((self.num_users, self.num_rb))
        user_priority = 1 / (self.user_rates + 1e-6)
        for rb in range(self.num_rb):
            user = np.argmax(user_priority)
            allocation[user, rb] = 1
            self.user_rates[user] += 1
            user_priority = 1 / (self.user_rates + 1e-6)
        return allocation

def plot_allocation(allocation, title="Resource Allocation", save_path=None, show=True):
    plt.figure(figsize=(8, 4))
    plt.imshow(allocation, cmap='Blues', aspect='auto')
    plt.xlabel("Resource Blocks")
    plt.ylabel("Users")
    plt.title(title)
    plt.colorbar(label="Allocated")
    if save_path:
        save_plot(save_path)
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    scheduler = Scheduler(num_users=5, num_rb=20)
    
    rr_alloc = scheduler.round_robin()
    plot_allocation(rr_alloc, title="Round-Robin Allocation", save_path="plots/rr_allocation.png")
    
    pf_alloc = scheduler.proportional_fair()
    plot_allocation(pf_alloc, title="Proportional Fair Allocation", save_path="plots/pf_allocation.png")
