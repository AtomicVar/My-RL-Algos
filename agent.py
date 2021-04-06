class Agent:
    def simulate(self, rollouts, max_ep_len):
        """
        进行 rollouts 轮的模拟，用于查看训练效果
        """
        total_ret = 0
        total_len = 0
        for rollout in range(rollouts):
            o, ep_ret, ep_len = self.env.reset(), 0, 0
            self.env.render()
            for t in range(max_ep_len):
                a, _ = self.pi(o, requires_grad=False)
                o, r, d, _ = self.env.step(a)
                self.env.render()
                ep_ret += r
                ep_len += 1
                if d:
                    break
            # 打日志
            print(f"Rollout {rollout}: return = {ep_ret}, length = {ep_len}")
            total_ret += ep_ret
            total_len += ep_len
        print(
            f">>> Average return={total_ret / rollouts}, average length = {total_len / rollouts}"
        )
