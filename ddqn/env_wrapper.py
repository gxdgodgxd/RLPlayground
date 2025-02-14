import gym
import numpy as np
import cv2

class CartPoleImageWrapper(gym.Wrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
        # 创建显示窗口
        cv2.namedWindow('CartPole', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CartPole', 600, 400)

    def render_frame(self):
        frame = self.env.render()
        # 显示原始画面
        cv2.imshow('CartPole', frame)
        cv2.waitKey(1)  # 添加短暂延迟，使画面可见
        
        # 处理图像数据
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(frame, axis=-1)

    def reset(self):
        self.env.reset()
        return self.render_frame()

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        obs = self.render_frame()
        return obs, reward, done, info

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()