import gymnasium as gym
import torch
from PIL import Image
from itertools import count
import os

from NeuralNetwork import DQN
from Algorithm import exploit

def load_model(model_path, n_observations, n_actions, device):
    model = DQN(n_observations, n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    return model

def run_episode(env, model, device):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    images = []
    terminated = False
    for t in count():
        action = exploit(state, model)
        ''' 항상 착취 행동을 선택한다.'''

        observation, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 이미지 렌더링
        img = env.render()
        images.append(img)
        print(f"Finished after {t+1} timesteps")

        if terminated:
            break

        if t > 1000:
            print("Episode is too long. Break.")
            break
    return images

def save_gif(images, filename):
    imgs = [Image.fromarray(img) for img in images]
    imgs[0].save(fp=filename, format='GIF', append_images=imgs[1:], save_all=True, duration=40, loop=0)

# GPU 사용 가능 여부 확인
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'outputs/policy_net_600.pth'  # 모델 파일 위치 지정

# 환경 생성
env = gym.make("CartPole-v1", render_mode="rgb_array")
n_actions = env.action_space.n
''' CartPole-v1 환경의 액션 수 '''
state, info = env.reset()
''' CartPole-v1 환경의 초기 상태 '''
n_observations = len(state)
''' CartPole-v1 환경의 관측치 수 '''

policy_net = load_model(model_path, n_observations, n_actions, device)

# 한 에피소드만을 실행한다.
images = run_episode(env, policy_net, device)

# GIF로 저장
if not os.path.exists('outputs'):
    os.makedirs('outputs')

gif_filename = 'outputs/episode.gif'
save_gif(images, gif_filename)

print(f"Saved an episode as a GIF to '{gif_filename}'")
