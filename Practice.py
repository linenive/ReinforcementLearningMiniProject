import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

# 신경망 모델을 구성하고 훈련하기 위한 메인 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim

from NeuralNetwork import DQN
from Algorithm import select_action

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# ion: 대화형 모드 켜기
plt.ion()

# GPU 사용 가능 여부 확인
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''상태, 행동, 보상, 다음 상태를 저장하기 위한 튜플. 화면의 차이인 state로 (state, action) 쌍을 (next_state, reward) 결과로 매핑한다.'''
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    학습을 위해 과거의 Transition들을 저장하는 순환 버퍼.
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''transition 저장'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        학습을 위한 전환의 무작위 배치를 선택한다.
        무작위로 샘플링하면 배치를 구성하는 전환들이 비상관(decorrelated)하게 된다.
        이것이 DQN 학습 절차를 크게 안정시키고 향상시키는 것으로 나타났다.
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
''' 한 번에 학습할 Transition의 수 '''
GAMMA = 0.99
''' 
보상의 할인율(discount factor)
할인율은 0과 1 사이의 값으로 설정된다. 이 값이 너무 높으면 (1에 가깝다면) 에이전트는 먼 미래의 보상을
현재 보상만큼 중요하게 고려하게 되어 학습 과정이 불안정해질 수 있다. 반대로, 너무 낮은 값은 미래 보상을 
거의 고려하지 않게 되므로, 단기적인 행동만을 취하게 된다.
'''
TAU = 0.005
'''
목표 네트워크의 업데이트 속도
업데이트 속도가 너무 크면 학습이 불안정해질 수 있고, 너무 작으면 학습 속도가 느려질 수 있다.
'''
LR = 1e-4
'''
학습률(learning rate)
오류(error) 신호에 기반하여 각 가중치의 조정 크기를 결정한다.
학습률이 높으면 가중치 조정이 크게 이루어져 빠르게 학습할 수 있지만, 
너무 높으면 과적합(overfitting)이나 발산(divergence)의 위험이 있다.
실험을 통해 최적의 값을 찾는 것이 일반적이다.
'''

# 환경 생성
env = gym.make("CartPole-v1", render_mode="rgb_array")
n_actions = env.action_space.n
''' CartPole-v1 환경의 액션 수 '''
state, info = env.reset()
''' CartPole-v1 환경의 초기 상태 '''
n_observations = len(state)
''' CartPole-v1 환경의 관측치 수 '''

policy_net = DQN(n_observations, n_actions).to(device)
''' 
정책 네트워크. 현재 학습된 최신 정책을 나타낸다. 
즉, 현재의 상태를 입력으로 받아 각 행동의 예상 가치(Q-value)를 출력한다.
- 용도: 어떤 행동이 현재 상태에서 가장 좋은 결과를 낼 지 예측한다.
'''
target_net = DQN(n_observations, n_actions).to(device)
''' 
목표 네트워크. 목표 네트워크는 정책 네트워크의 학습을 안정화하는 데 사용된다.
정책 네트워크와 동일한 구조를 가지지만, 그 가중치는 일정 간격으로만 업데이트된다.
- 용도: 목표 네트워크는 학습 과정에서 손실 함수의 목표 값을 계산하는 데 사용된다.
- 정책 네트워크가 최적화되는 동안 안정적인 학습 목표를 제공하여 발산을 방지한다.
'''
target_net.load_state_dict(policy_net.state_dict())
''' 목표 네트워크의 가중치를 정책 네트워크의 가중치로 초기화 '''

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
'''
AdamW 최적화 알고리즘을 사용하여 정책 네트워크를 최적화한다.
.parameters() 함수는 모델의 모든 학습 가능한 매개변수를 반복 가능한 객체로 반환한다.
PyTorch의 최적화 알고리즘 중 하나로, Adam 최적화 알고리즘의 변형이다.
기본적으로는 경사 하강법(gradient descent)이다.
AdamW는 Adam의 가중치 감쇠(weight decay)를 수정한 것으로, 가중치 감쇠가 더 안정적이다.
'''
memory = ReplayMemory(10000)

steps_done = 0
''' 진행한 학습 단계 수 '''

episode_durations = []
''' 각 에피소드의 지속 시간을 저장하기 위한 배열 '''

def plot_durations(show_result=False):
    ''' 
    각 에피소드의 지속 시간을 시각화한다.
    주요 선은 각 에피소드의 지속 시간을 나타내며,
    평균 선은 100개의 에피소드 평균을 나타낸다.
    '''
    plt.figure(1)
    ''' 도표 번호 지정 '''
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    ''' 에피소드 지속 시간을 텐서로 변환 '''
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations_t.numpy())
    ''' 각 에피소드의 지속 시간을 그린다. '''

    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) 
    ''' 도표가 업데이트되도록 잠시 멈춘다. '''

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    '''
    신경망의 가중치를 업데이트한다.
    샘플링된 배치에서 손실을 계산하고, 역전파를 통해 신경망을 최적화한다.
    '''

    if len(memory) < BATCH_SIZE:
        ''' 모델을 최적화하기 위한 충분한 Transition이 쌓일 때까지 기다린다. '''
        return
    
    transitions = memory.sample(BATCH_SIZE)
    ''' 
    무작위로 샘플링된 배치를 가져온다. 이렇게 하면 배치를 구성하는 전환들이 비상관(decorrelated)
    하게 된다. 즉, DQN 학습 절차를 크게 안정시키고 향상시킨다. 
    '''

    batch = Transition(*zip(*transitions))
    '''
    여러 개의 Transition 객체들로부터 새로운 Transition 객체를 생성하는 효율적인 방법.

    예:
    transitions = [Transition(1, 'a', 2, 3), Transition(4, 'b', 5, 6)]
    위 코드에서 zip(*transitions)을 호출하면, 
    [(1, 4), ('a', 'b'), (2, 5), (3, 6)]을 반환한다.
    최종적으로는: Transition(
        state=(1, 4), action=('a', 'b'), next_state=(2, 5), reward=(3, 6))
    '''

    non_final_mask = torch.tensor(
        data=tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device, dtype=torch.bool)
    '''
    각 next_state가 None이 아닌 경우를 확인하여 불리언 텐서를 생성한다. 
    여기서 None이 아니라는 것은 해당 상태가 최종 상태(에피소드가 종료된 상태)가 아니라는 의미이다.
    batch.next_state에서 각 상태에 대해 lambda 함수를 사용하여 None 여부를 검사한다. 
    None이 아니면 True, None이면 False를 반환하는 불리언 값의 시퀀스를 생성한다.
    이 시퀀스는 torch.tensor()를 통해 텐서로 변환되며, 이 텐서는 GPU 등의 device에 할당된다.
    '''

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None])
    '''
    최종 상태가 아닌 상태들을 모두 연결하여 하나의 텐서로 만든다.
    신경망을 통과하여 다음 상태의 값들을 예측하는 데 사용된다.
    '''

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    '''
    각 상태에서 실제 취한 행동에 대한 Q-value만을 추려 모은다.

    policy_net(state_batch)는 주어진 배치에 대해 가능한 모든 행동의 Q-values를 계산하고,
    gather 함수를 사용하여 각 상태에 대해 취해진 특정 행동의 Q-value만을 선택한다.
    
    예를 들어, policy_net은 각 상태에 대해 각 행동의 Q-value를 반환한다.
    [[Q(s1, a1), Q(s1, a2)], 
    [Q(s2, a1), Q(s2, a2)]]

    action_batch는 각 상태에 대해 선택된 행동의 인덱스를 포함한다. 
    예를 들어, [0, 1]이라면 첫 번째 상태에서 첫 번째 행동을,
    두 번째 상태에서 두 번째 행동을 선택했다는 의미이다.

    .gather(1, action_batch)는 action_batch에서 지정한 인덱스에 따라 각 행의 해당하는 
    요소를 선택한다.
    예를 들어, 위의 예시에서는 다음과 같이 선택된다.
    - 첫 번째 행에서 인덱스 0 (Q(s1, a1))
    - 두 번째 행에서 인덱스 1 (Q(s2, a2))

    최종적으로, state_action_values는 선택된 행동에 대한 Q-value를 포함한다.
    [Q(s1, a1),
     Q(s2, a2)]
    '''

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    '''
    주어진 크기(BATCH_SIZE)의 새로운 텐서를 생성하고, 모든 요소를 0으로 초기화한다.
    이 텐서는 나중에 각각의 다음 상태에 대한 예측된 Q-values를 저장하는데 사용된다.
    '''

    with torch.no_grad(): # 목표 Q-value를 계산할 때 모델의 가중치 업데이트를 방지한다.
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    ''' 
    목표 네트워크를 사용하여 각 상태에 대한 모든 가능한 행동의 예상 가치(Q-values)를 계산하고,
    최대 Q-value를 선택한다.

    아직 게임이 계속되고 있는 상태들의 인덱스에만 최대 Q-values를 저장하여, 이 값들을 이용해 
    나중에 얼마나 좋은 선택을 했는지를 평가할 수 있다.
    '''
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    '''
    기대 Q-value를 계산한다.
    기대 Q-value는 다음 상태의 최대 Q-value에 할인율을 곱한 값에 보상을 더한 값이다.
    '''

    criterion = nn.SmoothL1Loss()
    '''
    손실 함수로 Smooth L1 Loss를 사용한다. 이는 Huber 손실(Huber loss)의 변형이다.
    Huber 손실은 L1 손실과 L2 손실의 장점을 결합한 것으로, L1 손실의 이상치에 대한 강건함과
    L2 손실의 미분 가능성을 모두 가지고 있다.
    '''
    
    loss = criterion(
        state_action_values, expected_state_action_values.unsqueeze(1))
    '''
    손실을 계산한다. 두 값 사이의 차이를 측정하는 방법 중 하나.
    L1 손실 함수와 비슷하지만, 값의 차이가 작을 때는 제곱을 사용하여 더 '부드러운' 접근을 하고, 
    값의 차이가 클 때는 절대값을 사용한다. 이 '부드러움'은 모델 학습시 발생할 수 있는 급격한 변화나 
    이상치(outliers)에 덜 민감하게 만든다. 예측 오차가 클 때 너무 큰 패널티를 주지 않으면서도,
    작은 오류에 대해서는 민감하게 반응하여 모델의 안정적인 학습을 돕는다. 

    unsqueeze(1)은 텐서의 차원을 늘리는 함수이다. 두 텐서의 차원을 맞추기 위해 사용된다.
    '''

    optimizer.zero_grad()
    '''
    파이토치에서는 그래디언트가 누적되기 때문에, 
    각 배치에서 새로운 그래디언트를 계산하기 전에 이전 그래디언트를 0으로 만들어줘야 한다.
    '''
    loss.backward()
    '''
    손실 함수로부터 그래디언트를 계산한다. 경사 하강.
    '''
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    '''
    그래디언트 클리핑을 통해 그래디언트 폭주(gradient explosion) 문제를 방지한다. 
    그래디언트 폭주는 그래디언트 값이 너무 커져서 수치적으로 불안정해지는 현상이다.
    100과 -100 사이의 값으로 그래디언트를 잘라낸다. 안정적인 학습을 위한 테크닉.
    '''
    optimizer.step()
    ''' 계산된 그래디언트를 사용하여 모델의 가중치를 업데이트한다. '''

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 700
    print('Using GPU')
else:
    num_episodes = 50
    print('Using CPU')

''' 학습 루프. 각 에피소드에서 에이전트가 환경과 상호작용하면서 학습한다.'''
for i_episode in range(num_episodes):
    state, info = env.reset()
    ''' CartPole-v1 환경의 초기 상태'''
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ''' 상태를 텐서로 변환하고 차원을 추가한다. '''
    for t in count():
        action = select_action(state, policy_net, env, device, steps_done)
        ''' ε-greedy 전략에 따라 탐색 또는 착취 행동 선택 '''
        steps_done += 1
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        ''' 선택한 액션에 따라, 환경으로부터 다음 상태와 보상을 받는다. '''

        reward = torch.tensor([reward], device=device)
        ''' 보상을 텐서로 변환한다. '''

        done = terminated or truncated

        # 100 회마다 모델을 파일로 저장한다. 
        if i_episode % 100 == 0:
            # outputs 폴더 생성
            if not os.path.exists('outputs'):
                os.makedirs('outputs')

            torch.save(policy_net.state_dict(), f'outputs/policy_net_{i_episode}.pth')
            
        if terminated:
            next_state = None
            print(f'Episode {i_episode} finished after {t+1} timesteps, terminated')
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            ''' 다음 상태를 텐서로 변환한다. '''
            print(f'Episode {i_episode} finished after {t+1} timesteps, truncated')

        memory.push(state, action, next_state, reward)
        ''' Transition을 메모리에 저장한다. 모았다가 학습에 사용된다. '''

        state = next_state

        optimize_model()
        ''' 학습: 정책 네트워크의 최적화 한단계 수행 '''

        # 목표 네트워크의 가중치를 소프트 업데이트한다.
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU \
                + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
