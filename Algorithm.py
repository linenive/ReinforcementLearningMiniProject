import math
import random
import torch

EPS_START = 0.9 
''' ε-greedy 알고리즘의 시작 ε 값'''
EPS_END = 0.05
''' ε-greedy 알고리즘의 최종 ε 값'''
EPS_DECAY = 1000
''' ε-greedy 알고리즘의 지수 감쇠율. 높을수록 감쇠 속도가 느리다. '''

def select_action(state, policy_net, env, device, steps_done):
    '''
    행동 선택 함수. ε-greedy 전략에 따라 행동을 선택한다.
    랜덤하게 행동을 선택하거나 (탐색)
    신경망의 예측을 사용해 최적의 행동을 선택한다 (착취)
    '''
    sample = random.random()
    # 모델이 학습할수록 ε 값이 줄어든다. (탐색 행동이 줄어든다)
    # ε 값은 EPS_END로 수렴한다.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    # Exploit. 신경망을 기반으로 행동을 선택한다.
    if sample > eps_threshold:
        return exploit(state, policy_net)
    # Explore. 무작위로 행동을 선택한다.
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def exploit(state, policy_net):
    '''
    ε-greedy 전략에 따라 착취 행동을 선택한다.
    '''
    with torch.no_grad():
        '''
        해당 블록 내에서는 pytorch의 자동 미분 엔진이 텐서의 연산 기록을 추적하지 않게 된다.
        중간 결과(그래디언트)를 저장할 필요가 없어 메모리 사용이 최적화된다.
        그래디언트는 학습 과정에서만 필요하며, 지금과 같이 단순 실행만 할 때는 필요하지 않다.
        '''
        
        return policy_net(state).max(1)[1].view(1, 1)
        '''
        신경망에 현재 상태를 입력으로 넣어 예상 가치(Q-value)를 출력한다.
        .max(1) 함수는 텐서의 첫번째 차원을 따라 최대값을 찾는다.
        1은 함수가 행을 따라 작업, 즉 각 행에서 최대값을 찾도록 지시한다.
        [0]은 최대값을 반환하고, [1]은 최대값의 인덱스를 반환한다.
        .view(1, 1)은 반환된 최대값을 1x1 텐서로 차원을 재구성한다.
        이는 나중에 다른 연산과 호환되도록 하기 위함이다.
        '''
