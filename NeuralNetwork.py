import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    '''
    신경망 구조로, 세 개의 선형(완전 연결)계층을 포함한다. 
    입력 계층은 환경의 관측치를 받아들이고, 
    출력 계층은 각 가능한 행동에 대한 예상 가치를 출력한다.

    I(입력) --> H1(은닉1) --> H2(은닉2) --> O(출력)
    '''

    def __init__(self, n_observations, n_actions):
        '''
        신경망 구조로, 세 개의 선형(완전 연결)계층을 포함한다. 
        입력 계층은 환경의 관측치를 받아들이고, 
        출력 계층은 각 가능한 행동에 대한 예상 가치를 출력한다.

        I(입력) --> H1(은닉1) --> H2(은닉2) --> O(출력)
        '''
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        '''
        순전파 함수.
        입력 데이터 x를 받아 신경망을 통과시키고 출력을 반환한다.
        (착취) 다음 행동을 결정하기 위해서 사용된다.
        Output: [[left0exp,right0exp]...]
        '''
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
