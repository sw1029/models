import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from tqdm import tqdm

class Model(nn.Module):  # 모델: MLP(multi layer perceptron) + multi head attention
    def __init__(self, input_dim, hidden_dim, num_head, attention_stack, activation_func, linear_count,
                 linear_dim_list):
        # 모델의 생성자. hidden_dim = hidden dimention 차원, num_head = head 갯수, attention_stack = attention을 몇번이나 반복할건가
        # activation function : 활성화 함수. ReLu, sigmoid, tanh 등이 있다.
        super().__init__()  # 부모 class의 생성자
        self.input_dim = input_dim  # 입력받는 데이터의 차원
        self.hidden = hidden_dim  # 은닉 데이터의 차원
        self.head = num_head
        self.head_dim = self.hidden // self.head
        # multi head attention의 경우 데이터를 분할해서 attention을 수행 후 병합하여 attention score를 도출한다.
        # head_dim의 의미는 하나의 head가 처리해야 할 차원의 크기를 의미함
        self.stack = attention_stack  # attention 반복 횟수

        self.linear_count = linear_count
        self.linearContainer = nn.ModuleList()
        self.linearContainer.append(nn.Linear(self.input_dim, linear_dim_list[0]))
        for i in range(len(linear_dim_list) - 1):  # 마지막 출력 차원은 직접 조절해야함
            self.linearContainer.append(nn.Linear(linear_dim_list[i], linear_dim_list[i + 1]))

        self.q_linear = nn.Linear(1, hidden_dim)
        self.k_linear = nn.Linear(1, hidden_dim)
        self.v_linear = nn.Linear(1, hidden_dim)  # attention을 위한 query, key, value layer

        self.loop_q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.loop_k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.loop_v_linear = nn.Linear(hidden_dim,
                                       hidden_dim)  # 반복문 안에서 사용할 linear layer. loop 내부에서는 데이터의 형태가 변한 것을 다시 입력으로 받기 때문에 문제가 발생함

        self.attention_out_linear = nn.Linear(hidden_dim, hidden_dim)  # attention 결과를 가공하기 위한 layer
        self.final_attention_out_linear = nn.Linear(hidden_dim, input_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)  # 출력값 정규화를 위한 layer - 학습 안정성 증가
        self.final_norm = nn.LayerNorm(input_dim)
        self.sigmoid = nn.Sigmoid()  # 확률값으로 변환하기 위한 활성화 함수
        self.dropout = nn.Dropout(p=0.2)  # dropout 설정 - 임의의 뉴런을 비활성화 시키는 것으로 과적합 방지
        self.activation = activation_func  # 활성화 함수

    def multi_head_attention(self, x):
        batch_size, seq_length = x.size()
        # batch size와 sequence의 차원 얻어냄. 여기서 sequence는 특징의 개수를 의미함.
        # mlp를 통해 linear를 통과한 차원의 크기를 sequence로 활용하는것

        x = x.unsqueeze(-1)  # (batch size, input seqence length, 1) 형태로 변환. 각 숫자를 크기가 1인 차원의 형태를 가지게 하는것.

        Q = self.q_linear(x).view(batch_size, seq_length, self.head, self.head_dim).transpose(1,
                                                                                              2)  # 현재 집중을 어디에 하여야 하는가?
        K = self.k_linear(x).view(batch_size, seq_length, self.head, self.head_dim).transpose(1, 2)  # 집중을 하여야 하는 특징들
        V = self.v_linear(x).view(batch_size, seq_length, self.head, self.head_dim).transpose(1, 2)  # 실제 정보
        # linear 통과 -> head 단위로 분할하기 위해 view를 이용해 reshape -> 출력 차원의 순서의 변경을 위해 transpose 사용

        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Query와 Key의 내적을 통해 토큰의 관련성을 구함
        # -2 와 -1는 각각 마지막에서 두번째와 마지막 차원을 의미
        # matmul은 Q와 K의 내적을 하기 위함이고, head dimention의 제곱근을 나눠 주는 것으로 값이 지나치게 커지는 것을 방지
        attention_weight = F.softmax(score, dim=-1)  # softmax를 통해 weight를 확률처럼 변환. 확률이 낮더라도 기회는 주어지게 하는 것
        attention_weight = self.dropout(attention_weight)  # 특정 부분에 너무 집중하면 과적합된다거나 하는 문제가 생김. 이를 방지하기 위해 dropout 지정

        attention_output = torch.matmul(attention_weight, V)  # 이 부분이 attention을 적용한 결과값. 두 벡터를 곱하는 행위이다
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden)
        # 위에서 여러 head로 분할했던 결과를 다시 하나로 합침
        # contiguous : transpose를 통하면 a[1][2]를 통해 접근하던걸 a[2][1]로 접근할 수 있게 되지만, 접근 방식만 바뀌는거지 실제 데이터 내에서는 변하지 않음
        # contiguous는 이 경우 데이터를 메모리에서 연속적으로 존재하게 만들어준다.
        # 이후 view에 입력한 스타일로 데이터의 형태가 변환됨
        output = self.attention_out_linear(attention_output)  # output을 다시 한번 재가공
        output = self.dropout(output)
        output = self.norm(
            output + attention_output)  # 원래의 정보와 attention을 통한 출력값을 합하고 정규화하는것으로 이전의 정보를 지키며 attention도 수행함

        for st in range(self.stack - 1):
            # 이미 위에서 attention 한번은 했으니 1을 빼 준다
            # 위에서 했던 행위의 단순 반복
            Q = self.loop_q_linear(output).view(batch_size, seq_length, self.head, self.head_dim).transpose(1, 2)
            K = self.loop_k_linear(output).view(batch_size, seq_length, self.head, self.head_dim).transpose(1, 2)
            V = self.loop_v_linear(output).view(batch_size, seq_length, self.head, self.head_dim).transpose(1, 2)
            score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weight = F.softmax(score, dim=-1)
            attention_weight = self.dropout(attention_weight)
            attention_output = torch.matmul(attention_weight, V)
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden)
            if st != self.stack - 2:
                output = self.attention_out_linear(attention_output)
                output = self.dropout(output)
                output = self.norm(output + attention_output)
            else:
                output = self.final_attention_out_linear(attention_output)
                attention_output = self.final_attention_out_linear(attention_output)
                output = self.dropout(output)
                output = self.final_norm(output + attention_output)
        return output

    def forward(self, x):  # 모델의 forward 기능. 이 함수를 통해 실제적으로 학습이 진행된다.

        '''
        실제 모델의 input data의 형식은 dataframe의 행 단위로 잘라놓은 데이터를 batch로 묶어서 입력받는 형태
        multi head attention의 경우 어떻게 적용해야 할지 고민해볼것
        '''

        if self.stack != 0 :
            x = self.multi_head_attention(x)#(batch_size, input_dim, input_dim) 형태로 반환
            x = x.mean(dim = 1)#(batch, input dim)형태로 변환

        for lin in range(len(self.linearContainer)):
            linear = self.linearContainer[lin]
            x = self.activation(linear(x))
            if lin != len(self.linearContainer) - 1: x = self.dropout(x)
            else: x = self.sigmoid(x) #이진분류 모델이라 sigmoid 사용. 다른 task 수행 시 교체 요망

        return x.squeeze(-1)
