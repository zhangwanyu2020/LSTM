import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class RNN(object):
    '''
    Args:
      batch_size: how many samples of a batch,
      embedding_dim: the embedding dim of a word,
      hidden_units: how many units of hidden layer,
      y_units: how many classes of label.

    Returns:
      seq_state: the hidden state of all sequence;
      a_next: the last units's state of  sequence;
      y_pred: the prediction of all sequence.

    '''

    def __init__(self, batch_size, embedding_dim, hidden_units, y_units):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.y_units = y_units

    def rnn_cell_forward(self, xt, a_prev):
        # 输入层到隐藏层的参数矩阵，Wax.shape=[embedding_dim,hidden_units]
        Wax = np.random.randn(self.embedding_dim, self.hidden_units)
        # 隐藏层到隐藏层的参数矩阵，Waa.shape=[hidden_units,hidden_units]
        Waa = np.random.randn(self.hidden_units, self.hidden_units)
        # 隐藏层到输出层的参数矩阵，Wya.shape=[hidden_units,y_units]
        Wya = np.random.randn(self.hidden_units, self.y_units)
        # ba,by是偏置项，相加时会广播机制展开
        ba = np.random.randn(1, self.hidden_units)
        by = np.random.randn(1, self.y_units)

        a_next = np.tanh(np.dot(a_prev, Waa) + np.dot(xt, Wax) + ba)
        yt_pred = softmax(np.dot(a_next, Wya) + by)

        # 返回t时刻的隐状态和预测值，a_next.shape=[batch_size,hidden_units],yt_pred.shape=[batch_size,y_units]
        return a_next, yt_pred

    def rnn_forward(self, x):
        # 输入序列x的第一维是seq_len
        seq_len = x.shape[0]
        seq_state = np.zeros((seq_len, self.batch_size, self.hidden_units))
        y_pred = np.zeros((seq_len, self.batch_size, self.y_units))
        # 初始化第一个单元接受的隐状态
        a_next = np.random.randn(self.batch_size, self.hidden_units)
        # 根据输入序列的长度，进行RNN编码
        for t in range(seq_len):
            a_next, yt_pred = self.rnn_cell_forward(x[t, :, :], a_next)
            seq_state[t, :, :] = a_next
            y_pred[t, :, :] = yt_pred
        # 返回整个序列的隐状态，最后时刻的隐状态，整个序列的预测值
        # seq_state.shape=[seq_len,batch_size,hidden_units]
        # a_next.shape=[batch_size,hidden_units]
        # y_pred.shape=[seq_len,batch_size,y_units]
        print(seq_state.shape, a_next.shape, y_pred.shape)
        return seq_state, a_next, y_pred


model = RNN(batch_size=8, embedding_dim=10, hidden_units=5, y_units=2)
x = np.random.randn(10, 8, 10)
model.rnn_forward(x)