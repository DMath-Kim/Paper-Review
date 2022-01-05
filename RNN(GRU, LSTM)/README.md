<h3>GRU Networks (Gated Recurrent Unit aims to solve the vanishing gradient</h3>

Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

[Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014).]

GRU (Gated Recurrent Unit) aims to solve the vanishing gradient using update gate and reset gate.

<h5>Theses two types of gates decide what information should be passed to the output</h5>

( Specially, Trained to keep information from long ago [include irrelevant prediction information] )

*Key points*

[ref by https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be]

1. Update Gate

Helps the model to determine how much of the past information to update

Using activation function is sigmoid --> output val : [0, 1] {Reset gate also same}

![updategate](https://user-images.githubusercontent.com/96281316/148239916-fd839a00-5c5a-44b7-a981-4e30da1e3b61.png)

2. Reset Gate

Used from the model to decide how much of the post information to forget

![resetgate](https://user-images.githubusercontent.com/96281316/148239921-0e47e681-114d-4b3a-97b4-271d08897ce4.png)

3. Current memory content

Add term --> Determine what to remove from the previous information 

![currentmemorycontent](https://user-images.githubusercontent.com/96281316/148239923-ba0b8788-e374-4a04-b3d7-ae0c881c0e5c.png)

4. Final memory at current time step

Previous information update + Current information update

![finalmemoryatcurrenttimestep](https://user-images.githubusercontent.com/96281316/148239925-c0024ced-7cfe-46eb-9294-3476f23863e1.png)

<h5>ADD LSTM (Long Short Term Memory)</h5>

LSTM was designed to overcome the long-term dependency problem faced by RNN (Vanishing Gradients)

*Key points*

1. Forget gate (How much irrelevant data to remove from memory)

If components(newly input data + previous hidden state) are deemed irrelevant then forget gate is close to 0.

when relevant is close to 1

2. New Memory Network (Input gate)

[ Inputs are actually the same as the inputs to the forget gate. ]

Previous hidden cell and new input data to generate a new memory update vector.

Using tanh activation function

Apply sigmoid function to newly input data. This is because a value between 0 and 1 is returned to receive the ratio of how useful and usable input data to be used is new.

3. Output gate

Use newly updated cell state, previous hidden state, new input

just output this small network

figure by https://ko.wikipedia.org/wiki/%EC%9E%A5%EB%8B%A8%EA%B8%B0_%EB%A9%94%EB%AA%A8%EB%A6%AC

![LSTM_Cell](https://user-images.githubusercontent.com/96281316/148262972-c7be7761-eaf1-45a7-9b4e-7c1c6c235a6b.png)
