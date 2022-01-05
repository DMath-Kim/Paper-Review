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
