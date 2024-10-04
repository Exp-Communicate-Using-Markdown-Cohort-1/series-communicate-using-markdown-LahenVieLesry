- Jin, Lingxin, Xianyu Wen, Wei Jiang, and Jinyu Zhan. “A Survey of Trojan Attacks and Defenses to Deep Neural Networks.” arXiv, August 15, 2024. http://arxiv.org/abs/2408.08920.
    
    neural networks can fit almost any complex function model if sufficient hidden units are available. The complexity of neural networks also shows that part of the neurons in the model can not easily locate their explicit functions
    
    A reasonable explanation of the success of data-driven Trojan is that the fixed pattern equipped to the data makes the model learn more about the characteristics of triggers. Therefore, the manipulated model would produce the intended results of the poisoned data.
    

- **现有问题**
    - **时间域触发器的局限性**：传统的后门攻击方法主要在**时间**域（即直接在时间域上叠加不同的音频）注入触发器。这种方法虽然在视觉上隐蔽性较好，但可能会破坏图像的语义信息，影响图像质量。
    - **频率域触发器的挑战**：近期研究尝试在频率域注入触发器，通过修改图像的频谱信息实现更隐蔽的攻击。然而，现有的频率域攻击主要集中在中高频部分，这些触发器容易被图像处理操作（如压缩、滤波）破坏，同时在频率空间中引入明显的异常特征，容易被检测。

# Compressional-based Conditional Imperceptible Black-box Backdoor Attack

## Abstract
As deep neural networks (DNNs) have integrated into many aspects of our lives, many previous studies have shown that they are susceptible to be poisioned by backdoor attacks, attackers can control the output of the model through well-designed triggers. However, much of the previous researches have tend to focus on image domain and researches in audio domain is still understudied. Triggers in previous studies that have been done can either be filtered inadvertently by simple filters, or can often be heard by human ear, or inroduce significant difference in spectrograms. In his paper, we inroduce **Name**, a novel compression scenario-oriented approach to imperceptible backdoor attack by interpolating the original audio with its compressed variety in spectrum for iterative optimization, this ensures that the trigger is effective, while also balancing the stealthiness so that it is hidden from human perception. Our proposed attack has conducted extensive and comprehensive experiments, the results show high attack efficiency as well as strong robustness with only a small poisoning rate compared to previous methods.

## I. Introduction
Over the past several years, Deep Neural Networks (DNNs) have undergone continuous development and eventually won wide acclaim for their excellent performance and have penetrated into every aspect of our lives, with applications in various fields such as smartphones, autonomous driving and many other areas ralated to privacy and security. Therefore, it is crucial to ensure the security of DNNs, yet disappointingly is that although DNNs has excellent performance, it exposes a portion of the model's weaknesses at various stages such as data collection, model training, model development, and model interface, and these weaknesses invite the risk of DNN being maliciously attacked.
It is well known that the performance of DNNs mainly depends on a large amount of training data and computational resources, which is a great burden for some individual researchers or small teams. In order to avoid spending too much time and labor to produce datasets, researchers tend o use thrid-party datasets or collect data on the web by themselves, which is difficult toensure the accuracy and safety of the data. Second, the training of a large-scale DNN often requires several or even hundreds of high-performance arithmetic cards, so researchers are inclined to use outsourcing to reduce the cost of hardware resources required for training DNNs. For the first type, a complete outsourcing approach is adopted, i.e., Machine Learning as a Service (MLaaS) technology is leveraged, whereby he dataset and model archiecture are uploaded to a cloud computing platform, and the hardware resources of the cloud platform are leased to complete the training of the DNNs. The other is to adopt the transfer learning approach by purchasing or downloading a model from a model repository, which accomplishes a task similar to the one that needs to accomplish, and then researchers re-train the later layers of the model to adjust the model to match his or her own task. While these approachese are relatively convenient, they also expose some security issues, backdoor attacks being one of them.
Typically, the goal of backdoor attack is to embed a backdoor in a model. When the model is deployed and enters the inference pahse, if the model's input is a clean sample, the model behaves similarly to a normal model and outputs normally; however,, if the model's input is a poisioned sample with specific triggers embedded, the trigger will activate the backdoor-related neurons in the model, thus outputting the attacker-specified target labeling. Compared to poisoning attack, backdoor attacks do not affect the normal performance of the model; compared to adversarial attack, backdoor attack have a wider range of attack phases, whereas backdoor attacks modify the model by changing the weights or structure of the model.
In the image domain, backdoor attack researchers have started to focus on how to improve the concealment of triggers, but this kind of research is still lacking in the audio domain. **\*\*** et al. use ultrasonic as trigger, which can be easily eliminated inadvertently by high-pass filters in the preprocessing step. Some researchers also select corresponding trigger based on the samples from a pre-defined trigger pool, triggers in which are either audible natural sounds or ambient sounds, moveover, the number of triggers in the trigger pool is limited. Similary, **\*\*** et al. proposed to add a small piece of audio as trigger by utilizing the auditory masking effect, but this can leave significant traces on the spectrum. **\*\*** et al. generate backdoor samples through timbre conversion, it is a method that can severely disrupt the spectrum.
In order to break through the shortcomings of the above studies and achieve high attacks effiency while keeping backdoor samples indistinguishable from the ~~clean samples~~ corresponding trigger-free counterparts in terms of spectrograms, this paper proposes a method called **Name** that generates backdoor samples by compressing the original audio, and the triggers generated in this way are imperceptible, and furthermore, the backdoor samples can be further tuned by loss to to be more sensitive to the target's tiredness while being more similar to the original samples.
Our contributions can be summarized as follows:
* We propose a novel backdoor attack utilizing compression techniques as triggers, which is a scenario-orinted method that can be inadvertently triggered in a passive manner.
* We propose an optimization algorithm that can dynamically adjust the trigger for each sample so that it can effectively balance attack success rate and inperceptibility, and achieve more than 98% attack success with less than 3% poisoning rate.
* We conducted extensive experiments to thoroughly investigate the effectiveness of different audio compression algorithms, compared with other research methods in several aspects, and analyze the strength of our method in terms of effectiveness, steganography, and robustness.
The rest of the paper is organzied as follows. Related works are reviewed in Section II. A formal definition of the backdoor attack is provided in Section III. The proposed backdoor attack is elaborated in Section IV, followed by experimental results and analysis in Section V. The paper is concluded in Section VI.

## II. Related Works
* Perceptible Backdoor Attack
Existing studies generally agree that backdoor attacks in deep learning were pioneered by Gu et al. They set a fixed trigger pattern at a fixed location in the images, and at the same time modify their labels to the tarfet labels specified by the attacker. After that, more researchers started to study backdoor attacks. Chen et al. proposed to embed pictures or noise into clean samples by blend to generate backdoor samples, and from then on, researchers also started to pay attention to the practical application of backdoor attacks in the physical world. Li et al. found that du to the difference in the camera positions, changing the position or the appearance of the trigger in the test would lead to an extremely rapid decrease in attack success rate, so they improve the effectiveness of the backdoor attack by transforming the training data. Tang et al. add a module to capture triggers at fixed positions in poisioned sample when the model is deployed. It is worth mentioning that by modifying different patterns, the method can realize multi-target attacks. Liu et al. add a preset reflection pattern as a trigger. Similarly, the Barni et al. used sinusoidal signals as triggers. Ren et al. perform a backdoor attack on a speech quality assessment system, which is the first attempt in a regression task. Koffas et al. proposed a stylized backdoor attack that generates triggers by simulating electric guitar effects (e.g. pitch shift, distortion, chorus, reverb, gain, filtering, and phase shift) that alter the original signal so that it is distinguishable in the training model but still maintains the sound quality at an acceptable level. ~~This approach does not require complex generatice model training, simplifying the attack implementation process and making the attack easy to deploy. Howecer, stylized backdoors are highly dependent on the target application and may require different style generation methods in different domains and tasks.~~
* Imperceptible Backdoor Attack
Since perceptible triggers are more clearly characterized, they tend to be more effective in attacking. In contrast, imperceptible backdoor attacks are more concerned with how to reduce the difference between backdoor samples and clean samples so that they are imdestinguishable to human or detection algorithms, thus avoiding suspicion of the dataset, so imperceptible triggers are more research-worthy and relevant. Zhong et al. utilize a generic antagonistic pertubation as a backdoor trigger, and the number of paradigms is minimized to ensure that the pertubation is not visible. Li et al. utilize steganography that using the least significant bit (LSB) algorithm to embed triggers into the poisoned training set, making the triggers more stealthy than any previous work. Similarly, Li et al. train the encoder to embed strings into images while minizing the perceptual difference between the input image and the encoded image. Ye et al. utilize padding to fill in zeros at fixed positions of a given audio sample to act as trigger. This method enables efficient attacks and is difficult to detect by the human ear as the padding is very short. Cai et al. innovatively utilize the auditory masking phenomena to add triggers to the audio. For a given sample, the authors first raised its overall pitch, then found the highest-pitched part of the sample and inserted a relatively low-pitched and short-duration piece of audio as a trigger. Shi et al. chose ultrasound as a trigger which cannot be heard by human, and reduced the difference in model output when the trigger is injected at different time positions by co-optimizing the model and audio trigger during the training process, as well as reducing the difference in model output when the trigger overlaps between the voiced and unvoiced portions of the trigger, and taking into account the factors such as sound attenuation, absorption, and reverberation in the displayed world presence, Room Impulse Response (RIR) is simulated during the optimization process to enhance the robustness of the trigger after the physical propagation process. In these ways, the attack can be independent of the trigger being at a specific time position and still maintain considerable flexibility, robustness and effectiveness in the real world.
* Backdoor Defense
Fine-pruning is proposed by Liu et al., it removes neurons that activate only on poisoned samples by pruning and fine-tunes the model with clean samples after pruning to restore model performance. This process continues until the model performance drops to a certain threshold. Zhu et al. test the model for the presence of backdoor triggers by overlaying input samples with other test samples and then feeding them into the model. The method directly utilizes other samples for superimposition and screens out the ones with relatively small entropy values as backdoor samples. Subsequently, they use speech enhancement techniques to reduce the impact of the triggers and remove backdoors from the model through pruning techniques. Chen et al. separated the poisoned samples by clustering the samples based on activation values. The method first converts the activation values of the last hidden layer of the depth model into a one-dimensional vector and then performs dimensionality reduction by principal component analysis. By performing the above procedure for each sample in the dataset, followed by segmentation based on labels, each class is clustered using K-Means method in the low dimensional space after dimensionality reduction. Tran et al. researchers proposed a new backdoor defense technique based on spectral features. The technique detects whether the average feature vector of a category changes due to a poisoned sample by analyzing the feature vectors of the internal layers of the deep learning model. Through the covariance matrix decomposition and the calculation of outlier scores, the method can effectively distinguish normal samples from poisoned samples by simply comparing them with a preset threshold.
## III. PRELIMINARIES
### A. Problem formulation
一个关键词识别模型可以被定义为：$f_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}$，其中 $\mathcal{X}$ 是输入空间，$\mathcal{Y} = \{y_1, y_2，\ldots, y_k\}$ 是一个表示 $k$ 个类别的集合的输出空间。给定一个输入样本 $x \in \mathcal{X}$，模型 $f$ 将输出一个标签 $y \in \mathcal{Y}$，$\theta$ 表示模型在训练阶段从数据集 $D_{train}=\{(x_i, y_i)|x_i \in \mathcal{X}, y_i \in \mathcal{Y}, i=1,2,\ldots,n\}$。$(x_i^{\prime}, y_t)$ 是一个修改后的中毒样本，其中 $x_i^{\prime}$ 是通过 $G(x_i)$ 生成的带有触发器的输入样本，$G$是触发器的嵌入方式，$y_t \in \mathcal{Y}$ 是攻击者指定的目标标签。中毒数据集可以通过对干净数据集中的一部分样本注入触发器来生成，即 $D_{poi}=\{(x_i^{\prime}, y_t)|x_i^{\prime} = G(x_i), (x_i, y_i) \in D_{train} \backslash D_{clean}\}$。
干净模型的训练可以看作是一个单层优化问题，即要解决如下的优化问题：
$$
\begin{equation}
    \theta = \arg \min_{\theta} \sum_{(x, y) \in D_{train}} \mathcal{L}(f(x; \theta), y)，
\end{equation}
$$
其中 $\mathcal{L}$ 是损失函数。
然而，对于后门模型的训练而言，只有当在干净样本上表现正常，同时在中毒样本上表现出攻击者指定的目标标签时，我们才认为训练得到的模型 $f_{\theta^\prime}$ 认为模型是一个成功的后门模型，例如：
$$
\begin{equation}
\left\{
    \begin{array}{l}
    f(x; \theta^\prime) = y,\\
    f(x_i^{\prime}; \theta^\prime) = y_t.
    \end{array}
\right.    
\end{equation}
$$
因此，我们需要解决一个双层优化问题，即：
$$
\begin{equation}
\begin{aligned}
    \theta^\prime &= \arg \min_{\theta^\prime} \mathcal{L}(D_{train} \cup D_{poi}, f_{\theta^\prime})\\
   & =\arg \min_{\theta^\prime} \sum_{(x, y) \in D_{train}} \mathcal{L}(f(x; \theta^\prime), y) + \sum_{(x_i^{\prime}, y_t) \in D_{poi}} \mathcal{L}(f(x^{\prime}; \theta^\prime), y_t).
\end{aligned}
\end{equation}
$$
公式3的前一项是模型在干净样本上的损失，它约束着训练得到的中毒模型在干净样本上表现正常；后一项是模型在中毒样本上的损失，它约束当中毒样本输入到中毒模型中时，模型能够输出攻击者指定的目标标签。投毒率可以定义为 $\gamma=\lvert D_{poi}\rvert/\lvert D_{train}\rvert$，投毒率越低，说明攻击者注入的中毒样本越少，攻击越隐蔽。
### B. Threat Model
**Attack Scenarios:** 现有的攻击场景大致可以分为两类，第一类是machine learning as a service(MLaaS)，用户采用外包的方式来降低训练DNN时所需的硬件资源成本，通过将数据集和模型架构上传到云计算平台，租赁云平台的硬件资源完成DNN的训练。这类场景下，攻击者可以直接访问模型的接口，能够获得模型结构的信息和训练参数，也能够控制模型训练的全过程；第二类是用户采用第三方数据集来训练模型，这类场景下，受害者通过采用公开的数据集或者将数据集采集的过程外包，受害者无法确保数据集是否可信，这为攻击者修改数据集创造了机会，但是攻击者不再能够获得受害者模型的任何信息，也无法控制模型的训练过程。
使用第三方数据集训练模型更加具有现实意义但是更具挑战，这也是我们在本文中主要关注的场景。
**Attackers' Knowledge and Capabilities:** 我们的攻击是基于黑盒的场景设计的，我们假设攻击者可以通过访问训练数据来生成中毒样本，但是无法获得模型的任何信息，也无法控制模型的训练过程，但攻击者可以基于一小部分训练数据训练一个代理模型来提取相关数据的latent features from intermeidate layers。攻击者无法通过修改受害者模型结构和训练方式来直接嵌入后门。在推理阶段，攻击者只能向受害者模型输入中毒的测试样本，无法获取受害者模型的输出。
**Attackers' Goal:** 攻击者的目标是在不影响模型性能的前提下，在模型中植入后门。总得来说，可以概括为如下三个目标：
1. **Effectiveness:** 首要目标是实现高攻击成功率，希望模型将所有中毒样本分类为攻击者指定的目标标签，无论样本来自哪个源类别。攻击成功率可以用下式来定义：
    $$
    \begin{equation}
    \left\{
        \begin{array}{l}
        ASR = \frac{1}{\lvert D_{poi}\rvert}I(f(x_i^{\prime}; \theta^\prime),y_t),\\
        s.t. (x_i, y_i) \in D_{poi}, y_i \neq y_t.
        \end{array}
    \right.    
    \end{equation}
    $$
其中 $I$ 是指示函数，如果 $f(x_i^{\prime}; \theta^\prime) = y_t$，则 $I=1$，否则 $I=0$。
2. **Stealthness:** 同样的，我们要求生成的中毒样本足够隐蔽，即在干净样本和中毒样本之间的差异足够小，使得人类无法察觉。我们使用Log-Spectral Distance (LSD), [Peak Signal-to-Noise Ratio (PSNR)](https://ieeexplore.ieee.org/document/5596999)来衡量两者之间的差异，定义为：
    $$
    \begin{equation}
    \left\{
        \begin{array}{l}
        LSD = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\lvert \log_{10}(\frac{X_i}{Y_i})\rvert},\\
        PSNR = 10 \log_{10} \left(\frac{\max{(x_i)^2}}{MSE}\right),\\
        MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i^{\prime} - x_i)^2.
        \end{array}
    \right.
    \end{equation}
    $$
除此以外，我们还是用Perceptual Evaluation of Speech Quality (PESQ)模型来评估中毒样本和原始样本的质量差异。
3. **Robustness:** 我们将鲁棒性定义为两方面，第一方面我们希望模型中嵌入的后门具有一定的鲁棒性，即使受害者模型经过一些经典后门防御方法，例如Pruning, Fine-tuning等后，面对中毒样本，仍然能够正确触发，输出攻击者指定的目标标签。另一方面，对于生成的中毒样本，能够抵抗一些常用的数据增强方法，例如压缩、滤波等。
## IV. Proposed 
