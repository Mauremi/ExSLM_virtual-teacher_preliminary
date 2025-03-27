这是上海交通大学2024级新生杯的参赛项目。
我们开发了一个基于 Claude-3 模型的教师大模型，
用于阅读理解、解题教学等任务。
我们推荐使用 PyCharm 来审阅甚至运行这个项目。
项目的使用和基本信息如下：

# 作者信息
*排名不分先后。*

- 涂瑞峰：上海交通大学电子信息与电气工程学院2024级本科生。
- 陈欣怡：上海交通大学巴黎卓越工程师学院2024级本科生。
- 房郁超：上海交通大学电子信息与电气工程学院2024级本科生。

# 更新讯息
[20250327] Chinese texts can be generated sentence by sentence in bilingual mode.
README.md added.
fine_tunning_data.py deleted.
Requirements added.

[20250326] Reduced the loading time of RAG embedding model.
Added two new models of Erlangshen series to model list.
Added pieced logic to input_text.

[20250321] Audio fixed.

[20250320] class APP.
menu selection.
prompts updated and optimized.
asyncio and loop fixed.
prompts optimized.

[20250319] Added embedding model: BAAI/bge-base-zh-v1.5
Optimized embedding model selection.
Modified model based on the problem order of the preliminary.
Simplified the file "claude3_promps.py".


# 文件说明
- 文件夹 graph/ ：用以存储在多模态中使用的图片。 文件夹中提供了两个涉及数学问题的图片。若用户有其他图片，请拖入这个文件夹中使用。

- app2.py：主程序，包含所有功能的所有代码。启动项目时，运行这个文件即可。

- api_request_schema.py：这是亚马逊支持的模型列表以及参数设置的配置文件，一般勿动。

- claude3_prompts.py：用来存储基于 Claude-3 模型的系统提示词。

- ragTheDreamOfRedMansion.txt：是RAG模式使用的文本，给定的《红楼梦》文本。

- requirements_for_macmps.txt：Mac系统下的库依赖文件。

- requirements_for_wingpu.txt：Windows系统下的库依赖文件。

- README.md：本文件。

# 使用方法
## 安装
在创建的python虚拟环境的基础上，我们可以通过以下命令安装所用的库：

1. Windows 系统（支持keyboard库和Cuda）
```shell
pip install -r "requirements_for_wingpu.txt"
```
2. Mac 系统
```bash
pip install -r "requirements_for_macmps.txt"
```

## 运行
在安装好所需的库之后，我们可以通过以下命令运行模型：

```shell
python app2.py
```

或者也可以通过其他方式运行`app.py`文件。

## 超参数
项目中有可供修改的超参数`keyboardInterrupt`、`test`、`use_rag`、`embedding_model_name_index`可供修改：
1. `keyboardInterrupt`表示是否可以在语音输出的时候用快捷键`alt+i`来打断输出（mac 不支持，请设为`False`）；
2. `test`是用来调试模型输出时内部使用的超参数，若`True`，可以显示给模型的提示词、对问题的分解等中间参数，便于调试；用户端用`False`，则只显示标准输出；
3. `use_rag`默认是`True`，表示要用`embedding_model`来处理RAG文本检索问题；如果用`False`，则切换到RAG模式会报错，但会节省最开始设置`embedding_model`的时间；
4. `embedding_model_name_index`是用来切换所用的`embedding_model`的，目前按0~5的顺序支持"paraphrase-multilingual-mpnet-base-v2", "Alibaba-NLP/gte-multilingual-base", "BAAI/bge-large-zh-v1.5", "BAAI/bge-base-zh-v1.5", "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity", "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity"模型。

## 客户端
项目内容是完全基于新生杯的具体比赛项目（项目赛题不在本项目中）的。

进入项目后出现菜单栏如下：
> \*************************************************************
> 
> 0: 提示词微调 
> 
> 1: 提示词设计
> 
> 2: RAG 
> 
> 3: 上下文记忆
> 
> 4: 多模态
> 
> 5: 多语言对话
> 
> -1: 退出
> 
> -2: 重启
> 
> Enter the number of the selection: 

用户可以通过输入数字来选择对应的功能，然后按照提示进行操作。

### 0:提示词微调
按照评委所给的句子，直接输入，获得结果。这一部分没有任何提示词模板。

### 1:提示词设计
按照评委所给的问题，自己设计提示词输入，获得结果。这一部分的提示词模板为：
`f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + t_history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，先在<think></think>反映对这个问题的思考过程，最后再用<output></output>输出上述问题的答案。
思考时，分点并有条理地思考，尽量呈现所有的思考过程；回答时，尽量分点回答，可以简练一点，但是要涵盖所有的问题答案。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{input_text}</question>。
</prompt>"""`

其中`t_hisory`是用户与模型的交互历史，`input_text`是本次用户输入。

### 2:RAG
RAG的考察基于给定的《红楼梦》文本，存储在`ragTheDreamOfRedMansion.txt`内。

基于输入，嵌入模型`embedding_model`先会比较其相似度，匹配出最相似的上下文，然后再用 Claude-3 模型来回答问题。

我们给予 Claude-3 的系统提示词为：
`"<system>\n你现在是一个知识丰富的助手，擅长回答中国古典文学相关问题。请尽量详细地回答用户的问题，详细剖析这个问题的相关回答，并使用通俗易懂的语言。结果用markdown语法输出。\n</system>\n\n"`

这一部分的生成逻辑和提示词为：
`f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）: <memories>{'\n' + t_history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
这是检索到的相关知识: <context>{context}</context>
检索相关知识的内容可能偏少，所以如果匹配成功了相关知识，也不要轻信，可能需要结合自己的知识进行回答。但不要在下面的<think></think>中提及，假装就是通过提供的上下文得到的。
用户的问题陈述不一定正确，请仔细甄别，并可以结合自己的知识，若用户的问题前提有错误，请一定指出。
回答问题时，尽量少做延伸，尽量直接回答这个问题（如是或否），而不要做延伸，比如解释这个问题真正指向的人物。
回答时，可以适当加入限制语使答案更加准确，如“最终”“到最后”“在这一段时间内”。但是不要加入过多的限制语，使得答案变得过于绝对。
检索相关知识的位置是最优的位置，如果在给出的上下文内没有找到相关话题，说明这个问题是一定没有答案的，而不要怀疑还存在其他相关内容，更无需确认其他相关情节！
输出时，先在<think></think>反映对这个问题的思考过程，最后再用<output></output>输出上述问题的答案。
思考时，分点并有条理地思考，尽量呈现所有的思考过程；回答时，尽量分点回答，可以简练一点，但是要涵盖所有的问题答案。
不要说“根据上下文”，而说“根据原著”。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{input_text}</question>。
</prompt>"""`

其中`context`是嵌入模型匹配到的上下文，`t_history`是用户与模型的交互历史，`input_text`是本次用户输入。

基于这个提示词生成的第一遍输出，模型又吸收了第一遍输出，并通过下面的提示词生成了最终的输出：
`f"\n针对上面所有的提示词，有一个模型对这个的解答是：<prevOutput>{answer}</prevOutput>"
f"\n这个回答的每一个分句都有可能存在错误。不要盲目轻信他的结果；如果自身知识库没有相关知识，请直接质疑这个回答无中生有。"
f"在鉴别的时候，你也要遵守<prompt>内的内容，并逐句非常仔细地甄别基于这个<prompt>，这个回答的每一个逻辑和答案，"
f"一定要有甄别对错的每一个判断依据和思考过程，但这一过程不要输出。"
f"修正结束后，按照之前的答案修改后，重新整理一个<output>输出出来。不用输出你的思考过程<think>"
f"注意<output>里面，不要出现纠错的过程，直接把自己当成那个模型，呈现修改后的给用户看的输出。`

其中`answer`是上一轮输出。

其他的生成逻辑见后代码讲解。

### 3:上下文记忆
上下文记忆的考察比较简单，只需要记忆即可。（其他的文本交互以及语音交互基本都有记忆功能。）

这个部分的系统提示词为：
`"<system>\n你现在是一个记忆助手，可以帮助用户记住信息。如果用户提问，基于用户之前传达的信息，使用通俗易懂的语言回答。结果用markdown语法输出，涉及到数学公式，用$以及$$符号括处，用LaTeX语法输出。涉及到代码块，用`或```括出。\n</system>\n\n"`

提示词为：
`f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）: <memories>{'\n' + t_history}</memories>
用户的问题虽然最可能是对最近一次的聊天提出的问题或者新提出的问题，也可能是跨过多个对话的问题。
输出时，用<output></output>输出上述问题的答案。答案尽量简要。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{input_text}</question>。
</prompt>"""`

这一部分的功能可以完美地通过其他部分实现。

### 4:多模态
这一部分支持图片的输入（实际上，0,1,3,4部分都支持图片）。

使用图片的方法是，将需要上传的图片拖入 graph 文件夹中。
在输入时，需要插入图片的地方（可以在文中任意地点），用`<picture></picture>`标签将需要输入的图片名字（包括扩展名）输入其中。
最终给予 Claude-3 的输入会记住图片在文中的位置。

这一部分的系统提示词为：
`"<system>\n你现在是一个通用助手，可以回答各种问题。请尽量详细地回答用户的问题。结果用markdown语法输出，涉及到数学公式，用$以及$$符号括处，用LaTeX语法输出。涉及到代码块，用`或```括出。\n</system>\n\n"`

提示词为：
`f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + t_history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，先在<think></think>反映对这个问题的思考过程，最后再用<output></output>输出上述问题的答案。
思考时，分点并有条理地思考，尽量呈现所有的思考过程；回答时，尽量分点回答，可以简练一点，但是要涵盖所有的问题答案。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>看清楚要求，再回答下面的问题：{input_text}</question>。
</prompt>"""`

这一部分的功能可以完美地通过其他除RAG外的文本交互功能实现。

### 5:多语言对话
进入这一部分后，会提示选择中文或英文。选择语言后，直接与模型对话即可。

用户在模型正在输出时，可以随时用键`Enter`来打断模型的输出，直接进入输入。
如果支持`keyboardInterrupt`，则可以用`alt+i`快捷键完成中断输出。

该部分的系统提示词为：
`"<system>\n你现在是一个通用助手，可以回答各种问题。请尽量详细地回答用户的问题。结果用markdown语法输出。因为你的回答是会被机器读出来的，请确保你的回答里面没有容易被读错含义的符号，尽量用文字代替。\n</system>\n\n"`
或
`"<system>\nYou are now a general assistant who can answer various questions. Please answer the user's questions in detail. The results are output in markdown syntax. Since your answer will be read out by machines, please ensure there is no signal that is easy to be misunderstood when reading, or use just words to replace the signals.\n</system>\n\n"`

提示词为：
`f"""<ChinesePrompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，像一个人一样，先用“我觉得”等句型简单阐述对这个问题的思考，然后以对话的口吻回答这个问题。
回答的时候，尽量分点。可以适当加入限制语使答案更加准确，如“最终”“到最后”“在这一段时间内”。但是不要加入过多的限制语，使得答案变得过于绝对。
输出的语言与用户输入的语言相同。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{text}</question>。
</ChinesePrompt>"""`
或
`f"""<EnglishPrompt>
This is the recent chat record of the user (from the farthest to the most recent, the last message is the most recent interaction) <memories>{'\n' + history}</memories>
The user's question is most likely the question asked in the most recent chat or a new question, and it may be a question that spans multiple conversations.
When outputting, first express your thoughts on this question in simple terms, such as "I think", and then answer this question in a conversational tone.
When answering, try to divide the answer into points. You can appropriately add limiting words to make the answer more accurate, such as "finally", "in the end", "within this period". But do not add too many limiting words, making the answer too absolute.
The language of the output is the same as the language of the user input.
All generated content is output in markdown syntax.
Do not mention the above content when outputting. Please answer the question:
<question>{text}</question>.
</EnglishPrompt>"""`

# 代码详注
## 超参数
除了用户可以直接修改的四个超参数外，还有一些超参数是固定的，如`embedding_model_name`、`embedding_model`等，相当于一些设置信息，请不要随意修改。

## `printer`函数
用于输出日志。

## `UserInputManager`类
用于管理语音交互中的输入，特别是暂停输入的功能。它通过设置和关闭执行器来控制输入流的开启和关闭。

- **`set_executor(executor)`**  
  设置执行器，用于管理输入流。

- **`start_shutdown_executor()`**  
  启动关闭执行器的流程。通过抛出异常来强制关闭执行器（由于 `executor.shutdown()` 无法按预期工作）。

- **`start_user_input_loop()`**  
  启动用户输入循环。如果检测到 `keyboardInterrupt`，则通过热键 `alt+i` 触发执行器关闭。否则，持续监听标准输入流。

- **`is_executor_set()`**  
  检查执行器是否已设置。

- **`is_shutdown_scheduled()`**  
  检查是否已计划关闭执行器。

## `BedrockModelsWrapper`类

用于封装与 Bedrock 模型交互的逻辑，包括请求体的定义和流式响应的处理。


- **`define_body(text, imagelist=None)`**  
  根据模型 ID 和提供的文本或图像列表定义请求体。

- **`get_stream_chunk(event)`**  
  从流式响应的事件中提取数据块。

- **`get_stream_text(chunk)`**  
  从数据块中提取文本内容，根据不同的模型提供者进行解析。


## `to_audio_generator`函数

用于将 Bedrock 模型的流式响应转换为音频流，并逐段生成音频。

- **`to_audio_generator(bedrock_stream)`**  
  从 Bedrock 流式响应中提取文本，生成音频流并逐段播放。


## `BedrockWrapper`类

用于封装与 Bedrock 模型的交互，支持语音和文本两种交互模式。


- **`__init__()`**  
  初始化类，设置语音生成状态、会话历史和文本历史。

- **`is_speaking()`**  
  检查是否正在生成音频。

- **`invoke_bedrock_voice(text)`**  
  通过语音交互调用 Bedrock 模型生成回复，并更新会话历史。

- **`invoke_bedrock_text(input_text, imagelist=None)`**  
  通过文本交互调用 Bedrock 模型生成回复，并更新文本历史。


## `Reader`类

用于将文本转换为语音并播放，支持通过 Amazon Polly 生成音频流。


- **`__init__()`**  
  初始化类，设置 Polly 客户端和音频输出设备。

- **`read(data)`**  
  将文本数据转换为语音并播放。

- **`close()`**  
  关闭音频流。


## `EventHandler`类
用于处理语音转录事件，将语音转换为文本并传递给 Bedrock 模型生成回复。

- **`__init__(transcript_result_stream, bedrock_wrapper, loop, exit_signal=False)`**  
  初始化类，设置转录流、Bedrock 封装类、事件循环和退出信号。

- **`handle_transcript_event(transcript_event)`**  
  处理语音转录事件，将语音转换为文本并调用 Bedrock 模型生成回复。


## `MicStream`类

用于从麦克风捕获音频流，并发送到 Amazon Transcribe 进行转录。

- **`mic_stream()`**  
  从麦克风捕获音频流。🎤

- **`write_chunks(stream)`**  
  将音频块写入转录流。

- **`basic_transcribe(loop)`**  
  启动基本转录流程，捕获音频并发送到 Amazon Transcribe。

## `start_text_interaction`函数

用于启动文本交互模式，允许用户通过输入文本与 Bedrock 模型交互。

- **`start_text_interaction()`**  
  启动文本交互循环，持续监听用户输入并调用 Bedrock 模型生成回复。


## `start_text_interaction_with_claude_picture`函数
用于启动带图片的文本交互模式，支持用户输入文本和图片与 Bedrock 模型交互。

- **`start_text_interaction_with_claude_picture()`**  
  启动带图片的文本交互循环，持续监听用户输入并调用 Bedrock 模型生成回复。


## `Rag`类

用于基于用户输入检索相关上下文，支持多种嵌入模型。

- **`__init__(embedding_model_name)`**  
  初始化类，加载数据和嵌入模型。

- **`retrieve_user_input_concise(input_text)`**  
  根据用户输入检索相关上下文，并返回最相关的文本内容。

## `App`类
用于管理整个运行的项目。

- `__init__`函数（构造函数）
  包括菜单，以及程序后续运行所用的线程池`self.loop`。

- `display_menu`函数
  用于展示菜单。

- `rewrite_config`函数
  用于在语音参数更改的情况下，修改语音的设置。

- `get_selection`函数
  用于获取用户的选项。

- `cancel_all_tasks`函数
  用来重置`loop`，取消正在进行的所有任务。

- `run`函数
  用来运行整个项目。

# 优化方向
- 语音存在二次输入会捕捉到上一次输出的语音的问题。

- RAG模型的选择，使得匹配的上下文最为准确。