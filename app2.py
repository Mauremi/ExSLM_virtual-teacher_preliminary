import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice
from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from api_request_schema import api_request_list, get_model_ids
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import re
import torch

import logging

model_id = os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in get_model_ids():
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

voiceIndex = 0
voiceList = ["zh-CN", "en-US", "ja-JP", "ko-KR"]
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']
voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Seoyeon']
voicePromptList = ['Chinese', 'English', 'Japanese', 'Korean']
api_request = api_request_list[model_id]
config = {
    'log_level': 'none',
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': voiceLanguageList[voiceIndex],
        'VoiceId': voiceNameList[voiceIndex],
        'OutputFormat': 'pcm',
    },
    'bedrock': {
        'api_request': api_request
    }
}

p = pyaudio.PyAudio()
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])
polly = boto3.client('polly', region_name=config['region'])
transcribe_streaming = TranscribeStreamingClient(region=config['region'])
tfidf_vectorizer = TfidfVectorizer()

n = 0
selection_list = ['noPrompt', 'prompt', 'rag', 'memory', 'multiModel', 'Chinese', 'English']
selection = selection_list[n]

use_rag = False
embedding_model_name_index = 2
embedding_model_list = \
    ["paraphrase-multilingual-mpnet-base-v2", "Alibaba-NLP/gte-multilingual-base", "BAAI/bge-large-zh-v1.5",
     "BAAI/bge-base-zh-v1.5", "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity", "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity"]
# 模型一定要支持sentence-transformers, transformers工具，大小不能太大1G

test = True

embedding_model_name = embedding_model_list[embedding_model_name_index]

rag = None

keyboardInterrupt = True

if keyboardInterrupt:
    import keyboard

def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)


# 语音交互的输入管理（用于暂停输入）
class UserInputManager:
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = False
        raise Exception()  # Workaround to shutdown exec, as executor.shutdown() doesn't work as expected.

    @staticmethod
    def start_user_input_loop():
        if keyboardInterrupt:
            def shutdown_executor_true():
                print()
                print('You: ', end='')
                printer(f'[DEBUG] User input to shut down executor...', 'debug')
                UserInputManager.shutdown_executor = True
            while True:
                keyboard.add_hotkey('alt+i', shutdown_executor_true)

        else:
            while True:
                sys.stdin.readline().strip()
                printer(f'[DEBUG] User input to shut down executor...', 'debug')
                UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor


class BedrockModelsWrapper:
    # 请求体
    @staticmethod
    def define_body(text, imagelist = None):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'anthropic':
            if "claude-3" in model_id:
                import claude3_prompts as cp

                if imagelist:
                    add = [
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": image['type'],
                                            "data": image['data'],
                                        },
                                    }
                                    for image in imagelist
                                ],
                                {
                                    "type": "text",
                                    "text": text,
                                },
                            ],
                        }
                    ]
                else:
                    add = [
                        {"role": "user", "content": text}
                    ]
                global selection
                body['messages'] = cp.prompt_list[selection]['messages'] + add
                body['system'] = cp.prompt_list[selection]['system']

                if test:
                    print(body['system'] + text, '\n\n')
            else:
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'

        return body

        # 从流式响应的事件（event）中提取数据块（chunk）

    # 在下文 to_audio_generator 函数中被调用，用于提取数据块生成音频
    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = json.loads(chunk.get('bytes').decode())
        text = ''
        if model_provider == 'amazon':
            text = chunk_obj['outputText']
        elif model_provider == 'meta':
            text = chunk_obj['generation']
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                if chunk_obj['type'] == 'message_delta':
                    print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                    print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                    print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")

                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        text = chunk_obj['delta']['text']
            else:
                text = chunk_obj['completion']
        elif model_provider == 'cohere':
            text = ' '.join([c["text"] for c in chunk_obj['generations']])
        elif model_provider == 'mistral':
            text = chunk_obj['outputs'][0]['text']
        else:
            raise NotImplementedError('Unknown model provider.')

        printer(f'[DEBUG] {chunk_obj}', 'debug')
        return text


# 语音交互中用于生成音频
def to_audio_generator(bedrock_stream):
    prefix = ''
    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)
                if '.' in text:
                    a = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    print(to_polly, flush=True, end='')
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])
        if prefix != '':
            if any('\u4e00' <= char <= '\u9fff' for char in prefix):
                yield f'{prefix}。'
            else:
                yield f'{prefix}.'
            print(prefix, flush=True, end='')
        print('\n')


class BedrockWrapper:
    def __init__(self):
        self.speaking = False
        self.conversation_history = []
        self.text_history = []

    # 检查是否正在生成音频
    def is_speaking(self):
        return self.speaking

    # 用于语音交互
    def invoke_bedrock_voice(self, text):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        # 目前用不到的
        '''if use_rag:
            # Retrieve relevant context using RAG
            rag=Rag('paraphrase-multilingual-mpnet-base-v2',text)
            relevant_text = rag.retrieve_user_input_concise()
            context = relevant_text
        else:
            context = None'''

        history = "\n".join(self.conversation_history)

        texts_with_prompt = [
            f"""<ChinesePrompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，像一个人一样，先用“我觉得”等句型简单阐述对这个问题的思考，然后以对话的口吻回答这个问题。
回答的时候，尽量分点。可以适当加入限制语使答案更加准确，如“最终”“到最后”“在这一段时间内”。但是不要加入过多的限制语，使得答案变得过于绝对。
输出的语言与用户输入的语言相同。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{text}</question>。
</ChinesePrompt>""",

            f"""<EnglishPrompt>
This is the recent chat record of the user (from the farthest to the most recent, the last message is the most recent interaction) <memories>{'\n' + history}</memories>
The user's question is most likely the question asked in the most recent chat or a new question, and it may be a question that spans multiple conversations.
When outputting, first express your thoughts on this question in simple terms, such as "I think", and then answer this question in a conversational tone.
When answering, try to divide the answer into points. You can appropriately add limiting words to make the answer more accurate, such as "finally", "in the end", "within this period". But do not add too many limiting words, making the answer too absolute.
The language of the output is the same as the language of the user input.
All generated content is output in markdown syntax.
Do not mention the above content when outputting. Please answer the question:
<question>{text}</question>.
</EnglishPrompt>"""
        ]

        text_with_prompt = texts_with_prompt[voiceIndex]

        # Update conversation history
        self.conversation_history.append(text)
        if len(self.conversation_history) > 3:
            self.conversation_history.pop(0)

        body = BedrockModelsWrapper.define_body(text_with_prompt)

        try:

            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            bedrock_stream = response.get('body')
            printer(f"[DEBUG] Bedrock_stream: {bedrock_stream}", 'debug')

            audio_gen = to_audio_generator(bedrock_stream)
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            reader = Reader()
            for audio in audio_gen:
                reader.read(audio)

            reader.close()

        except Exception as e:
            print(e)
            # time.sleep(2)
            self.speaking = False

        # time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')

    # 用于文本交互
    async def invoke_bedrock_text(self, input_text, imagelist = None):

        def getAnswer(body, printAnswer = False):
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            answer = ''

            bedrock_stream = response.get('body')
            for event in bedrock_stream:
                chunk = event.get('chunk')
                if chunk:
                    text = BedrockModelsWrapper.get_stream_text(chunk)
                    if printAnswer:
                        print(text, end='', flush=True)
                    answer += text

            if printAnswer:
                print('\n')

            return answer

        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        if use_rag:
            # Retrieve relevant context using RAG
            relevant_t_text = rag.retrieve_user_input_concise(input_text)
            context = relevant_t_text
        else:
            context = None

        t_history = "\n".join(self.text_history)

        texts_with_prompt = [
            f"""<prompt>
{input_text}
</prompt>""",

            f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + t_history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，先在<think></think>反映对这个问题的思考过程，最后再用<output></output>输出上述问题的答案。
思考时，分点并有条理地思考，尽量呈现所有的思考过程；回答时，尽量分点回答，可以简练一点，但是要涵盖所有的问题答案。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{input_text}</question>。
</prompt>""",

            f"""<prompt>
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
</prompt>""",

            f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）: <memories>{'\n' + t_history}</memories>
用户的问题虽然最可能是对最近一次的聊天提出的问题或者新提出的问题，也可能是跨过多个对话的问题。
输出时，用<output></output>输出上述问题的答案。答案尽量简要。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>{input_text}</question>。
</prompt>""",

            f"""<prompt>
这是最近用户的聊天记录（顺序由最远到最近，最后一条消息是最近一次的互动）<memories>{'\n' + t_history}</memories>
用户的问题最可能是对最近一次的聊天提出的问题或者新提出的问题，然后可能是跨过多个对话的问题。
输出时，先在<think></think>反映对这个问题的思考过程，最后再用<output></output>输出上述问题的答案。
思考时，分点并有条理地思考，尽量呈现所有的思考过程；回答时，尽量分点回答，可以简练一点，但是要涵盖所有的问题答案。
所有生成内容，均用markdown语法输出。
输出时不要提及上述内容。请回答问题：
<question>看清楚要求，再回答下面的问题：{input_text}</question>。
</prompt>"""
            ]

        #print('<context> \n' + context + '\n </context>\n\n')

        text_with_prompt = texts_with_prompt[n]
        #print(text_with_prompt)

        body = BedrockModelsWrapper.define_body(text_with_prompt, imagelist = imagelist)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            if n == 2:
                answer = getAnswer(body, printAnswer=test)

                print('\n')

                text_with_prompt += \
                    (f"\n针对上面所有的提示词，有一个模型对这个的解答是：<prevOutput>{answer}</prevOutput>"
                    f"\n这个回答的每一个分句都有可能存在错误。不要盲目轻信他的结果；如果自身知识库没有相关知识，请直接质疑这个回答无中生有。"
                    f"在鉴别的时候，你也要遵守<prompt>内的内容，并逐句非常仔细地甄别基于这个<prompt>，这个回答的每一个逻辑和答案，"
                    f"一定要有甄别对错的每一个判断依据和思考过程，但这一过程不要输出。"
                    f"修正结束后，按照之前的答案修改后，重新整理一个<output>输出出来。不用输出你的思考过程<think>"
                    f"注意<output>里面，不要出现纠错的过程，直接把自己当成那个模型，呈现修改后的给用户看的输出。"
                    # f"最终呈现出来的回答中，不用显示<correct></correct>标签及其中的内容。"
                    # f"只是在回答的最后加上一句“已进行修改检查！”"
                    )

                body = BedrockModelsWrapper.define_body(text_with_prompt, imagelist=imagelist)
                printer(f"[DEBUG] Request body: {body}", 'debug')

            answer = getAnswer(body, printAnswer=True)

            # Update conversation history
            if(voiceIndex == 0):
                self.text_history.append('用户输入：<input>' + input_text + '</input>\n' + '模型回答：<output>' + answer + '</output>')
            else:
                self.text_history.append('User: <input>' + input_text + '</input>\nModel: <output>' + answer + '</output>')
            if len(self.text_history) > 3:
                self.text_history.pop(0)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.speaking = False
            printer('[DEBUG] Bedrock generation completed', 'debug')


# 文本转语音并播放
class Reader:

    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])
        self.audio = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        self.chunk = 1024

    def read(self, data):
        response = self.polly.synthesize_speech(
            Text=data,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )

        stream = response['AudioStream']

        while True:
            # Check if user signaled to shutdown Bedrock speech
            # UserInputManager.start_shutdown_executor() will raise Exception. If not ideas but is functional.
            if UserInputManager.is_executor_set() and UserInputManager.is_shutdown_scheduled():
                UserInputManager.start_shutdown_executor()

            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()


# 语音转文本，传递给Bedrock模型生成回复
class EventHandler(TranscriptResultStreamHandler):
    text = []
    last_time = 0
    sample_count = 0
    max_sample_counter = 4
    history = []

    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper, loop, exit_signal = False):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper
        self.exit_signal = exit_signal
        self.loop = loop

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if not self.bedrock_wrapper.is_speaking():
            if results:
                for result in results:
                    EventHandler.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            print(alt.transcript, flush=True, end=' ')
                            EventHandler.text.append(alt.transcript)

            else:
                EventHandler.sample_count += 1
                if EventHandler.sample_count == EventHandler.max_sample_counter:
                    if len(EventHandler.text) != 0:
                        if (''.join(EventHandler.text) == 'Exit' or ''.join(EventHandler.text) == 'Exit.'
                            or ''.join(EventHandler.text) == 'Quit' or ''.join(EventHandler.text) == 'Quit.'
                            or ''.join(EventHandler.text) == '退出' or ''.join(EventHandler.text) == '退出。'):
                            print("Exiting...")
                            UserInputManager.start_shutdown_executor()
                            self.exit_signal = True
                            return
                        input_text = ''.join(EventHandler.text)
                        printer(f'\n[INFO] User input: {input_text}', 'info')

                        # executor = ThreadPoolExecutor(max_workers=1)
                        # Add executor so Bedrock execution can be shut down, if user input signals so.

                        '''output_text = await self.bedrock_wrapper.invoke_bedrock_voice(input_text)
                        EventHandler.history.append((input_text, output_text))
                        if len(EventHandler.history) > 10:
                            EventHandler.history.pop(0)'''

                        executor = ThreadPoolExecutor(max_workers=1)
                        UserInputManager.set_executor(executor)
                        self.loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.invoke_bedrock_voice,
                            input_text
                        )

                    EventHandler.text.clear()
                    EventHandler.sample_count = 0


# 从麦克风捕获音频流，发送到Amazon Transcribe转录
class MicStream:

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=callback, blocksize=2048 * 2, dtype="int16")
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    async def basic_transcribe(self, loop):
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputManager.start_user_input_loop)
        stream = await transcribe_streaming.start_stream_transcription(
                language_code=voiceList[voiceIndex],
                media_sample_rate_hz=16000,
                media_encoding="pcm",
        )
        handler = EventHandler(stream.output_stream, BedrockWrapper(), loop)
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())


# 文字交互
async def start_text_interaction():
    bedrock_wrapper = BedrockWrapper()
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        await bedrock_wrapper.invoke_bedrock_text(user_input)

async def start_text_interaction_with_claude_picture():

    import base64

    bedrock_wrapper = BedrockWrapper()
    while True:
        imagelist = []

        user_input = input('You: ').strip()

        # 查找所有 <picture></picture> 标签
        matches = re.findall(r'<picture>.*?</picture>', user_input)
        contents = re.findall(r'<picture>(.*?)</picture>', user_input)

        i = 0

        # 逐个替换
        for match, content in zip(matches, contents):
            path_image = r'graph/' + content

            image = dict()

            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            image['data'] = encode_image(path_image)

            image['type'] = "image/" + content.split(".")[-1]

            imagelist.append(image)

            user_input = user_input.replace(match, '<image> image ' + str(i) + ' </image>')

            i += 1

        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        await bedrock_wrapper.invoke_bedrock_text(user_input, imagelist=imagelist)

class Rag:

    def __init__(self, embedding_model_name):
        with open('ragTheDreamOfRedMansion.txt', 'r', encoding='utf-8') as file:
            self.data = file.read().split('\n')

        self.embedding_model_name = embedding_model_name

        if (self.embedding_model_name == 'Alibaba-NLP/gte-multilingual-base'
                or self.embedding_model_name == 'BAAI/bge-large-zh-v1.5'
                or self.embedding_model_name == "BAAI/bge-base-zh-v1.5"):
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cuda', trust_remote_code=True)
            self.embeddings = self.embedding_model.encode(self.data, normalize_embeddings=True)
        elif self.embedding_model_name == 'paraphrase-multilingual-mpnet-base-v2':
            self.embedding_model = BertForMaskedLM.from_pretrained(embedding_model_name)
            self.embeddings = np.array(self.embedding_model.encode(self.data), dtype='float32')
        elif (self.embedding_model_name == "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity" or
              self.embedding_model_name == 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'):
            self.tokenizer = BertTokenizer.from_pretrained(self.embedding_model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.embedding_model_name)

        def find_chapter_indexes(data):
            pattern = r"\d+"
            matches = [d for d in data if re.search(pattern, d)]
            indexes = [data.index(match) for match in matches]
            return indexes

        self.chapter_indices = (find_chapter_indexes(self.data))
        self.chapter_indices.append(len(self.data))

    def retrieve_user_input_concise(self, input_text):
        retrieved_text = ''
        max_indexes = []
        max_index = 0
        pieced = True

        if (self.embedding_model_name == 'Alibaba-NLP/gte-multilingual-base'
                or self.embedding_model_name == 'BAAI/bge-large-zh-v1.5'
                or self.embedding_model_name == "BAAI/bge-base-zh-v1.5"):
            if pieced:
                pieces = 3
                while (len(input_text) - 6) % pieces != 0:
                    input_text += ' '
                input_text += ' '
                input_text_pieces = [input_text[pieces * i: pieces * (i + 1) + 6] for i in range((len(input_text) - 6) // pieces)]
                input_text_pieces.append(input_text)
                if test:
                    print(input_text_pieces)
                input_embeddings = [self.embedding_model.encode(input_text_piece, normalize_embeddings=True) for input_text_piece in input_text_pieces]
                scores_list = [self.embedding_model.similarity(input_embedding, self.embeddings) for input_embedding in input_embeddings]
                if test:
                    print(scores_list)
                max_indexes = [int(np.argmax(scores)) for scores in scores_list]
                max_indexes = list(set(max_indexes))  # 去重
                if test:
                    print(max_indexes)
            else:
                input_embedding = self.embedding_model.encode([input_text], normalize_embeddings=True)
                scores = self.embedding_model.similarity(input_embedding, self.embeddings)
                max_index = np.argmax(scores)

        elif self.embedding_model_name == 'paraphrase-multilingual-mpnet-base-v2':
            input_embedding = np.array(self.embedding_model.encode([input_text]), dtype='float32')

            # FAISS similarity search
            index = faiss.IndexFlatL2(self.embeddings.shape[1])
            index.add(self.embeddings)
            _, indices = index.search(input_embedding, 1)
            faiss_index = indices[0][0]

            # cosine similarity search
            similarities = cosine_similarity(input_embedding, self.embeddings)
            max_sim_index = np.argmax(similarities)
            cosine_index = max_sim_index

            # TF-IDF similarity search
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.data)
            input_tfidf = tfidf_vectorizer.transform([input_text])
            cosine_similarities = (tfidf_matrix * input_tfidf.T).toarray()
            tfidf_index = np.argmax(cosine_similarities)

            options = [faiss_index, cosine_index, tfidf_index]
            max_index = options[0]

        elif (self.embedding_model_name == "IDEA-CCNL/Erlangshen-Roberta-110M-Similarity" or
              self.embedding_model_name == 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'):
            if pieced:
                pieces = 6
                while (len(input_text) - 6) % pieces != 0:
                    input_text += ' '
                input_text += ' '
                input_text_pieces = [input_text[pieces * i: pieces * (i + 1) + 6] for i in
                                     range((pieces - 6) // pieces)]
                input_text_pieces.append(input_text)
                print(input_text_pieces)
                inputs_list = [
                    [torch.tensor([self.tokenizer.encode(input_text_piece, sentence)]) for sentence in self.data] for
                    input_text_piece in input_text_pieces]
                similarities_list = [[torch.nn.functional.softmax(self.model(input).logits, dim=-1) for input in inputs]
                                     for inputs in inputs_list]
                max_indexes = [np.argmax(similarities) for similarities in similarities_list]
                print(max_indexes)
                max_indexes = list(set(max_indexes))  # 去重
                print(max_indexes)
            else:
                # 批量编码输入文本与候选句子
                encoded_inputs = self.tokenizer(
                    [input_text] * len(self.data),
                    self.data,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # 模型推理
                with torch.no_grad():
                    outputs = self.model(**encoded_inputs)

                # 计算相似度
                similarities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                max_index = torch.argmax(similarities).item()


        #print('<context>\n' + self.data[max_index] + '\n</context>\n\n')
        # 通过 max_index 返回整个章节
        def find_max_chapter_index(max_index):
            max_chapter_index = 0
            for chapter_index in self.chapter_indices:
                if max_index < chapter_index:
                    max_chapter_index = self.chapter_indices.index(chapter_index) - 1
                    break
            return max_chapter_index

        if max_indexes:
            max_chapter_indexes = [find_max_chapter_index(max_index) for max_index in max_indexes]
            max_chapter_indexes = list(set(max_chapter_indexes))
            retrieved_texts = []
            for max_chapter_index in max_chapter_indexes:
                retrieved_texts += self.data[self.chapter_indices[max_chapter_index]:self.chapter_indices[max_chapter_index + 1] - 1]
            retrieved_texts = list(set(retrieved_texts))
            retrieved_text = ''.join(retrieved_texts)
        else:
            max_chapter_index = find_max_chapter_index(max_index)
            retrieved_texts = self.data[self.chapter_indices[max_chapter_index]:self.chapter_indices[max_chapter_index + 1] - 1]
            retrieved_text = ''.join(retrieved_texts)


        return retrieved_text

# 原来的多模态
"""
class multiModal():

    def __init__(self, pic):
        self.pic = pic
        self.langs = ["en"]  # Replace with your languages or pass None (recommended to use None)
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    # opencv识别文字
    def cv(self):
        # 读取图像
        image = cv2.imread(self.pic)
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用 Tesseract 进行 OCR
        text = pytesseract.image_to_string(gray_image, lang='chi_sim')
        print("raw text:", text)
        # 去除换行符并合并成一个字符串
        cleaned_text = " ".join(text.splitlines())
        print("cleaned text:", cleaned_text)


    def surya_rec(self):
        image = Image.open(self.pic)
        predictions = self.recognition_predictor([image], [self.langs], self.detection_predictor)
        texts = [line.text for line in predictions[0].text_lines]
        combined_texts = " ".join(texts)
        return combined_texts
"""

class App:
    def __init__(self):
        self.menu = {
            "0": "提示词微调",
            "1": "提示词设计",
            "2": "RAG",
            "3": "上下文记忆",
            "4": "多模态",
            "5": "多语言对话",
            "-1": "退出",
            "-2": "重启"
        }
        self.loop = asyncio.new_event_loop()

    def display_menu(self):
        print("*************************************************************\n")
        for key, value in self.menu.items():
            print(f"{key}: {value}")

    @staticmethod
    def rewrite_config():
        global config
        config = {
            'log_level': 'none',
            'region': aws_region,
            'polly': {
                'Engine': 'neural',
                'LanguageCode': voiceLanguageList[voiceIndex],
                'VoiceId': voiceNameList[voiceIndex],
                'OutputFormat': 'pcm',
            },
            'bedrock': {
                'api_request': api_request
            }
        }

    @staticmethod
    def get_selection():
        nn = int(input("Enter the number of the selection: "))
        return nn

    def cancel_all_tasks(self):
        # 获取所有未完成的任务
        tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done()]

        # 取消所有任务
        for task in tasks:
            task.cancel()

        # 等待所有任务正式取消
        if tasks:
            self.loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=True)
            )

    def run(self):
        global use_rag
        if use_rag:
            global rag
            rag = Rag(embedding_model_name)
        while True:
            self.display_menu()
            global n, selection
            try:
                n = self.get_selection()
                if (n != -1) and (n != -2):
                    selection = selection_list[n]
            except:
                print("[Error] invalid input.")
                continue
            if test:
                print(selection)
            if (n == 1) or (n == 3) or (n == 4) or (n == 0):
                use_rag = False
                info_text = f'''
*************************************************************
[INFO] Supported FM models: {get_model_ids()}.
[INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
[INFO] Log level: {config['log_level']}

[INFO] Type "exit" or "quit" to end the conversation.
[INFO] Type "<picture></picture>" to cite the picture.
*************************************************************
                            '''
                print(info_text)
                asyncio.run(start_text_interaction_with_claude_picture())
                '''if m == '0':
                    asyncio.run(start_text_interaction())
                else:
                    print('Invalid Input')'''
            elif n == 2:
                use_rag = True
                info_text = f'''
*************************************************************
[INFO] Supported FM models: {get_model_ids()}.
[INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
[INFO] Log level: {config['log_level']}

[INFO] Type "exit" or "quit" to end the conversation.
[INFO] Type "<picture> relative path to your picture </picture>" to read the picture.

[INFO] Embedding Model: {embedding_model_name}
*************************************************************'''
                print(info_text)
                asyncio.run(start_text_interaction())
                '''if m == '0':
                    asyncio.run(start_text_interaction())
                else:
                    print('Invalid Input')'''
            elif n == 5:
                global voiceIndex
                voiceIndex = int(input("Enter the voice index (0 for Chinese, 1 for English): "))
                if voiceIndex == 1:
                    selection = selection_list[6]
                    n = 6
                self.rewrite_config()

                use_rag = False
                info_text = f'''
*************************************************************
[INFO] Supported FM models: {get_model_ids()}.
[INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
[INFO] Polly config: engine {config['polly']['Engine']}, voice {config['polly']['VoiceId']}
[INFO] Log level: {config['log_level']}

[INFO] Hit ENTER to interrupt Amazon Bedrock. After you can continue speaking!
[INFO] Go ahead with the voice chat with Amazon Bedrock!
*************************************************************
                            '''
                asyncio.set_event_loop(self.loop)
                print(info_text)
                try:
                    self.loop.run_until_complete(MicStream().basic_transcribe(self.loop))
                except (KeyboardInterrupt, Exception) as e:
                    print()

                self.cancel_all_tasks()
                self.loop.stop()
                self.loop.close()
                self.loop = None
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

            elif n == -1:
                print("Exiting...")
                break
            else:
                print("Restarting...")
                time.sleep(2)
                os.system('cls')


if __name__ == "__main__":
    app = App()
    app.run()