import torch
import os
import argparse
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor

# run script : python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 opt_30b_en_mutigpu.py
os.environ["ENV_TYPE"] = "deepspeed+mpu"
model_parallel_size = 4
world_size = 4

os.environ["MODEL_PARALLEL_SIZE"] = str(model_parallel_size)
os.environ["WORLD_SIZE"] = str(world_size)

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

parser = argparse.ArgumentParser()
local_rank = int(os.getenv('LOCAL_RANK', '0'))

master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
master_port = os.environ.get('MASTER_PORT', '17501')

device = torch.device("cuda", local_rank)

def initialize_distributed():
    """Initialize torch.distributed."""
    torch.backends.cudnn.enabled = False
    # Manually set the device ids.
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'

    init_method += master_addr + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',  # gloo
        world_size=world_size,
        rank=local_rank,
        init_method=init_method)
    mpu.initialize_model_parallel(model_parallel_size)

initialize_distributed()

set_random_seed(123)

print(f"building model...")
loader = AutoLoader("lm", model_name="opt-1.3b-en")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.half()

model.parallel_output = False
model.eval()
model.to(device)

torch.distributed.barrier(group=mpu.get_model_parallel_group())

text = """I think The Old Man and the Sea is a very good book, what do you think? I think """

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text)
if mpu.get_model_parallel_rank() == 0:
    print(f"pred is {out}")


def calculate_outputs(model, tokenizer, device, text, tgt):
    input_ids = tokenizer.encode(text)
    tgt_ids = tokenizer.encode(tgt)[1:]
    output_ids = [-100] * len(input_ids)
    output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=output_ids,
            output_hidden_states=True
        )

    print(outputs)
    loss, logits, hidden_states = outputs['loss'], outputs['logits'], outputs['hidden_states']
    loss = loss.item()

    return loss, logits, hidden_states

loss, logits, hidden_states = calculate_outputs(model, tokenizer, device, '唐朝赋税制度的变化反映了唐朝社会经济的繁荣发展和政府对财政收入的重视程度增加。', "在春秋战国时期，各国为了打击旧的奴隶主贵族势力，建立封建政治和经济秩序，巩固新生政权，增强竞争实力，纷纷开展变法运动。")
print(-loss)
#
# loss, logits, hidden_states = calculate_outputs(model, tokenizer, device, 'A la turca restaurant is in the cheap price range. Convert the following text into another expression that preserves key information:', 'world.')
# print(-loss)
#
# loss, logits, hidden_states = calculate_outputs(model, tokenizer, device, 'A la turca restaurant is in the cheap price range. Convert the following text into another expression that preserves key information:', 'can I confirm what area you are looking for a hotel near ?')
# print(-loss)

# {'logits': tensor([[[-4.1211, -5.6836, -4.2539,  ..., -4.4766, -5.4961, -3.5820],
#          [-6.3711, -2.4121, -3.7676,  ..., -6.6055, -3.5117, -4.9531],
#          [-3.1934, -4.4922, -1.5420,  ..., -4.7852, -0.1908, -4.1953],
#          ...,
#          [-3.3945, -2.1484, -4.0000,  ..., -5.1133, -2.8555, -3.4043],
#          [-7.8906, -4.7539, -7.7578,  ..., -8.2266, -2.9980, -5.5234],
#          [-7.1523, -4.3906, -6.1133,  ..., -6.3867, -1.3779, -4.6211]]],
#        device='cuda:2', dtype=torch.float16), 'loss': tensor(9.3766, device='cuda:2'), 'hidden_states': None}
# -9.37663745880127{'logits': tensor([[[-5.6992, -5.7227,  3.3887,  ..., -3.5215, -3.5859, -1.2881],
#          [-5.8203, -5.8047,  3.1816,  ..., -1.9629, -2.2578, -2.9160],
#          [-5.4727, -5.9062, -1.4033,  ..., -5.7070, -4.4219, -3.7051],
#          ...,
#          [-3.3320, -2.8516,  5.5430,  ..., -2.5332,  0.2461,  0.7905],
#          [-6.5859, -6.8945,  2.7383,  ..., -3.8145, -3.5488, -5.1133],
#          [-6.0195, -6.2422, -0.8438,  ..., -3.8008, -3.3652, -4.1406]]],
#        device='cuda:0', dtype=torch.float16), 'loss': tensor(9.3766, device='cuda:0'), 'hidden_states': None}
#
# -9.37663745880127
# {'logits': tensor([[[-2.5098, -1.7910, -2.9453,  ..., -5.3516, -5.4453, -5.3633],
#          [-5.0859, -5.2812, -1.5088,  ..., -5.9531, -5.7578, -5.8477],
#          [-3.1465, -5.4531, -4.0195,  ..., -5.2773, -5.4844, -5.6055],
#          ...,
#          [-5.4297, -3.5820, -2.3125,  ..., -3.4141, -3.3398, -3.0918],
#          [-5.9453, -6.1406, -5.4102,  ..., -6.9180, -6.5078, -6.4766],
#          [-5.6875, -5.7188, -5.3945,  ..., -6.4219, -6.2734, -5.9375]]],
#        device='cuda:3', dtype=torch.float16), 'loss': tensor(9.3766, device='cuda:3'), 'hidden_states': None}
# -9.37663745880127
# {'logits': tensor([[[-4.1914, -2.7578, -2.8027,  ..., -1.0430, -3.7637, -3.8145],
#          [-0.1682, -2.1504,  0.1509,  ..., -2.1016, -5.9062, -3.2090],
#          [-4.2344, -3.1191, -0.7261,  ..., -3.6074, -2.4062, -4.5625],
#          ...,
#          [-2.0684, -1.2334, -1.9512,  ..., -1.0879, -2.9824, -1.2725],
#          [-3.9961, -4.7031, -0.6753,  ..., -3.2695, -4.4023, -5.4219],
#          [-3.8770, -4.3867, -0.1359,  ..., -2.9180, -3.9004, -5.6484]]],
#        device='cuda:1', dtype=torch.float16), 'loss': tensor(9.3766, device='cuda:1'), 'hidden_states': None}
# -9.37663745880127
# {'logits': tensor([[[-5.1055, -5.5977, -3.4102,  ..., -5.1133, -4.7539, -4.4922],
#          [-5.2461, -3.8965, -1.0645,  ..., -4.4766, -5.5820, -5.8359],
#          [-3.5137, -1.2979, -1.4648,  ..., -2.5410, -1.9805, -2.4102],
#          ...,
#          [-5.4648, -3.2500, -1.9336,  ..., -3.7832, -2.7188, -3.5625],
#          [-3.1699, -2.8477, -0.3269,  ..., -2.1895, -0.1003, -2.4961],
#          [-1.9023, -1.9551,  0.2732,  ..., -3.1602,  1.0996, -1.7646]]],
#        device='cuda:2', dtype=torch.float16), 'loss': tensor(1.6949, device='cuda:2'), 'hidden_states': None}{'logits': tensor([[[-6.5078, -6.5898,  2.3809,  ..., -1.6826, -4.5156, -1.6621],
#          [-4.1133, -3.8965,  1.8359,  ..., -2.4648, -6.0195, -1.9561],
#          [-0.6523, -0.4373,  3.7012,  ..., -1.7725, -3.9590,  0.1041],
#          ...,
#          [-4.0039, -3.7402,  4.4375,  ...,  0.4321,  0.1111, -1.3213],
#          [-2.5449, -2.2012,  8.6172,  ..., -1.3877, -3.0020, -3.7363],
#          [-3.2871, -3.1738,  9.0703,  ..., -1.2139, -4.0312, -1.1680]]],
#        device='cuda:0', dtype=torch.float16), 'loss': tensor(1.6949, device='cuda:0'), 'hidden_states': None}
#
# -1.6948860883712769-1.6948860883712769
#
# {'logits': tensor([[[-3.2754, -3.7324, -4.8281,  ..., -6.2188, -6.3711, -6.3164],
#          [-6.9883, -5.0938, -4.2578,  ..., -4.0312, -4.2148, -4.0156],
#          [-5.7695, -2.1406, -3.1094,  ..., -0.8188, -0.7402, -0.6943],
#          ...,
#          [-4.6953, -1.8027,  1.5508,  ..., -4.1211, -3.9844, -4.2109],
#          [-1.9365, -0.4124,  0.4866,  ..., -2.6152, -2.1562, -2.4512],
#          [-3.3711,  0.7832,  0.0413,  ..., -3.4414, -3.5059, -3.3672]]],
#        device='cuda:3', dtype=torch.float16), 'loss': tensor(1.6949, device='cuda:3'), 'hidden_states': None}
# -1.6948860883712769
# {'logits': tensor([[[-4.2539, -3.5352, -2.0918,  ..., -2.9727, -3.1465, -2.1172],
#          [-3.9863, -2.1250, -0.1639,  ..., -4.7891, -5.0586, -2.1699],
#          [-3.6816, -0.6802, -0.9971,  ..., -2.7598,  1.3135, -0.5293],
#          ...,
#          [-0.9507, -2.2012, -1.7051,  ..., -2.7910, -1.3965, -4.0156],
#          [-1.6826, -0.8643, -1.9170,  ..., -1.2920, -3.8887, -5.1289],
#          [-2.4785,  0.7139, -0.3826,  ..., -0.5679, -5.0508, -2.9004]]],
#        device='cuda:1', dtype=torch.float16), 'loss': tensor(1.6949, device='cuda:1'), 'hidden_states': None}
# -1.6948860883712769
# {'logits': tensor([[[-6.5078, -6.5898,  2.3809,  ..., -1.6826, -4.5156, -1.6621],
#          [-4.1133, -3.8965,  1.8359,  ..., -2.4648, -6.0195, -1.9561],
#          [-0.6523, -0.4373,  3.7012,  ..., -1.7725, -3.9590,  0.1041],
#          ...,
#          [-4.0039, -3.7402,  4.4375,  ...,  0.4321,  0.1111, -1.3213],
#          [-2.5449, -2.2012,  8.6172,  ..., -1.3877, -3.0020, -3.7363],
#          [-3.2871, -3.1738,  9.0703,  ..., -1.2139, -4.0312, -1.1680]]],
#        device='cuda:0', dtype=torch.float16), 'loss': tensor(5.9306, device='cuda:0'), 'hidden_states': None}{'logits': tensor([[[-5.1055, -5.5977, -3.4102,  ..., -5.1133, -4.7539, -4.4922],
#          [-5.2461, -3.8965, -1.0645,  ..., -4.4766, -5.5820, -5.8359],
#          [-3.5137, -1.2979, -1.4648,  ..., -2.5410, -1.9805, -2.4102],
#          ...,
#          [-5.4648, -3.2500, -1.9336,  ..., -3.7832, -2.7188, -3.5625],
#          [-3.1699, -2.8477, -0.3269,  ..., -2.1895, -0.1003, -2.4961],
#          [-1.9023, -1.9551,  0.2732,  ..., -3.1602,  1.0996, -1.7646]]],
#        device='cuda:2', dtype=torch.float16), 'loss': tensor(5.9306, device='cuda:2'), 'hidden_states': None}
#
# -5.930598258972168
# -5.930598258972168
# {'logits': tensor([[[-4.2539, -3.5352, -2.0918,  ..., -2.9727, -3.1465, -2.1172],
#          [-3.9863, -2.1250, -0.1639,  ..., -4.7891, -5.0586, -2.1699],
#          [-3.6816, -0.6802, -0.9971,  ..., -2.7598,  1.3135, -0.5293],
#          ...,
#          [-0.9507, -2.2012, -1.7051,  ..., -2.7910, -1.3965, -4.0156],
#          [-1.6826, -0.8643, -1.9170,  ..., -1.2920, -3.8887, -5.1289],
#          [-2.4785,  0.7139, -0.3826,  ..., -0.5679, -5.0508, -2.9004]]],
#        device='cuda:1', dtype=torch.float16), 'loss': tensor(5.9306, device='cuda:1'), 'hidden_states': None}
# -5.930598258972168
# {'logits': tensor([[[-3.2754, -3.7324, -4.8281,  ..., -6.2188, -6.3711, -6.3164],
#          [-6.9883, -5.0938, -4.2578,  ..., -4.0312, -4.2148, -4.0156],
#          [-5.7695, -2.1406, -3.1094,  ..., -0.8188, -0.7402, -0.6943],
#          ...,
#          [-4.6953, -1.8027,  1.5508,  ..., -4.1211, -3.9844, -4.2109],
#          [-1.9365, -0.4124,  0.4866,  ..., -2.6152, -2.1562, -2.4512],
#          [-3.3711,  0.7832,  0.0413,  ..., -3.4414, -3.5059, -3.3672]]],
#        device='cuda:3', dtype=torch.float16), 'loss': tensor(5.9306, device='cuda:3'), 'hidden_states': None}
# -5.930598258972168
# {'logits': tensor([[[-4.2539, -3.5352, -2.0918,  ..., -2.9727, -3.1465, -2.1172],
#          [-3.9863, -2.1250, -0.1639,  ..., -4.7891, -5.0586, -2.1699],
#          [-3.6816, -0.6802, -0.9971,  ..., -2.7598,  1.3135, -0.5293],
#          ...,
#          [-0.9507, -2.2012, -1.7051,  ..., -2.7910, -1.3965, -4.0156],
#          [-1.6826, -0.8643, -1.9170,  ..., -1.2920, -3.8887, -5.1289],
#          [-2.4785,  0.7139, -0.3826,  ..., -0.5679, -5.0508, -2.9004]]],
#        device='cuda:1', dtype=torch.float16), 'loss': tensor(5.2953, device='cuda:1'), 'hidden_states': None}
# -5.295323371887207
# {'logits': tensor([[[-6.5078, -6.5898,  2.3809,  ..., -1.6826, -4.5156, -1.6621],
#          [-4.1133, -3.8965,  1.8359,  ..., -2.4648, -6.0195, -1.9561],
#          [-0.6523, -0.4373,  3.7012,  ..., -1.7725, -3.9590,  0.1041],
#          ...,
#          [-4.0039, -3.7402,  4.4375,  ...,  0.4321,  0.1111, -1.3213],
#          [-2.5449, -2.2012,  8.6172,  ..., -1.3877, -3.0020, -3.7363],
#          [-3.2871, -3.1738,  9.0703,  ..., -1.2139, -4.0312, -1.1680]]],
#        device='cuda:0', dtype=torch.float16), 'loss': tensor(5.2953, device='cuda:0'), 'hidden_states': None}
# -5.295323371887207
# {'logits': tensor([[[-3.2754, -3.7324, -4.8281,  ..., -6.2188, -6.3711, -6.3164],
#          [-6.9883, -5.0938, -4.2578,  ..., -4.0312, -4.2148, -4.0156],
#          [-5.7695, -2.1406, -3.1094,  ..., -0.8188, -0.7402, -0.6943],
#          ...,
#          [-4.6953, -1.8027,  1.5508,  ..., -4.1211, -3.9844, -4.2109],
#          [-1.9365, -0.4124,  0.4866,  ..., -2.6152, -2.1562, -2.4512],
#          [-3.3711,  0.7832,  0.0413,  ..., -3.4414, -3.5059, -3.3672]]],
#        device='cuda:3', dtype=torch.float16), 'loss': tensor(5.2953, device='cuda:3'), 'hidden_states': None}
# -5.295323371887207
# {'logits': tensor([[[-5.1055, -5.5977, -3.4102,  ..., -5.1133, -4.7539, -4.4922],
#          [-5.2461, -3.8965, -1.0645,  ..., -4.4766, -5.5820, -5.8359],
#          [-3.5137, -1.2979, -1.4648,  ..., -2.5410, -1.9805, -2.4102],
#          ...,
#          [-5.4648, -3.2500, -1.9336,  ..., -3.7832, -2.7188, -3.5625],
#          [-3.1699, -2.8477, -0.3269,  ..., -2.1895, -0.1003, -2.4961],
#          [-1.9023, -1.9551,  0.2732,  ..., -3.1602,  1.0996, -1.7646]]],
#        device='cuda:2', dtype=torch.float16), 'loss': tensor(5.2953, device='cuda:2'), 'hidden_states': None}

def score(model, tokenizer, device, srcs, tgts, prompt_text, batch_size):
    """ Score a batch of examples """

    def trunk_input(inputs, outputs, reduce_seq, max_length):
        input_ids = tokenizer.encode(inputs)[1:-1]
        output_ids = tokenizer.encode(outputs)[1:-1]
        reduce_seq_ids = tokenizer.encode(reduce_seq)[1:-1]
        total_len = len(input_ids) + len(output_ids)
        if total_len > max_length:
            del_len = len(input_ids) + len(output_ids) - max_length
            reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
            reduce_seq = tokenizer.decode(reduce_seq_ids[1:-1])
        return reduce_seq

    score_list = []
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        print('process:' + str(i) + '/' + str(len(srcs)))
        new_src = trunk_input(src, tgt, src, max_length=2000)
        src = new_src
        text = src + prompt_text + tgt
        if i < 1:
            print('text: ', text)
            print('tgt: ', tgt)

        try:
            loss, logits, hidden_states = calculate_outputs(model, tokenizer, device, text, tgt)
            score = -loss
            score_list.append(score)
            print('score: ', score)
        except RuntimeError:
            # traceback.print_exc()
            print('input_ids: ', input_ids)
            print('output_ids: ', output_ids)
            print(f'source: {src}')
            print(f'target: {tgt}')
            # exit(0)
    return score_list
