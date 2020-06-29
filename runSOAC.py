import torch
import os
import logging
import gym
import numpy as np
import random
import argparse
import mujoco_py

from SOACagent import SOACTrainer
from SOACevaluate import SOACTask
from SOACwrapper import NormalizedBoxEnv
from SOACnet import get_net
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--GPU', type=str, default="cuda:0", help='bool')
    parser.add_argument('--env',
                        type=str,
                        default='Ant-v2',
                        help='environment')
    parser.add_argument('--seed', type=int, default=0, help='environment')
    parser.add_argument('--MIweight',
                        type=float,
                        default=0,
                        help='environment')
    parser.add_argument('--length',
                        type=int,
                        default=1000000,
                        help='environment')

    args = parser.parse_args()
    return args


def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def experiment(args, device):

    env = gym.make(args.env)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    option_dim = 4

    main_name = f"SOAC_{args.env}_MIweight_{args.MIweight}"
    exp_name = f"{main_name}/seed_{args.seed}"
    log_path = f"{exp_name}/logfile.log"
    csv_path = f"{exp_name}/reward.csv"
    load_path = f"{exp_name}/para/"
    board_path = f"runs/{main_name}"
    writer = SummaryWriter(log_dir=board_path)

    for dir in [main_name, exp_name, load_path]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    logging = logger_config(log_path=log_path, logging_name='fly')
    logging.info("Running Urban Planning")
    logging.info(f"args: {args}")
    '''if args.env == "Walker2d_v2":
        MI_para = 0.3
    else:
        MI_para = 1  # tried Hopper-v2 HalfCheetah-v2 Ant-v2'''

    MI_para = args.MIweight

    logging.info(f'arg: {args}')
    hiddenlayer = [256, 256]

    qf1, qf2, qf1_target, qf2_target, u1, u2, u1_target, u2_target, policy, beta_net, option_pi_net = get_net(
        obs_dim, action_dim, option_dim, hiddenlayer)

    SOAC_trainer = SOACTrainer(
        policy,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        u1,
        u2,
        u1_target,
        u2_target,
        beta_net,
        option_pi_net,
        device,
        option_dim,
        obs_dim,
        writer=writer,
        logger=logging,
        IMpara=MI_para,
        load_path=load_path,
        length=args.length,
    )

    task = SOACTask(
        env,
        SOAC_trainer,
        obs_dim,
        action_dim,
        option_dim,
        device,
        csv_path,
        writer=writer,
        logging=logging,
        batch_size=256,
        length=args.length,
    )
    task.run()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    args = get_args()
    setup_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.GPU if use_cuda else "cpu")
    if use_cuda:
        if args.GPU == "cuda:0":
            torch.cuda.set_device(0)

        if args.GPU == "cuda:1":
            torch.cuda.set_device(1)

        if args.GPU == "cuda:2":
            torch.cuda.set_device(2)

        if args.GPU == "cuda:3":
            torch.cuda.set_device(3)

    experiment(args, device)
