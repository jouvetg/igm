
# from igm.processes.utils import *

# def params_pretraining(parser):
 
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default="surflib3d_shape_100",
#         help="Directory of the data of the glacier catalogu",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=1,
#         help="Batch size",
#     )
#     parser.add_argument(
#         "--freq_test",
#         type=int,
#         default=20,
#         help="Frequence of the test",
#     )
#     parser.add_argument(
#         "--train_iceflow_emulator_restart_lr",
#         type=int,
#         default=2500,
#         help="Restart frequency for the learning rate",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=5000,
#         help="Number of epochs",
#     )

#     parser.add_argument(
#         "--min_arrhenius",
#         type=float,
#         default=5,
#         help="Minium Arrhenius factor",
#     )
#     parser.add_argument(
#         "--max_arrhenius",
#         type=float,
#         default=151,
#         help="Maximum Arrhenius factor",
#     )
#     parser.add_argument(
#         "--min_slidingco",
#         type=float,
#         default=0,
#         help="Minimum sliding coefficient",
#     )
#     parser.add_argument(
#         "--max_slidingco",
#         type=float,
#         default=20000,
#         help="Maximum sliding coefficient",
#     )
#     parser.add_argument(
#         "--min_coarsen",
#         type=int,
#         default=0,
#         help="Minimum coarsening factor",
#     )
#     parser.add_argument(
#         "--max_coarsen",
#         type=int,
#         default=2,
#         help="Maximum coarsening factor",
#     )

#     parser.add_argument(
#         "--soft_begining",
#         type=int,
#         default=500,
#         help="soft_begining, if 0 explore all parameters btwe min and max, otherwise, \
#               only explore from this iteration while keeping mid-value fir the first it.",
#     )