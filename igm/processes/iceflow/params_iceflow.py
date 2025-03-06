
# from igm.processes.utils import *

# def params_iceflow(parser):
 
#     parser.add_argument(
#         "--iflo_run_pretraining",
#         type=str2bool,
#         default=False,
#         help="Run the data assimilation scheme",
#     )

#     parser.add_argument(
#         "--iflo_run_data_assimilation",
#         type=str2bool,
#         default=False,
#         help="Run the data assimilation scheme",
#     )

#     # type of ice flow computations
#     parser.add_argument(
#         "--iflo_type",
#         type=str,
#         default="emulated",
#         help="Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver",
#     )

#     parser.add_argument(
#         "--iflo_pretrained_emulator",
#         type=str2bool,
#         default=True,
#         help="Do we take a pretrained emulator or start from scratch?",
#     )
#     parser.add_argument(
#         "--iflo_emulator",
#         type=str,
#         default="",
#         help="Directory path of the deep-learning pretrained ice flow model, take from the library if empty string",
#     )

#     # physical parameters
#     parser.add_argument(
#         "--iflo_init_slidingco",
#         type=float,
#         default=0.0464,
#         help="Initial sliding coefficient slidingco",
#     )
#     parser.add_argument(
#         "--iflo_init_arrhenius",
#         type=float,
#         default=78,
#         help="Initial arrhenius factor arrhenuis",
#     )
#     parser.add_argument(
#         "--iflo_enhancement_factor",
#         type=float,
#         default=1.0,
#         help="Enhancement factor multiying the arrhenius factor",
#     )
#     parser.add_argument(
#         "--iflo_regu_glen",
#         type=float,
#         default=10 ** (-5),
#         help="Regularization parameter for Glen's flow law",
#     )
#     parser.add_argument(
#         "--iflo_regu_weertman",
#         type=float,
#         default=10 ** (-10),
#         help="Regularization parameter for Weertman's sliding law",
#     )
#     parser.add_argument(
#         "--iflo_exp_glen",
#         type=float,
#         default=3,
#         help="Glen's flow law exponent",
#     )
#     parser.add_argument(
#         "--iflo_exp_weertman", type=float, default=3, help="Weertman's law exponent"
#     )
#     parser.add_argument(
#         "--iflo_gravity_cst",
#         type=float,
#         default=9.81,
#         help="Gravitational constant",
#     )
#     parser.add_argument(
#         "--iflo_ice_density",
#         type=float,
#         default=910,
#         help="Density of ice",
#     )
#     parser.add_argument(
#         "--iflo_new_friction_param",
#         type=str2bool,
#         default=True,
#         help="Sliding coeeficient (this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco as before)",
#     )
#     parser.add_argument(
#         "--iflo_save_model",
#         type=str2bool,
#         default=False,
#         help="save the iceflow emaultor at the end of the simulation",
#     )

#     # vertical discretization
#     parser.add_argument(
#         "--iflo_Nz",
#         type=int,
#         default=10,
#         help="Number of grid point for the vertical discretization",
#     )
#     parser.add_argument(
#         "--iflo_vert_spacing",
#         type=float,
#         default=4.0,
#         help="Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.",
#     )
#     parser.add_argument(
#         "--iflo_thr_ice_thk",
#         type=float,
#         default=0.1,
#         help="Threshold Ice thickness for computing strain rate",
#     )

#     # solver parameters
#     parser.add_argument(
#         "--iflo_solve_step_size",
#         type=float,
#         default=1,
#         help="Step size for the optimizer using when solving Blatter-Pattyn in solver mode",
#     )
#     parser.add_argument(
#         "--iflo_solve_nbitmax",
#         type=int,
#         default=100,
#         help="Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode",
#     )
#     parser.add_argument(
#         "--iflo_solve_stop_if_no_decrease",
#         type=str2bool,
#         default=True,
#         help="This permits to stop the solver if the energy does not decrease",
#     )

#     # emualtion parameters
#     parser.add_argument(
#         "--iflo_fieldin",
#         type=list,
#         default=["thk", "usurf", "arrhenius", "slidingco", "dX"],
#         help="Input fields of the iceflow emulator",
#     )
#     parser.add_argument(
#         "--iflo_dim_arrhenius",
#         type=int,
#         default=2,
#         help="Dimension of the arrhenius factor (horizontal 2D or 3D)",
#     )

#     parser.add_argument(
#         "--iflo_retrain_emulator_freq",
#         type=int,
#         default=10,
#         help="Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.",
#     )
#     parser.add_argument(
#         "--iflo_retrain_emulator_lr",
#         type=float,
#         default=0.00002,
#         help="Learning rate for the retraining of the emulator",
#     )
#     parser.add_argument(
#         "--iflo_retrain_emulator_nbit_init",
#         type=float,
#         default=1,
#         help="Number of iterations done at the first time step for the retraining of the emulator",
#     )
#     parser.add_argument(
#         "--iflo_retrain_emulator_nbit",
#         type=float,
#         default=1,
#         help="Number of iterations done at each time step for the retraining of the emulator",
#     )
#     parser.add_argument(
#         "--iflo_retrain_emulator_framesizemax",
#         type=float,
#         default=750,
#         help="Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded",
#     )
#     parser.add_argument(
#         "--iflo_multiple_window_size",
#         type=int,
#         default=0,
#         help="If a U-net, this force window size a multiple of 2**N",
#     )
#     parser.add_argument(
#         "--iflo_force_max_velbar",
#         type=float,
#         default=0,
#         help="This permits to artifically upper-bound velocities, active if > 0",
#     )

#     # CNN parameters
#     parser.add_argument(
#         "--iflo_network",
#         type=str,
#         default="cnn",
#         help="This is the type of network, it can be cnn or unet",
#     )
#     parser.add_argument(
#         "--iflo_activation",
#         type=str,
#         default="LeakyReLU",
#         help="Activation function, it can be lrelu, relu, tanh, sigmoid, etc.",
#     )
#     parser.add_argument(
#         "--iflo_nb_layers",
#         type=int,
#         default=16,
#         help="Number of layers in the CNN",
#     )
#     parser.add_argument(
#         "--iflo_nb_blocks",
#         type=int,
#         default=4,
#         help="Number of block layer in the U-net",
#     )
#     parser.add_argument(
#         "--iflo_nb_out_filter",
#         type=int,
#         default=32,
#         help="Number of output filters in the CNN",
#     )
#     parser.add_argument(
#         "--iflo_conv_ker_size",
#         type=int,
#         default=3,
#         help="Size of the convolution kernel",
#     )
#     parser.add_argument(
#         "--iflo_dropout_rate",
#         type=float,
#         default=0,
#         help="Dropout rate in the CNN",
#     )
#     parser.add_argument(
#         "--iflo_weight_initialization",
#         type=str,
#         default="glorot_uniform",
#         help="glorot_uniform, he_normal, lecun_normal",
#     )
#     parser.add_argument(
#         "--iflo_exclude_borders",
#         type=int,
#         default=0,
#         help="This is a quick fix of the border issue, other the physics informed emaulator shows zero velocity at the border",
#     )
    
#     parser.add_argument(
#         "--iflo_cf_eswn",
#         type=list,
#         default=[],
#         help="This forces calving front at the border of the domain in the side given in the list",
#     )
    
#     parser.add_argument(
#         "--iflo_cf_cond",
#         type=str2bool,
#         default=False,
#         help="This forces calving front at the border of the domain in the side given in the list",
#     )
    
#     parser.add_argument(
#         "--iflo_regu",
#         type=float,
#         default=0.0,
#         help="This regularizes the energy forcing ice flow to be smooth in the horizontal direction",
#     )
#     parser.add_argument(
#         "--iflo_min_sr",
#         type=float,
#         default=10**(-20),
#         help="Minimum strain rate",
#     )
#     parser.add_argument(
#         "--iflo_max_sr",
#         type=float,
#         default=10**(20),
#         help="Maximum strain rate",
#     )
#     parser.add_argument(
#         "--iflo_force_negative_gravitational_energy",
#         type=str2bool,
#         default=False,
#         help="Force energy gravitational term to be negative",
#     )

#     parser.add_argument(
#         "--iflo_optimizer_solver",
#         type=str,
#         default='Adam',
#         help="Tpe of Optimizer for the solver",
#     )
#     parser.add_argument(
#         "--iflo_optimizer_lbfgs",
#         type=str2bool,
#         default=False,
#         help="iflo_optimizer_lbfgs",
#     )
    
#     parser.add_argument(
#         "--iflo_optimizer_emulator",
#         type=str,
#         default='Adam',
#         help="Tpe of Optimizer for the emulator",
#     )
#     parser.add_argument(
#         "--iflo_save_cost_emulator",
#         type=str,
#         default=""
#     )
#     parser.add_argument(
#         "--iflo_save_cost_solver",
#         type=str,
#         default=""
#     )
#     parser.add_argument(
#         "--iflo_output_directory",
#         type=str,
#         default=""
#     )

