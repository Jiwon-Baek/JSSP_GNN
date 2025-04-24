import argparse


class Configure:
    def __init__(self):
        self.vessl= 0 # whether to use vessl (0: False, 1:True)

        self.n_episode= 1000 # number of episodes
        self.load_model= 0 # whether to load the trained model
        self.model_path= None # model file path

        self.n_ships= 80 # number of ships in data
        self.data_path= "./input/configurations/v1/config (m=28).xlsx" # input data path

        self.use_gnn= 1 # whether to use gnn
        self.use_added_info= 1 # whether to use additional information
        self.encoding= "DG" # state encoding method
        self.restriction= 0 # whether to use restricted action space (0: False, 1:True)
        self.look_ahead= 3 # number of operations included in states
        self.embed_dim= 128 # node embedding dimension
        self.num_heads= 4 # multi-head attention in HGT layers
        self.num_HGT_layers= 2 # number of HGT layers
        self.num_actor_layers= 2 # number of actor layers
        self.num_critic_layers= 2 # number of critic layers
        self.lr= 0.00005 # learning rate
        self.lr_decay= 1.0 # learning rate decay ratio
        self.lr_step= 2000 # step size to reduce learning rate
        self.gamma= 0.98 # discount ratio
        self.lmbda= 0.95 # GAE parameter
        self.eps_clip= 0.2 # clipping paramter
        self.K_epoch= 5 # optimization epoch
        self.T_horizon= 10 # the number of steps to obtain samples
        self.P_coeff= 1 # coefficient for policy loss
        self.V_coeff= 0.5 # coefficient for value loss
        self.E_coeff= 0.01 # coefficient for entropy loss

        self.w_delay= 0.0 # weight for minimizing delays
        self.w_move= 1.0 # weight for minimizing the number of ship movements
        self.w_priority= 1.0 # weight for maximizing the efficiency

        self.eval_every= 100 # Evaluate every x episodes
        self.save_every= 1000 # Save a model every x episodes
        self.new_instance_every= 10 # Generate new scenarios every x episodes

        self.val_dir= "./input/validation/v1/28-80/" # directory where the validation data are stored
