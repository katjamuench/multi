class SPRPolicy(FeedForwardPolicy):
    """
    Custom policy. Exactly the same as MlpPolicy but with different NN configuration
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        self.params: Params = _kwargs['params']
        pi_layers = self.params.agent_config['pi_nn']
        vf_layers = self.params.agent_config['vf_nn']
        activ_function_name = self.params.agent_config['nn_activ']
        activ_function = eval(activ_function_name)
        net_arch = [dict(vf=vf_layers, pi=pi_layers)]
        super(SPRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        net_arch=net_arch, act_fun=activ_function, feature_extraction="spr", **_kwargs)

