#### 2D TRANSFER LEARNING ####
sim_tag = "rect2x2_transfer"
ids = [("rect2x2_reference", "2mx2m reference"),
       ("rect2x2_freeze_BN_0_1_TN_0_batch_size_600_src_pos_3ppw_target_2D", "2mx2m target (60% samples)")]

sim_tag = "Lshape_2_5x2_5_transfer"
ids = [("Lshape2_5x2_5_reference", "L-shape 2.5mx2.5m reference"),
       ("Lshape2_5x2_5_srcpos_5ppw_freeze_BN_0_1_TN_0_batch_size_600_target_2D", "L-shape 2.5mx2.5m target"),
       ("Lshape2_5x2_5_freeze_BN_0_1_TN_0_batch_size_600_src_pos_3ppw_target_2D", "L-shape 2.5mx2.5m target (60% samples)")]

sim_tag = "rect3x3_furn_transfer"
ids = [("rect_furn_3x3_BS_600_reference", "3mx3m 'furnished' reference (TN BS 600)"),
       ("rect3x3_furn_freeze_BN_0_1_TN_0_batch_size_600_target_2D", "3mx3m 'furnished' target"),
       ("rect3x3_furn_freeze_BN_0_1_TN_0_batch_size_600_src_pos_3ppw_target_2D", "3mx3m 'furnished' target (60% samples)")]

#### SENSIBILITY ANALYSIS ####
ids = [("1_baseline_tanh_mlp", "tanh MLP"),
       ("1_baseline_relu_mlp", "relu MLP"),
       ("1_baseline_sine_mlp", "sine MLP"),
       ("1_baseline_tanh_mlp_pos", "tanh MLP Positional"),
       ("1_baseline_tanh_mlp_gaussian", "tanh MLP Gaussian"),
       ("1_baseline_tanh_mlp_es", "tanh MLP ES"),
       ("1_baseline_relu_mlp_pos", "relu MLP Positional"),
       ("1_baseline_relu_mlp_gaussian", "relu MLP Gaussian"),
       ("1_baseline_relu_mlp_es", "relu MLP ES")
    #    ("1_baseline_sine_mlp_pos", "sine MLP Positional"),
    #    ("1_baseline_sine_mlp_gaussian", "sine MLP Gaussian"),
    #    ("1_baseline_sine_mlp_es", "sine MLP ES")
       ]

ids = [("1_baseline_tanh_mod_mlp_nofeat", "tanh Mod-MLP"),
       ("1_baseline_relu_mod_mlp_nofeat", "relu Mod-MLP"),
       ("1_baseline_sine_mod_mlp_nofeat", "sine Mod-MLP"),
       ("1_baseline_tanh_mod_mlp_pos", "tanh Mod-MLP Positional"),
       ("1_baseline_tanh_mod_mlp_gaussian", "tanh Mod-MLP Gaussian"),
       ("1_baseline_tanh_mod_mlp_es", "tanh Mod-MLP ES"),
       ("1_baseline_relu_mod_mlp_pos", "relu Mod-MLP Positional"),
       ("1_baseline_relu_mod_mlp_gaussian", "relu Mod-MLP Gaussian"),
       ("1_baseline_relu_mod_mlp_es", "relu Mod-MLP ES"),
       ("1_baseline_sine_mod_mlp_pos", "sine Mod-MLP Positional"),
       # ("1_baseline_sine_mod_mlp_gaussian", "sine Mod-MLP Gaussian"),
       # ("1_baseline_sine_mod_mlp_es", "sine Mod-MLP ES")
       ]


ids = [("4_baseline_batchsize_16_100", "BS 16/100"),
       ("4_baseline_batchsize_16_200", "BS 16/200"),
       ("4_baseline_batchsize_16_400", "BS 16/400"),
       ("4_baseline_batchsize_16_600", "BS 16/600"),
       ("4_baseline_batchsize_32_100", "BS 32/100"),
       ("4_baseline_batchsize_32_200", "BS 32/200"),
       ("4_baseline_batchsize_32_400", "BS 32/400"),
       ("4_baseline_batchsize_32_600", "BS 32/600"),
       ("4_baseline_batchsize_64_100", "BS 64/100"),
       ("4_baseline_batchsize_64_200", "BS 64/200"),
       ("4_baseline_batchsize_64_400", "BS 64/400"),
       ("4_baseline_batchsize_64_600", "BS 64/600"),
       ("4_baseline_batchsize_96_100", "BS 96/100"),
       ("4_baseline_batchsize_96_200", "BS 96/200"),
       ("4_baseline_batchsize_96_400", "BS 96/400"),
       ("4_baseline_batchsize_96_600", "BS 96/600")]

ids = [("5_baseline_2_512", "2/512"),
       ("5_baseline_2_1024", "2/1024"),
       ("5_baseline_2_2048", "2/2048"),
       ("5_baseline_3_512", "3/512"),
       ("5_baseline_3_1024", "3/1024"),
       ("5_baseline_3_2048", "3/2048"),
       ("5_baseline_4_512", "4/512"),
       ("5_baseline_4_1024", "4/1024"),
       ("5_baseline_4_2048", "4/2048"),
       ("5_baseline_5_512", "5/512"),
       ("5_baseline_5_1024", "5/1024"),
       ("5_baseline_5_2048", "5/2048")]

ids = [("6_baseline_bn2_tn2", "BN=2PPW TN=2PPW"),
       ("6_baseline_bn2_tn4", "BN=2PPW TN=4PPW"),
       ("6_baseline_bn2_tn6", "BN=2PPW TN=6PPW"),
       ("6_baseline_bn4_tn2", "BN=4PPW TN=2PPW"),
       ("6_baseline_bn4_tn4", "BN=4PPW TN=4PPW"),
       ("6_baseline_bn4_tn6", "BN=4PPW TN=6PPW")]

ids = [("6_baseline_bn2_tn6", "dx_src=5 dx=6PPW, dt=6PPW"),
       ("7_dxu2_dx6_dt4_srcdx2", "dx_src=2 dx=6PPW, dt=4PPW"),
       ("7_dxu2_dx6_dt2_srcdx2", "dx_src=2 dx=6PPW, dt=2PPW"),
       ("7_dxu2_dx6_dt4_srcdx3", "dx_src=3 dx=6PPW, dt=4PPW"),
       ("7_dxu2_dx6_dt4_srcdx4", "dx_src=4 dx=6PPW, dt=4PPW"),
       ("7_dxu2_dx6_dt4_srcdx5", "dx_src=5 dx=6PPW, dt=4PPW"),
       ("7_dxu2_dx6_dt2_srcdx5", "dx_src=5 dx=6PPW, dt=2PPW")]

ids = [("7_dxu2_dx6_dt2_srcdx2", "FNN reference"),
       ("resnet_3333_relu_ppw2_val_meshrand", "ResNet-{3,3,3,3}"),
       ("resnet_33333_relu_ppw2_val_meshrand", "ResNet-{3,3,3,3,3}"),
       ("resnet_33333_relu_ppw4_val_meshrand", "ResNet-{3,3,3,3,3} BN=4PPW")]