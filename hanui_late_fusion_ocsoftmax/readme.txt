1. train: python discriminator_train_fmap.py --checkpoint_path fmap_mrd_half_ch_ocsoftmax_large
2. eval: python discriminator_evaluate_fmap.py --checkpoint_path outdir/fmap_mrd_half_ch_ocsoftmax_large/chkpt/ --name best_model.ckpt-438_d.pt
3. eval_ctrsvdd:python ctr_evaluate_layer_band.py --checkpoint_path outdir/fmap_mrd_half_ch_ocsoftmax_large/chkpt/ --name best_model.ckpt-438_d.pt
