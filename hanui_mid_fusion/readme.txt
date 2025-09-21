1. train:  python discriminator_train_fmap.py --checkpoint_path fmap_half_ch_layer_band_8
2. eval: python discriminator_evaluate_fmap.py --checkpoint_path outdir/fmap_half_ch_layer_band_8/chkpt/ --name best_model.ckpt-481_d.pt
3: eval_ctrsvdd: python ctr_evaluate_layer_band.py --checkpoint_path outdir/fmap_half_ch_layer_band_8/chkpt/ --name best_model.ckpt-481_d.pt
