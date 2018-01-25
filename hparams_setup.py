import tensorflow as tf

# Working HParams:
# num_units < ?, batch_size < 500, attention = True, max_len = 500
# num_units = 50, batch_size = 100, attention = True, max_len = 500 (verified)
# num_units = 100, batch_size = 100, attention = True, max_len = 500


def get_hparams():
    PROJECT_DIR = '/home/nave01314/IdeaProjects/tf-nmt/'

    hparams = tf.contrib.training.HParams(model_dir=PROJECT_DIR+'ckpts/',
                                          data_dir=PROJECT_DIR+'data/',

                                          train_print_freq=10,
                                          eval_print_freq=100,
                                          infer_print_freq=500,
                                          eval_max_printouts=5,
                                          infer_max_printouts=10,

                                          epochs=2000,
                                          l_rate=0.001,
                                          num_units=100,
                                          batch_size=300,
                                          max_gradient_norm=5.0,
                                          attention=True,

                                          src_vsize=27,                       # A-Z + pad
                                          tgt_vsize=11,                       # 8 + pad + sos + eos
                                          src_emsize=15,
                                          tgt_emsize=10,

                                          sos=1,
                                          eos=2,
                                          src_pad=0,
                                          tgt_pad=0,

                                          graph_seed=0,
                                          shuffle_seed=0,
                                          shuffle_buffer_size=10000,
                                          num_buckets=10,                     # 1 for no buckets
                                          max_len=500,                        # Largest is 5037
                                          dataset_max_size=1000000,
                                          max_weight=1.0,
                                          delta_weight=1.0,
                                          min_weight=0.1)

    return hparams
