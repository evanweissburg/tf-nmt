import tensorflow as tf


def get_hparams():
    PROJECT_DIR = '/home/nave01314/IdeaProjects/tf-nmt/'

    hparams = tf.contrib.training.HParams(model_dir=PROJECT_DIR+'ckpts/',
                                          data_dir=PROJECT_DIR+'data/',

                                          train_print_freq=10,
                                          eval_print_freq=100,
                                          infer_print_freq=100,
                                          eval_max_printouts=100,
                                          infer_max_printouts=100,

                                          epochs=2000,
                                          l_rate=0.001,
                                          num_units=10,
                                          batch_size=15,
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
                                          dataset_max_size=1000000)

    return hparams
