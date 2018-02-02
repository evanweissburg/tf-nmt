import tensorflow as tf
import os


def get_hparams():
    project_dir = '/home/nave01314/IdeaProjects/tf-nmt/'

    hparams = tf.contrib.training.HParams(model_dir=os.path.join(project_dir, 'ckpts/'),
                                          data_dir=os.path.join(project_dir, 'data/'),
                                          log_dir=os.path.join(project_dir, 'logs/'),

                                          train_log_freq=10,
                                          eval_log_freq=50,
                                          infer_log_freq=100,
                                          eval_max_printouts=5,
                                          infer_max_printouts=10,

                                          epochs=2000,
                                          l_rate=0.001,
                                          num_units=100,
                                          batch_size=150,
                                          max_gradient_norm=5.0,
                                          attention=True,
                                          beam_search=True,
                                          beam_width=10,                       # Num top K preds to keep at timestep
                                          length_penalty_weight=0.0,          # Penalize length (disabled with 0.0)
                                          bidir_encoder=True,
                                          num_layers=2,                       # Must be even if bidirectional is enabled

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
                                          delta_weight=0.3,
                                          min_weight=0.1)

    return hparams
