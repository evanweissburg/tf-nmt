import tensorflow as tf
import os


def get_hparams():
    project_dir = '/media/nave01314/Storage/IdeaProjects/tf-nmt/'

    hparams = tf.contrib.training.HParams(model_dir=os.path.join(project_dir, 'ckpts/'),
                                          data_dir=os.path.join(project_dir, 'dataset/'),

                                          test_split_rate=10,
                                          validate_split_rate=5,

                                          train_log_freq=10,
                                          test_log_freq=50,
                                          test2_log_freq=100,
                                          validate_log_freq=10000000,
                                          test_max_printouts=5,
                                          infer_max_printouts=10,

                                          num_train_steps=400000,
                                          l_rate=0.0001,
                                          num_units=150,
                                          batch_size=100,
                                          max_gradient_norm=5.0,
                                          attention=True,
                                          beam_search=True,
                                          beam_width=10,                      # Num top K preds to keep at timestep
                                          length_penalty_weight=5.0,          # Penalize length (disabled with 0.0)
                                          bidir_encoder=True,
                                          num_layers=2,                       # Must be even if bidirectional is enabled

                                          src_vsize=25,                       # FASTA + eos - only used for embedding
                                          tgt_vsize=10,                       # 8 + sos + eos - only used for embedding
                                          src_emsize=15,
                                          tgt_emsize=10,

                                          src_eos='/s',                     # Also used for padding
                                          tgt_sos='s',
                                          tgt_eos='/s',                     # Also used for padding

                                          graph_seed=0,
                                          shuffle_seed=3,
                                          shuffle_buffer_size=10000,
                                          num_buckets=10,                     # 1 for no buckets
                                          max_len=500,                        # Largest is 5037
                                          dataset_max_size=1000000,
                                          max_weight=1.0,
                                          delta_weight=0.3,
                                          min_weight=0.1,

                                          fragment_radius=6,
                                          fragment_jump=1
                                          )

    return hparams
