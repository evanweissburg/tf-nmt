class HParams:
    def __init__(self, model_dir, l_rate, num_units, batch_size, max_gradient_norm, src_vsize, tgt_vsize, src_emsize, tgt_emsize, sos, eos, src_pad, tgt_pad, shuffle_seed, shuffle_buffer_size, num_buckets, max_len):
        self.model_dir = model_dir
        self.l_rate = l_rate
        self.num_units = num_units
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.src_vsize = src_vsize
        self.tgt_vsize = tgt_vsize
        self.src_emsize = src_emsize
        self.tgt_emsize = tgt_emsize
        self.sos = sos
        self.eos = eos
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_buckets = num_buckets
        self.max_len = max_len
